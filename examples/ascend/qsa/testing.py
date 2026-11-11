"""Generate QSA inputs, inject contract corners, and run a reference."""

import torch

from examples.ascend.qsa.kernel import HEAD_DIM, NUM_HEADS, NUM_KV_HEADS, PAD_M


def gen_indices(total_q, total_kv, top_k, fixed_budget=None, seed=0):
    """Random whole-block selections honoring the index contract."""
    torch.manual_seed(seed)
    num_blocks = total_kv // 4
    indices = torch.full((total_q, top_k), -1, dtype=torch.int32)
    if fixed_budget is not None:
        counts = torch.full((total_q,), fixed_budget, dtype=torch.int64)
    else:
        counts = torch.randint(1, top_k + 1, (total_q,))
    offsets = torch.arange(4, dtype=torch.int64)
    for t in range(total_q):
        n_blocks = min(int(counts[t]) // 4, num_blocks)
        blocks, _ = torch.randperm(num_blocks)[:n_blocks].sort()
        ids = blocks.unsqueeze(1) * 4 + offsets.unsqueeze(0)
        indices[t, : n_blocks * 4] = ids.reshape(-1).to(torch.int32)
    return indices


def split_indices(indices):
    """safe = clamped indices; valid = slot mask."""
    valid = (indices >= 0).to(torch.int32).contiguous()
    safe = indices.clamp_min(0).to(torch.int32).contiguous()
    return safe, valid


def pad_q_heads(q):
    """Regroup Q to (num_tokens, num_kv_heads, group, head_dim), zero-pad
    group to MP rows."""
    total_q, num_heads, head_dim = q.shape
    group = num_heads // NUM_KV_HEADS
    qp = q.reshape(total_q, NUM_KV_HEADS, group, head_dim)
    if group < PAD_M:
        pad = qp.new_zeros(total_q, NUM_KV_HEADS, PAD_M - group, head_dim)
        qp = torch.cat([qp, pad], dim=2)
    return qp.contiguous()


def ref_sparse_gqa(q, k, v, indices, sm_scale=None):
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group = num_heads // num_kv_heads
    scale = sm_scale if sm_scale is not None else head_dim**-0.5
    qf = q.float().view(total_q, num_kv_heads, group, head_dim)
    out = torch.zeros(total_q, num_heads, head_dim, dtype=torch.float32, device=q.device)
    for t in range(total_q):
        ids = indices[t][indices[t] >= 0].long()
        if ids.numel() == 0:
            continue
        kk = k[ids].float()
        vv = v[ids].float()
        s = torch.einsum("hgd,nhd->hgn", qf[t], kk) * scale
        p = s.softmax(dim=-1)
        out[t] = torch.einsum("hgn,nhd->hgd", p, vv).reshape(num_heads, head_dim)
    return out


def make_inputs(total_q, total_kv, top_k, seed=0, fixed_budget=None):
    torch.manual_seed(seed)
    q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    k = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    v = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    indices = gen_indices(total_q, total_kv, top_k, fixed_budget=fixed_budget, seed=seed).to("npu")
    return q, k, v, indices


def fill_contract_corners(indices, kv_length, top_k):
    """Cover the contract corners: a fully invalid query, a short leading
    run, and a tail quad ending at the KV boundary."""
    if indices.shape[0] > 0:
        indices[0] = -1
    if indices.shape[0] > 1:
        indices[1, :8] = -1
    if indices.shape[0] > 2:
        indices[2, top_k - 4 :] = torch.tensor([kv_length - 3, kv_length - 2, kv_length - 1, -1], dtype=torch.int32)


def run_qsa(kernel, q, k, v, indices):
    """Invoke the kernel on padded Q and clamped indices; return the output
    as (num_tokens, NUM_HEADS, HEAD_DIM)."""
    safe, valid = split_indices(indices)
    qp = pad_q_heads(q)
    group = NUM_HEADS // NUM_KV_HEADS
    out = kernel(qp, k, v, safe, valid)
    return out[:, :, :group, :].reshape(q.shape[0], NUM_HEADS, HEAD_DIM)
