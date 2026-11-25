"""Generate QSA inputs and run the fixed Qwen attention contract."""

import torch
import torch.nn.functional as F

from examples.ascend.qsa.kernel import (
    HEAD_DIM,
    LOGICAL_TOP_K,
    NUM_HEADS,
    NUM_KV_HEADS,
    PAD_M,
    PHYSICAL_TOP_K,
)


def gen_indices(total_q, total_kv, seed=0):
    """Generate whole four-token blocks in the fixed physical slot tensor."""
    assert total_q > 0 and total_kv >= 4
    torch.manual_seed(seed)
    num_blocks = total_kv // 4
    indices = torch.full((total_q, PHYSICAL_TOP_K), -1, dtype=torch.int32)
    counts = torch.full((total_q,), LOGICAL_TOP_K, dtype=torch.int64)
    offsets = torch.arange(4, dtype=torch.int64)
    for row in range(total_q):
        n_blocks = min(int(counts[row]) // 4, num_blocks)
        blocks, _ = torch.randperm(num_blocks)[:n_blocks].sort()
        ids = blocks.unsqueeze(1) * 4 + offsets.unsqueeze(0)
        indices[row, : n_blocks * 4] = ids.reshape(-1).to(torch.int32)
    return indices


def split_indices(indices):
    """Return safe GM indices and the valid-slot mask consumed by AscendC."""
    valid = (indices >= 0).to(torch.int32).contiguous()
    safe = indices.clamp_min(0).to(torch.int32).contiguous()
    return safe, valid


def pad_q_heads(q):
    """Regroup Q to KV groups and pad each 12-row group to 16 rows."""
    assert tuple(q.shape[1:]) == (NUM_HEADS, HEAD_DIM)
    grouped = q.reshape(q.shape[0], NUM_KV_HEADS, NUM_HEADS // NUM_KV_HEADS, HEAD_DIM)
    padding = grouped.new_zeros(q.shape[0], NUM_KV_HEADS, PAD_M - NUM_HEADS // NUM_KV_HEADS, HEAD_DIM)
    return torch.cat((grouped, padding), dim=2).contiguous()


def ref_sparse_gqa(q, k, v, indices):
    """Compute the BF16-input, FP32-accumulation reference result."""
    assert tuple(q.shape[1:]) == (NUM_HEADS, HEAD_DIM)
    assert tuple(k.shape[1:]) == (NUM_KV_HEADS, HEAD_DIM)
    assert tuple(v.shape) == tuple(k.shape)
    assert indices.shape == (q.shape[0], PHYSICAL_TOP_K)
    group = NUM_HEADS // NUM_KV_HEADS
    qf = q.float().view(q.shape[0], NUM_KV_HEADS, group, HEAD_DIM)
    out = torch.zeros_like(qf, dtype=torch.float32)
    for row in range(q.shape[0]):
        ids = indices[row][indices[row] >= 0].long()
        if ids.numel() == 0:
            continue
        scores = torch.einsum("hgd,nhd->hgn", qf[row], k[ids].float()) * (HEAD_DIM**-0.5)
        probs = scores.softmax(dim=-1)
        out[row] = torch.einsum("hgn,nhd->hgd", probs, v[ids].float())
    return out.reshape(q.shape[0], NUM_HEADS, HEAD_DIM)


def make_inputs(total_q, total_kv, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    k = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    indices = gen_indices(total_q, total_kv, seed=seed).to("npu")
    return q, k, v, indices


def fill_contract_corners(indices, kv_length):
    """Cover empty, short-prefix, and partial-tail selections."""
    assert indices.shape[1] == PHYSICAL_TOP_K
    assert kv_length >= 4
    if indices.shape[0] > 0:
        indices[0] = -1
    if indices.shape[0] > 1:
        indices[1, :8] = -1
    if indices.shape[0] > 2:
        indices[2, -4:] = torch.tensor(
            [kv_length - 3, kv_length - 2, kv_length - 1, -1],
            dtype=torch.int32,
            device=indices.device,
        )


def run_qsa(kernel, q, k, v, indices):
    """Invoke the kernel and return an unpadded [T, 24, 256] result."""
    safe, valid = split_indices(indices)
    q_padded = pad_q_heads(q)
    k_padded = F.pad(k, (0, 0, 0, 0, 0, 3)).contiguous()
    v_padded = F.pad(v, (0, 0, 0, 0, 0, 3)).contiguous()
    output = kernel(q_padded, k_padded, v_padded, safe, valid)
    return output[:, :, : NUM_HEADS // NUM_KV_HEADS, :].reshape(q.shape[0], NUM_HEADS, HEAD_DIM)
