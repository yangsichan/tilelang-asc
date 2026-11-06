"""GLM-5.3 DSA (DeepSeek Sparse Attention) forward on Ascend 950 (A5).

Static attributes: ``num_heads = 64``, ``head_dim = 256``, softmax scale
``0.0625`` (``1 / sqrt(head_dim)``), ``index_topk = 2048``, ``dropout = 0``.
Batch, query and KV lengths are dynamic (each shape is specialised at compile
time, as in the rest of the Ascend examples).

Layouts::

    Q        [B, 64, sq, 256]  bfloat16
    K        [B, 64, T,  256]  bfloat16
    V        [B, 64, T,  256]  bfloat16
    indices  [B, sq, Ksel]     int32, Ksel = min(index_topk, T)
    mask     [B, 1, sq, T]     uint8 (torch.bool), nonzero = participates
    O        [B, sq, 64, 256]  bfloat16

``indices`` contract: produced by
``index_scores.topk(min(index_topk, T), dim=-1).indices``.  Every slot holds a
KV position in ``[0, T)``; there is no ``-1`` padding.  ``Ksel`` need not be a
multiple of the 128-wide index tile, so the last tile may be partial.

``mask`` contract: boolean, broadcast over heads and shared by all heads of a
query.  A selected KV position whose byte is zero is dropped from the softmax.
A query whose whole selection is dropped emits zeros instead of NaN.

Kernel shape: unlike the shared-KV MLA example, K and V are per head here, so a
task is one ``(batch, query, head-group)`` tuple and the Cube M dimension carries a single
query row.  M is padded to 16 rows, each holding a copy of the same query
vector; the padding is pure shape compliance (Cube M granularity is 16) and does
not change the result.

Performance note: M contains 16 copies of one query, not 16 useful queries.
The vector mask path widens an aligned raw row once per task and gathers flags
per KV tile. Other lengths retain the scalar path. Further performance changes
must preserve the mask contract and be measured against the same GBSA boundary.
"""

import argparse
from dataclasses import dataclass

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.carver.arch.driver import get_num_cube_cores
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_major_k_layout
from tilelang.profiler import do_bench
from tvm import tirx


NUM_HEADS = 64
HEAD_DIM = 256
SOFTMAX_SCALE = 0.0625
INDEX_TOPK = 2048

DTYPE = "bfloat16"
ACCUM_DTYPE = "float32"

# Replaces a masked score outright.  exp(MASK_VALUE - row_max) underflows to
# exactly zero, so a masked lane contributes nothing to either the softmax sum
# or the PV GEMM.
MASK_VALUE = -3.402823e38

# Floor applied to a row maximum.  A row whose lanes are all masked would give
# exp(0) = 1 for every lane; flooring the maximum keeps them underflowing to 0.
MAX_FLOOR = -1.0e30


@dataclass(frozen=True)
class FwdTiling:
    # The SIMD probability packing path is built around a 128-wide index tile.
    block_topk: int = 128
    heads_per_task: int = 4


def _copy_row_pair(dst, src, row_stride):
    """Gather two rows in one GM->UB DMA whose source rows are ``row_stride`` apart.

    Only an ascending pair is expressible: the first burst lands on the smaller
    KV position and the second on the larger one, regardless of the order the two
    indices appear in ``indices``.  ``mask_penalty`` reproduces that pairing.
    """
    row_bytes = HEAD_DIM * 2
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.ascend_copy_gm_to_ubuf"),
        dst,
        src,
        0,
        2,
        row_bytes,
        0,
        0,
        0,
        0,
        row_stride * row_bytes,
        row_bytes,
    )


@T.macro
def _expand_mask_row(mask_row_ub, mask_words_ub, kv_len):
    """Widen the raw uint8 mask row to uint32 using only vector loads/stores."""
    with T.SimdVF():
        S.mem_bar("VST_VLD")
        full32 = S.pset(32, "PAT_ALL")
        for chunk in T.serial(kv_len // 256):
            raw = S.vld(mask_row_ub[chunk * 256], "NORM")
            packed = T.reinterpret(raw, "uint32x64")
            byte_mask = S.vdup(T.uint32(255), "uint32", full32)
            byte0 = S.vand(packed, byte_mask, full32)
            byte1 = S.vand(S.vshrs(packed, 8, full32), byte_mask, full32)
            byte2 = S.vand(S.vshrs(packed, 16, full32), byte_mask, full32)
            byte3 = S.vand(S.vshrs(packed, 24, full32), byte_mask, full32)
            even_low, even_high = S.vintlv(byte0, byte2)
            odd_low, odd_high = S.vintlv(byte1, byte3)
            S.vst2(mask_words_ub[chunk * 256], even_low, odd_low)
            S.vst2(mask_words_ub[chunk * 256 + 128], even_high, odd_high)



@T.macro
def _vector_mask_penalty(indices_ub, mask_words_ub, penalty_ub, kv_len):
    """Produce this tile's 128 flags, matching the ascending paired K/V DMA."""
    with T.SimdVF():
        S.mem_bar("VST_VLD")
        full = S.pset(32, "PAT_ALL")
        even, odd = S.vld2(indices_ub[0], "DINTLV_B32")
        first = T.reinterpret(S.vmin(even, odd, full), "uint32x64")
        second = T.reinterpret(S.vmax(even, odd, full), "uint32x64")
        keep_first = S.vgather2(T.access_ptr(mask_words_ub[0], "r", extent=kv_len), first, full)
        keep_second = S.vgather2(T.access_ptr(mask_words_ub[0], "r", extent=kv_len), second, full)
        zero = S.vdup(T.float32(0.0), ACCUM_DTYPE, full)
        one = S.vdup(T.float32(1.0), ACCUM_DTYPE, full)
        first_flag = S.vsel(zero, one, S.vcmps(keep_first, T.uint32(0), full, op="ne"))
        second_flag = S.vsel(zero, one, S.vcmps(keep_second, T.uint32(0), full, op="ne"))
        S.vst2(penalty_ub[0], first_flag, second_flag)


def glm_dsa_fwd(
    batch_size,
    seq_len,
    kv_len,
    top_k,
    *,
    num_blocks=None,
    tiling=None,
):
    tiling = tiling or FwdTiling()

    # Capture plain compile-time scalars in the hygienic T.macros below.
    BLOCK_TOP_K = tiling.block_topk
    HEADS_PER_TASK = tiling.heads_per_task
    assert HEADS_PER_TASK in (1, 2, 4, 8, 16, 32, 64)
    HEAD_GROUPS = NUM_HEADS // HEADS_PER_TASK
    CHUNK_ROWS = 16
    PIPELINE_STAGES = 2
    M_PAD = 16  # Cube M granularity; every row carries the same query vector.
    ROWS_PER_AIV = M_PAD // 2
    KV_ROWS_PER_AIV = BLOCK_TOP_K // 2
    VECTOR_LANES = 64  # Ascend fp32 SIMD lane count

    assert batch_size > 0 and seq_len > 0 and kv_len > 0, "batch_size, seq_len and kv_len must be positive"
    assert top_k == min(INDEX_TOPK, kv_len), "top_k must be min(index_topk, kv_len)"
    # The softmax handles two fp32x64 vectors and packs one bf16x128 vector.
    assert BLOCK_TOP_K == 128, "The current SIMD softmax packing requires block_topk == 128"

    # One attention_mask row costs kv_len bytes of UB and its DMA source origin
    # is the row start, so bulk staging is only legal when that origin and the
    # row length are both 32B aligned and the row fits the window.  Any other
    # kv_len reads the mask scalar from GM instead: same result, slower pass.
    MASK_STAGE_BYTES = 32 * 1024
    MASK_STAGED = (kv_len % 32 == 0) and (kv_len <= MASK_STAGE_BYTES)
    # Bound the widened row's UB footprint; other lengths keep the scalar path.
    VECTOR_MASK = MASK_STAGED and kv_len % 256 == 0 and kv_len <= 8192
    # K/V always read their indices from GM, independent of the mask path.

    NUM_BLOCKS = (
        pick_num_blocks(seq_len, batch_size, heads_per_task=HEADS_PER_TASK)
        if num_blocks is None else num_blocks
    )
    assert NUM_BLOCKS > 0, "num_blocks must be positive"
    NUM_TASKS = batch_size * seq_len * HEAD_GROUPS
    assert NUM_BLOCKS <= NUM_TASKS, "num_blocks exceeds the number of query-head tasks"
    NUM_TOPK_BLOCKS = (top_k + BLOCK_TOP_K - 1) // BLOCK_TOP_K

    # -- shared T.macros --------------------------------------------------

    def read_index(Indices, batch_id, query_id, block_base, lane):
        """Read a sparse index directly from GM."""
        return T.int32(Indices[batch_id, query_id, block_base + lane])

    def read_keep(Mask, mask_row_ub, batch_id, query_id, position):
        """Whether a KV position participates, staged in UB or read scalar."""
        if MASK_STAGED:
            return mask_row_ub[position] != T.uint8(0)
        return Mask[batch_id, 0, query_id, position] != T.uint8(0)

    @T.macro
    def gather_rows(src, batch_id, query_id, head, block_base, valid_count, sid, indices_src, raw_ub, nz_ub, l1):
        """Each AIV gathers one half of the sparse tile.

        Lanes at or past ``valid_count`` re-gather the last valid position rather
        than being skipped.  Their probabilities are zero, but the row still
        feeds the PV MMA: an unwritten row comes from whatever the buffer held
        before -- uninitialised UB on the first tile -- and one Inf/NaN there
        turns ``0 * V`` into NaN and poisons the whole output.  Repeating a real
        index keeps every L1 row finite, and ``mask_penalty`` masks those lanes
        out by position.
        """
        for chunk in T.Serial(KV_ROWS_PER_AIV // CHUNK_ROWS):
            row_start = sid * KV_ROWS_PER_AIV + chunk * CHUNK_ROWS
            for pair in T.Serial(CHUNK_ROWS // 2):
                lane0 = T.min(row_start + 2 * pair, valid_count - 1)
                lane1 = T.min(row_start + 2 * pair + 1, valid_count - 1)
                index0 = read_index(indices_src, batch_id, query_id, block_base, lane0)
                index1 = read_index(indices_src, batch_id, query_id, block_base, lane1)
                T.assume(index0 >= 0)
                T.assume(index0 < kv_len)
                T.assume(index1 >= 0)
                T.assume(index1 < kv_len)
                if index0 == index1:
                    T.copy(src[batch_id, head, index0, :], raw_ub[2 * pair, :])
                    T.copy(src[batch_id, head, index1, :], raw_ub[2 * pair + 1, :])
                else:
                    first_index = T.min(index0, index1)
                    row_stride = T.max(index0, index1) - first_index
                    _copy_row_pair(
                        T.access_ptr(raw_ub[2 * pair, 0], "w"),
                        T.access_ptr(src[batch_id, head, first_index, 0], "r"),
                        row_stride,
                    )

            T.copy(raw_ub, nz_ub[:CHUNK_ROWS, :])
            T.copy(nz_ub[:CHUNK_ROWS, :], l1[row_start : row_start + CHUNK_ROWS, :])

    @T.macro
    def mask_penalty(Mask, mask_row_ub, penalty_ub, valid_count, batch_id, query_id, block_base, indices_src):
        """Fold attention_mask and the partial-tile tail into one keep flag.

        ``penalty_ub[i]`` becomes 0.0 for a participating slot and 1.0
        otherwise; ``softmax_pack`` turns that into a lane predicate.  The mask
        is indexed by KV position. This fallback keeps scalar lookups; aligned
        full-tile lengths within the UB budget use _vector_mask_penalty instead.

        Every AIV covers the whole ``BLOCK_TOP_K`` block: the dual copy splits
        ``qk_acc_l0c`` along M only, so each AIV's ``S_ub`` rows still span all
        ``BLOCK_TOP_K`` KV columns.  The gather divides the rows between the two
        AIVs, but the flag vector must not -- a half-filled vector would mask
        the other AIV's columns as well as its own.

        The lane order must match ``gather_rows``: a paired DMA lands the smaller
        KV position in ``lane0``, so the mask is looked up through the same
        min/max pairing.  Looking it up by the raw ``indices`` order would drop
        the wrong lane whenever two adjacent indices are not ascending.  Lanes at
        or past ``valid_count`` are never written here, which leaves their
        penalty at 1.0 -- they are padding rows repeating a real KV position and
        must stay out of the softmax.
        """
        for lane in T.Serial(BLOCK_TOP_K):
            penalty_ub[lane] = T.float32(1.0)
        for chunk in T.Serial(BLOCK_TOP_K // CHUNK_ROWS):
            row_start = chunk * CHUNK_ROWS
            valid_rows = T.max(T.min(valid_count - row_start, CHUNK_ROWS), 0)
            for pair in T.Serial(valid_rows // 2):
                lane0 = row_start + 2 * pair
                index0 = read_index(indices_src, batch_id, query_id, block_base, lane0)
                index1 = read_index(indices_src, batch_id, query_id, block_base, lane0 + 1)
                index0 = T.max(T.min(index0, kv_len - 1), 0)
                index1 = T.max(T.min(index1, kv_len - 1), 0)
                if read_keep(Mask, mask_row_ub, batch_id, query_id, T.min(index0, index1)):
                    penalty_ub[lane0] = T.float32(0.0)
                if read_keep(Mask, mask_row_ub, batch_id, query_id, T.max(index0, index1)):
                    penalty_ub[lane0 + 1] = T.float32(0.0)
            if valid_rows % 2 != 0:
                tail_lane = row_start + valid_rows - 1
                tail_index = read_index(indices_src, batch_id, query_id, block_base, tail_lane)
                tail_index = T.max(T.min(tail_index, kv_len - 1), 0)
                if read_keep(Mask, mask_row_ub, batch_id, query_id, tail_index):
                    penalty_ub[tail_lane] = T.float32(0.0)

    @T.macro
    def softmax_pack(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, penalty_ub):
        with T.SimdVF(latency=735):
            S.mem_bar("VST_VLD")
            fp32_mask = S.pset(32, "PAT_ALL")
            scalar_mask = S.pset(32, "PAT_VL1")
            bf16_mask = S.pset(16, "PAT_VL128")
            mask_value = S.vdup(T.float32(MASK_VALUE), ACCUM_DTYPE, fp32_mask)
            p_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS_PER_AIV + 1, BLOCK_TOP_K),
                DTYPE,
            )
            nz_stride = T.int32(((ROWS_PER_AIV + 1) << 16) | 1)
            keep_lo = S.vcmps(S.vld(penalty_ub[0], "NORM"), T.float32(0.5), fp32_mask, op="lt")
            keep_hi = S.vcmps(S.vld(penalty_ub[VECTOR_LANES], "NORM"), T.float32(0.5), fp32_mask, op="lt")

            for row in range(ROWS_PER_AIV):
                scores_lo = S.vmuls(S.vld(S_ub[row, 0], "NORM"), T.float32(SOFTMAX_SCALE), fp32_mask)
                scores_hi = S.vmuls(S.vld(S_ub[row, VECTOR_LANES], "NORM"), T.float32(SOFTMAX_SCALE), fp32_mask)
                masked_lo = S.vsel(scores_lo, mask_value, keep_lo)
                masked_hi = S.vsel(scores_hi, mask_value, keep_hi)
                S.vsts(S_ub[row, 0], masked_lo, fp32_mask, "NORM_B32")
                S.vsts(S_ub[row, VECTOR_LANES], masked_hi, fp32_mask, "NORM_B32")
                max_acc = S.vmax(masked_lo, masked_hi, fp32_mask)
                if NUM_TOPK_BLOCKS > 1:
                    # Online softmax: m_new = max(m_old, block_max).
                    old_max = S.vld(m_ub[row], "BRC_B32")
                    S.vsts(old_m_ub[row], old_max, scalar_mask, "ONEPT_B32")
                    max_acc = S.vmax(max_acc, old_max, fp32_mask)
                row_max = S.vcmax(max_acc, fp32_mask)
                row_max = S.vmaxs(row_max, T.float32(MAX_FLOOR), scalar_mask)
                S.vsts(m_ub[row], row_max, scalar_mask, "ONEPT_B32")

            S.mem_bar("VST_VLD")

            # Pack unnormalized P = exp(score - m_new) directly in BF16 NZ.
            for row in range(ROWS_PER_AIV):
                row_max = S.vld(m_ub[row], "BRC_B32")
                even_scores, odd_scores = S.vld2(S_ub[row, 0], "DINTLV_B32")
                even_prob = S.vexpdif(even_scores, row_max, fp32_mask)
                odd_prob = S.vexpdif(odd_scores, row_max, fp32_mask)
                sum_acc = S.vadd(even_prob, odd_prob, fp32_mask)
                even_bf16 = S.vcvt(even_prob, DTYPE, fp32_mask, sat=False, part=0)
                odd_bf16 = S.vcvt(odd_prob, DTYPE, fp32_mask, sat=False, part=1)
                packed = S.vor(
                    T.reinterpret(even_bf16, "uint16x128"),
                    T.reinterpret(odd_bf16, "uint16x128"),
                    bf16_mask,
                )
                p_nz_ptr = S.vsstb(
                    T.reinterpret(packed, "bfloat16x128"),
                    p_nz_ptr,
                    nz_stride,
                    bf16_mask,
                    update=True,
                )
                prob_sum = S.vcadd(sum_acc, fp32_mask)
                if NUM_TOPK_BLOCKS > 1:
                    # alpha = exp(m_old - m_new); l_new = alpha * l_old + sum(P).
                    alpha = S.vexpdif(S.vld(old_m_ub[row], "BRC_B32"), row_max, fp32_mask)
                    S.vsts(alpha_ub[row], alpha, scalar_mask, "ONEPT_B32")
                    denominator = S.vadd(
                        S.vmul(S.vld(l_ub[row], "BRC_B32"), alpha, fp32_mask),
                        prob_sum,
                        scalar_mask,
                    )
                    S.vsts(l_ub[row], denominator, scalar_mask, "ONEPT_B32")
                else:
                    S.vsts(l_ub[row], prob_sum, scalar_mask, "ONEPT_B32")

    @T.macro
    def init_softmax_state(m_ub, l_ub):
        with T.SimdVF(latency=200):
            full = S.pset(32, "PAT_ALL")
            one = S.pset(32, "PAT_VL1")
            initial_max = S.vdup(T.float32(MAX_FLOOR), ACCUM_DTYPE, full)
            initial_sum = S.vdup(T.float32(0.0), ACCUM_DTYPE, full)
            for row in range(ROWS_PER_AIV):
                S.vsts(m_ub[row], initial_max, one, "ONEPT_B32")
                S.vsts(l_ub[row], initial_sum, one, "ONEPT_B32")

    @T.macro
    def clear_output(O_ub):
        with T.SimdVF(latency=142):
            full = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0.0), ACCUM_DTYPE, full)
            for row in range(ROWS_PER_AIV):
                for column in range(0, HEAD_DIM, VECTOR_LANES):
                    S.vsts(O_ub[row, column], zero, full, "NORM_B32")

    @T.macro
    def accumulate_output(O_ub, O_tmp_ub, alpha_ub):
        # O = alpha * O + P@V.  At the first tile alpha is exp(MAX_FLOOR - m),
        # which underflows to 0, so the cleared accumulator needs no special case.
        with T.SimdVF(latency=318):
            full = S.pset(32, "PAT_ALL")
            for row in range(ROWS_PER_AIV):
                alpha = S.vld(alpha_ub[row], "BRC_B32")
                for column in range(0, HEAD_DIM, VECTOR_LANES):
                    prev = S.vld(O_ub[row, column], "NORM")
                    cur = S.vld(O_tmp_ub[row, column], "NORM")
                    updated = S.vadd(S.vmul(alpha, prev, full), cur, full)
                    S.vsts(O_ub[row, column], updated, full, "NORM_B32")

    @T.macro
    def finalize_output(O_ub, O_cast_ub, l_ub, O, batch_id, query_id, head, sid):
        # Normalize in FP32, convert to BF16, then store the single live row.
        # A fully masked query has l == 0; clamp the denominator so it emits
        # zeros instead of dividing by zero.
        with T.SimdVF(latency=814):
            full = S.pset(32, "PAT_ALL")
            denominator = S.vld(l_ub[0], "BRC_B32")
            safe_denominator = S.vmaxs(denominator, T.float32(1.0e-30), full)
            for column in range(0, HEAD_DIM, VECTOR_LANES):
                normalized = S.vdiv(S.vld(O_ub[0, column], "NORM"), safe_denominator, full)
                packed = S.vcvt(normalized, DTYPE, full, sat=False)
                S.vsts(O_cast_ub[0, column], packed, full, "PK_B32")
        # Every M row is a copy of the same query, so row 0 holds the same values
        # on both AIVs.  Split the store by column rather than guarding it with
        # ``sid == 0``: InsertSync rejects a sid-guarded MTE3 because both AIV
        # programs then carry a write to the same region (AIV0.MTE3 -> AIV1.MTE3,
        # distance 0).  Disjoint column halves need no cross-AIV ordering.
        half = HEAD_DIM // 2
        T.copy(
            O_cast_ub[0, sid * half : (sid + 1) * half],
            O[batch_id, query_id, head, sid * half : (sid + 1) * half],
        )

    # -- auto-scheduled Cube + Vector kernel ------------------------------

    @T.prim_func
    def main(
        Q: T.Buffer((batch_size, NUM_HEADS, seq_len, HEAD_DIM), DTYPE),
        K: T.Buffer((batch_size, NUM_HEADS, kv_len, HEAD_DIM), DTYPE),
        V: T.Buffer((batch_size, NUM_HEADS, kv_len, HEAD_DIM), DTYPE),
        Indices: T.Buffer((batch_size, seq_len, top_k), "int32"),
        Mask: T.Buffer((batch_size, 1, seq_len, kv_len), "uint8"),
        O: T.Buffer((batch_size, seq_len, NUM_HEADS, HEAD_DIM), DTYPE),
    ):
        with T.MixedKernel(NUM_BLOCKS) as (core_id, sid):
            Q_l1 = T.alloc_l1((M_PAD, HEAD_DIM), DTYPE)
            K_l1 = T.alloc_l1((BLOCK_TOP_K, HEAD_DIM), DTYPE)
            V_l1 = T.alloc_l1((BLOCK_TOP_K, HEAD_DIM), DTYPE)
            P_l1 = T.alloc_l1((M_PAD, BLOCK_TOP_K), DTYPE)
            # The L1 GEMM path is NT-only, so PV stages both operands in L0 and
            # transposes V on the L1->L0 copy: B is (N=HEAD_DIM, K=BLOCK_TOP_K).
            pv_a_l0 = T.alloc_l0a((M_PAD, BLOCK_TOP_K), DTYPE)
            pv_b_l0 = T.alloc_l0b((HEAD_DIM, BLOCK_TOP_K), DTYPE)
            qk_acc_l0c = T.alloc_l0c((M_PAD, BLOCK_TOP_K), ACCUM_DTYPE)
            pv_acc_l0c = T.alloc_l0c((M_PAD, HEAD_DIM), ACCUM_DTYPE)

            Q_ub = T.alloc_shared((M_PAD, HEAD_DIM), DTYPE)
            K_ub = T.alloc_shared((CHUNK_ROWS, HEAD_DIM), DTYPE)
            K_nz_ub = T.alloc_shared((CHUNK_ROWS + 1, HEAD_DIM), DTYPE)
            V_ub = T.alloc_shared((CHUNK_ROWS, HEAD_DIM), DTYPE)
            V_nz_ub = T.alloc_shared((CHUNK_ROWS + 1, HEAD_DIM), DTYPE)
            S_ub = T.alloc_shared((ROWS_PER_AIV, BLOCK_TOP_K), ACCUM_DTYPE)
            P_nz_ub = T.alloc_shared((ROWS_PER_AIV + 1, BLOCK_TOP_K), DTYPE)
            O_ub = T.alloc_shared((ROWS_PER_AIV, HEAD_DIM), ACCUM_DTYPE)
            O_tmp_ub = T.alloc_shared((ROWS_PER_AIV, HEAD_DIM), ACCUM_DTYPE)
            # Same bytes as O_ub would do, but a separate buffer keeps the
            # in-place compaction trick out of the correctness path.
            O_cast_ub = T.alloc_shared((ROWS_PER_AIV, HEAD_DIM), DTYPE)
            # Per-row running max m, exp sum l, and output rescaling factor alpha.
            m_ub = T.alloc_shared((ROWS_PER_AIV,), ACCUM_DTYPE)
            l_ub = T.alloc_shared((ROWS_PER_AIV,), ACCUM_DTYPE)
            alpha_ub = T.alloc_shared((ROWS_PER_AIV,), ACCUM_DTYPE)
            old_m_ub = T.alloc_shared((ROWS_PER_AIV,), ACCUM_DTYPE)
            penalty_ub = T.alloc_shared((BLOCK_TOP_K,), ACCUM_DTYPE)
            mask_row_ub = T.alloc_shared((kv_len if MASK_STAGED else 32,), "uint8")
            mask_words_ub = T.alloc_shared((kv_len if VECTOR_MASK else 1,), "uint32")
            mask_indices_ub = T.alloc_shared((BLOCK_TOP_K,), "int32")

            T.annotate_layout(
                {
                    # Cube operand views in L1 must be MajorK: the compact-NZ UB
                    # staging is only accepted into a canonical NZ destination.
                    Q_l1: make_ascend_major_k_layout(Q_l1),
                    K_l1: make_ascend_major_k_layout(K_l1),
                    V_l1: make_ascend_major_k_layout(V_l1),
                    K_nz_ub: make_ascend_compact_nz_layout(K_nz_ub),
                    V_nz_ub: make_ascend_compact_nz_layout(V_nz_ub),
                    P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub),
                }
            )
            T.annotate_buffer_versions(
                {
                    K_l1: 2,
                    V_l1: 2,
                    P_l1: 2,
                    pv_a_l0: 2,
                    # A full HEAD_DIM x BLOCK_TOP_K tile already fills L0B.
                    pv_b_l0: 1,
                    qk_acc_l0c: 2,
                    pv_acc_l0c: 2,
                    S_ub: 2,
                    K_ub: 2,
                    K_nz_ub: 2,
                    V_ub: 2,
                    V_nz_ub: 2,
                    P_nz_ub: 2,
                    O_tmp_ub: 2,
                    alpha_ub: 2,
                    # Loop-carried across tiles: must stay single-versioned.
                    O_ub: 1,
                    m_ub: 1,
                    l_ub: 1,
                    old_m_ub: 1,
                }
            )

            if VECTOR_MASK:
                T.annotate_buffer_versions({mask_words_ub: 1, mask_indices_ub: 2})

            # Each core gets exactly its share, including the partial final round.
            for task in T.serial((NUM_TASKS + NUM_BLOCKS - 1 - core_id) // NUM_BLOCKS):
                query_group = task * NUM_BLOCKS + core_id
                query = query_group // HEAD_GROUPS
                head_start = (query_group % HEAD_GROUPS) * HEADS_PER_TASK
                batch_id = query // seq_len
                query_id = query % seq_len
                # The mask row is per query and per KV position, so when it can
                # be staged it is loaded once per query/head-group task and reused by all
                # index tiles.  Otherwise every lookup stays a GM scalar read.
                if MASK_STAGED:
                    T.copy(Mask[batch_id, 0, query_id, 0:kv_len], mask_row_ub)

                if VECTOR_MASK:
                    _expand_mask_row(mask_row_ub, mask_words_ub, kv_len)

                for head_offset in T.serial(HEADS_PER_TASK):
                    head = head_start + head_offset
                    # Every padded M row carries the same query vector, so all
                    # 16 Cube rows -- and both AIVs -- compute the same result.
                    # GM->UB row copies followed by one UB->L1 copy keep the
                    # ND2NZ conversion on a full 16-row tile.
                    # AIV0 owns the full Q tile; two writers would overlap in L1.
                    if sid == 0:
                        for row in range(M_PAD):
                            T.copy(Q[batch_id, head, query_id, :], Q_ub[row, :])
                        T.copy(Q_ub, Q_l1)
                    if NUM_TOPK_BLOCKS > 1:
                        init_softmax_state(m_ub, l_ub)
                        clear_output(O_ub)

                    for block in T.Pipelined(
                        NUM_TOPK_BLOCKS,
                        num_stages=PIPELINE_STAGES,
                        annotations={"enable_offset": True},
                    ):
                        block_base = block * BLOCK_TOP_K
                        block_valid = T.min(top_k - block_base, BLOCK_TOP_K)

                        # QK -> masked sparse softmax -> PV.
                        gather_rows(
                            K,
                            batch_id,
                            query_id,
                            head,
                            block_base,
                            block_valid,
                            sid,
                            Indices,
                            K_ub,
                            K_nz_ub,
                            K_l1,
                        )
                        # The L1 GEMM uses hidden L0 scratch shared with PV.
                        with T.Cube():
                            T.ascend_pipe_barrier("PIPE_ALL")
                            T.gemm(Q_l1, K_l1, qk_acc_l0c, transpose_B=True, clear_accum=True)
                            T.ascend_pipe_barrier("PIPE_ALL")
                        T.dual_copy(qk_acc_l0c, S_ub)

                        # Fetch V while the softmax runs; K and V are independent.
                        gather_rows(
                            V,
                            batch_id,
                            query_id,
                            head,
                            block_base,
                            block_valid,
                            sid,
                            Indices,
                            V_ub,
                            V_nz_ub,
                            V_l1,
                        )

                        if VECTOR_MASK:
                            T.copy(Indices[batch_id, query_id, block_base : block_base + BLOCK_TOP_K], mask_indices_ub)
                            _vector_mask_penalty(mask_indices_ub, mask_words_ub, penalty_ub, kv_len)
                        else:
                            mask_penalty(
                                Mask,
                                mask_row_ub,
                                penalty_ub,
                                block_valid,
                                batch_id,
                                query_id,
                                block_base,
                                Indices,
                            )
                        softmax_pack(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, penalty_ub)
                        T.dual_copy(P_nz_ub[:ROWS_PER_AIV, :], P_l1)
                        # Keep PV operand loads and GEMM in one Cube task.
                        with T.Cube():
                            T.ascend_pipe_barrier("PIPE_ALL")
                            T.copy(P_l1, pv_a_l0)
                            T.copy(V_l1, pv_b_l0, transpose=True)
                            T.ascend_pipe_barrier("PIPE_ALL")
                            T.gemm(pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True)
                            T.ascend_pipe_barrier("PIPE_ALL")

                        if NUM_TOPK_BLOCKS > 1:
                            T.dual_copy(pv_acc_l0c, O_tmp_ub)
                            accumulate_output(O_ub, O_tmp_ub, alpha_ub)
                        else:
                            T.dual_copy(pv_acc_l0c, O_ub)

                    finalize_output(O_ub, O_cast_ub, l_ub, O, batch_id, query_id, head, sid)

    return main


# -- standalone correctness check and benchmark ---------------------------


def make_inputs(batch_size: int, seq_len: int, kv_len: int):
    torch.manual_seed(88888888)
    top_k = min(INDEX_TOPK, kv_len)
    q = torch.randn((batch_size, NUM_HEADS, seq_len, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    k = torch.randn((batch_size, NUM_HEADS, kv_len, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    v = torch.randn((batch_size, NUM_HEADS, kv_len, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    # Dense top-k, exactly what the caller produces:
    #   topk = min(index_topk, index_scores.shape[-1])
    #   ind = index_scores.topk(topk, dim=-1).indices.to(torch.int32)
    index_scores = torch.rand(batch_size, seq_len, kv_len, device="npu")
    indices = index_scores.topk(top_k, dim=-1).indices.to(torch.int32)
    mask = torch.rand(batch_size, 1, seq_len, kv_len, device="npu") > 0.1
    return q, k, v, indices, mask


def reference_attention(q, k, v, indices, mask):
    """Per-query reference in float32; the gathered tiles are too large to hold at once."""
    batch_size, heads, seq_len, dim = q.shape
    top_k = indices.shape[-1]
    mask_u8 = mask.view(torch.uint8) if mask.dtype == torch.bool else mask
    out = torch.empty((batch_size, seq_len, heads, dim), dtype=torch.float32, device=q.device)
    for batch in range(batch_size):
        k_batch = k[batch]
        v_batch = v[batch]
        for query in range(seq_len):
            # torch.gather/advanced indexing require int64 indices.
            idx = indices[batch, query].long()
            gathered_k = k_batch[:, idx, :].float()  # [H, Ksel, D]
            gathered_v = v_batch[:, idx, :].float()
            scores = torch.einsum("hd,hkd->hk", q[batch, :, query, :].float(), gathered_k) * SOFTMAX_SCALE
            keep = mask_u8[batch, 0, query][idx].ne(0)  # [Ksel]
            scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
            # Softmax over an all-masked row is NaN; the kernel emits zeros there.
            probabilities = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
            out[batch, query] = torch.einsum("hk,hkd->hd", probabilities, gathered_v)
    return out


def compile_kernel(batch_size, seq_len, kv_len, *, num_blocks=None, tiling=None):
    return tilelang.compile(
        glm_dsa_fwd(batch_size, seq_len, kv_len, min(INDEX_TOPK, kv_len), num_blocks=num_blocks, tiling=tiling),
        target="ascend",
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


def pick_num_blocks(seq_len, batch_size=1, cores=None, *, heads_per_task=None):
    heads_per_task = FwdTiling().heads_per_task if heads_per_task is None else heads_per_task
    assert heads_per_task in (1, 2, 4, 8, 16, 32, 64)
    assert batch_size > 0 and seq_len > 0
    cores = get_num_cube_cores(torch.npu.current_device()) if cores is None else cores
    assert cores > 0, "cores must be positive"
    tasks = batch_size * seq_len * (NUM_HEADS // heads_per_task)
    return min(cores, tasks)


def run(batch_size=1, seq_len=448, *, kv_len=4096, num_blocks=None, tiling=None):
    kernel = compile_kernel(batch_size, seq_len, kv_len, num_blocks=num_blocks, tiling=tiling)
    q, k, v, indices, mask = make_inputs(batch_size, seq_len, kv_len)
    expected = reference_attention(q, k, v, indices, mask)
    actual = kernel(q, k, v, indices, mask.view(torch.uint8))
    torch.npu.synchronize()
    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)
    print(f"Verification passed; max error: {(actual.float() - expected).abs().max().item():.6f}", flush=True)

    latency_ms = do_bench(lambda: kernel(q, k, v, indices, mask.view(torch.uint8)), backend="msprof", _n_warmup=30, _n_repeat=100)
    top_k = min(INDEX_TOPK, kv_len)
    flops = 4 * batch_size * seq_len * NUM_HEADS * HEAD_DIM * top_k
    print(f"{latency_ms * 1000:.2f} us/iter | {flops / (latency_ms * 1e-3) / 1e12:.1f} TFLOPS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=4096)
    args = parser.parse_args()
    run(batch_size=args.batch_size, seq_len=args.seq_len, kv_len=args.kv_len)
