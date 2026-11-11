"""Cube implementation of KDA chunk prefill for Ascend A5.

``total_tokens`` is ``T.dynamic()`` and is bound from the packed ``q`` tensor.
``num_seqs`` and ``aligned`` are compile-time constants. ``cu_seqlens`` has
shape ``[num_seqs + 1]`` and every kernel reads it. Omitting ``cu_seqlens``
on the host synthesizes ``[0, q.shape[1]]``.

A launch is aligned when every sequence length is a multiple of 64, including
an empty sequence. That specialization keeps the 2026-11-05 full-chunk Cube
math: W/U use a direct L0C-to-GM fixpipe. Prepare, state, and output share one
serial task loop; ``aligned`` is a Python constant, so the tail loads and the
varlen Cube stores are absent from the aligned IR. A pipeline over that stripe
does not lower: the trip count depends on the block index, and the overlapped
body does not fit in UB. The task count and the chunk-to-sequence map are
recomputed from ``cu_seqlens`` so the Cube loop bound is a scalar, not a UB
value. Any remainder uses the varlen kernel. There the tail is the task whose
local chunk is the short last chunk of its sequence; it loads only the valid
rows and zero-fills the rest.

The supported range is ``B=1``, ``H=HV`` even, ``K=V=128``, ``BT=64``, BF16
activations, FP32 raw-gate parameters, and a raw sigmoid gate lower bound of
``-5``. Head count, sequence count, alignment, and the AIC count select the
compiled specialization. The grid is one block per AIC. Prepare and output
stripe ``chunk × head`` tasks (a head pair for prepare). State stripes
``sequence × value tile`` and keeps one sequence's chunk chain on that block.
State reloads ``h0[s]`` for its sequence and does not prefetch the tail or
the next sequence. The host entry is ``KDAChunkPrefill``.
"""

import functools
import math
from typing import Final

import tilelang
import tilelang.ascend.language as T
import torch

__all__ = [
    "KDAChunkPrefill",
    "output",
    "prepare",
    "state",
]

TARGET: Final = "ascend"
BATCH: Final = 1
SEQ_LEN: Final = 16384
HEADS: Final = 96
VALUE_HEADS: Final = 96
DIM_K: Final = 128
DIM_V: Final = 128
BLOCK_T: Final = 64
# The 32nd token, 0-based. One 64x64 Gram uses it as the exponent reference.
# Its prefix sits in the low half of the log-decay tile, which the Gram
# epilogue does not overwrite.
GATE_REF: Final = 31
NUM_CHUNKS: Final = SEQ_LEN // BLOCK_T
DT_LOWER_BOUND: Final = -5.0
LOG2E: Final = 1.4426950408889634
LN2: Final = 0.6931471805599453
ATTN_SCALE: Final = 1.0 / math.sqrt(DIM_K)
ACTIVATION_DTYPE: Final = "bfloat16"
MAX_SEQS: Final = 1024


@T.macro
def stage_pair(sid, src_a, src_b, a0, b0, a1, b1):
    """Each AIV writes its private UB tile into a distinct L1 slot."""

    if sid == 0:
        T.copy(src_a, a0)
        T.copy(src_b, b0)
    else:
        T.copy(src_a, a1)
        T.copy(src_b, b1)


@T.macro
def mma(a_l1, b_l1, a_l0, b_l0, c, transpose_a=False, transpose_b=False):
    """L1 to L0, then gemm. Boolean flags are Python constants and fold away."""

    if transpose_a:
        T.copy(a_l1, a_l0, transpose=True)
    else:
        T.copy(a_l1, a_l0)
    if transpose_b:
        T.copy(b_l1, b_l0, transpose=True)
    else:
        T.copy(b_l1, b_l0)
    T.gemm(a_l0, b_l0, c, transpose_B=True, clear_accum=True)


def live_chunks(num_seqs):
    """Chunk count over ``num_seqs``. The bound stays a scalar read from GM."""

    @T.macro
    def count(cu_seqlens):
        live = T.alloc_var("int32", init=0)
        for seq in T.serial(num_seqs):
            span = T.cast(cu_seqlens[seq + 1] - cu_seqlens[seq], "int32")
            live = live + (span + T.int32(BLOCK_T - 1)) // T.int32(BLOCK_T)
        return live

    return count


def owning_sequence(num_seqs):
    """Sequence whose chunk prefix contains one flat chunk id.

    The prefix is recomputed from ``cu_seqlens`` rather than a UB copy.
    Prepare's Cube store uses the result as a GM index, and a UB scalar is
    not visible on AIC.
    """

    @T.macro
    def owner(cu_seqlens, flat):
        seq = T.alloc_var("int32", init=0)
        running = T.alloc_var("int32", init=0)
        for candidate in T.serial(num_seqs):
            span = T.cast(cu_seqlens[candidate + 1] - cu_seqlens[candidate], "int32")
            count = (span + T.int32(BLOCK_T - 1)) // T.int32(BLOCK_T)
            if running <= flat and flat < running + count:
                seq = candidate
            running = running + count
        return seq

    return owner


def chunk_origin(num_seqs):
    """Flat chunk id where one sequence starts. Same scan as ``owning_sequence``."""

    @T.macro
    def origin(cu_seqlens, seq):
        origin = T.alloc_var("int32", init=0)
        for candidate in T.serial(num_seqs):
            if candidate < seq:
                span = T.cast(cu_seqlens[candidate + 1] - cu_seqlens[candidate], "int32")
                count = (span + T.int32(BLOCK_T - 1)) // T.int32(BLOCK_T)
                origin = origin + count
        return origin

    return origin


def _check_shape(heads: int, value_heads: int, num_seqs: int, core_num: int) -> None:
    if value_heads % 2:
        raise ValueError("dual-AIV kernels pair heads and require an even value-head count")
    if heads != value_heads:
        raise ValueError(f"H and HV must match; received heads={heads} value_heads={value_heads}")
    if not 1 <= num_seqs <= MAX_SEQS:
        raise ValueError(f"num_seqs must be in 1..{MAX_SEQS}; received {num_seqs}")
    if core_num < 1:
        raise ValueError(f"core_num must be positive; received {core_num}")


def _token_axis(num_seqs: int):
    """One dynamic token axis, shared by the tensor shapes and the kernel body."""

    total_tokens = T.dynamic("total_tokens")
    chunk_bound = total_tokens // BLOCK_T + num_seqs
    return total_tokens, chunk_bound


def prepare(
    heads: int = HEADS,
    value_heads: int = VALUE_HEADS,
    num_seqs: int = 1,
    aligned: bool = False,
    core_num: int = 32,
):
    """Stage 1. One block per AIC. Each block stripes chunk × head-pair tasks."""

    _check_shape(heads, value_heads, num_seqs, core_num)
    aligned = bool(aligned)
    core_num = int(core_num)
    num_seqs = int(num_seqs)
    total_tokens, chunk_bound = _token_axis(num_seqs)

    def build():
        count_chunks = live_chunks(num_seqs)
        owner = owning_sequence(num_seqs)
        origin = chunk_origin(num_seqs)

        @T.macro
        def prepare_math(
            base,
            slot,
            rows,
            is_tail,
            head,
            sid,
            head0,
            head1,
            v,
            w,
            u,
            qg,
            kg,
            aqk,
            dec,
            raw_g_ub,
            beta_ub,
            a_log_ub,
            dt_bias_ub,
            qg_ub,
            kp_ub,
            kn_ub,
            gram,
            p_fp32,
            rhs_128,
            p,
            a0_l1,
            b0_l1,
            a1_l1,
            b1_l1,
            p0_l1,
            p1_l1,
            a_l0,
            b_l0,
            p_l0,
            b64_l0,
            c64,
            c128,
        ):
            # Both arms are pure. The padded rows stay zero, so dec reads this row.
            # The aligned kernel has no tail, so the index is the last full row.
            if aligned:
                last_row = T.int32(BLOCK_T - 1)
            else:
                last_row = T.if_then_else(is_tail, rows - T.int32(1), T.int32(BLOCK_T - 1))
            center_row = T.if_then_else(last_row < T.int32(GATE_REF), last_row, T.int32(GATE_REF))
            # q and k are already in qg_ub / kp_ub. Padded tail rows are zero, so a
            # 64-row scan leaves the valid prefix unchanged. dec reads the last
            # valid row, not the padded one.
            with T.SimdVF():
                mask = T.simd.pset(32)
                rate = T.simd.vexp(T.simd.vld(a_log_ub[0], "BRC_B32"), mask)
                neg_rate = T.simd.vmuls(rate, T.float32(-1.0), mask)
                running = T.simd.alloc_local((2,), "float32")
                running[0] = T.simd.vdup(T.float32(0.0), "float32", mask)
                running[1] = T.simd.vdup(T.float32(0.0), "float32", mask)
                for row in T.serial(BLOCK_T):
                    for part in (0, 1):
                        col = part * 64
                        shifted = T.simd.vadd(T.simd.vld(raw_g_ub[row, col]), T.simd.vld(dt_bias_ub[col]), mask)
                        denom = T.simd.vadds(
                            T.simd.vexp(T.simd.vmul(shifted, neg_rate, mask), mask),
                            T.float32(1.0),
                            mask,
                        )
                        decay = T.simd.vdiv(T.simd.vdup(T.float32(DT_LOWER_BOUND), "float32", mask), denom, mask)
                        decay = T.simd.vsel(
                            T.simd.vdup(T.float32(DT_LOWER_BOUND * 0.5), "float32", mask),
                            decay,
                            T.simd.vcmps(shifted, T.float32(0.0), mask, op="eq"),
                        )
                        running[part] = T.simd.vadd(running[part], decay, mask)
                        T.simd.vsts(
                            raw_g_ub[row, col],
                            T.simd.vmuls(running[part], T.float32(LOG2E), mask),
                            mask,
                        )
                T.simd.mem_bar("VST_VLD")
                # kg = k * exp(L_last - L) before k is overwritten. The gap is <= 0.
                for row in T.serial(BLOCK_T):
                    for col in (0, 64):
                        k_value = T.simd.vcvt(T.simd.vld(kp_ub[row, col], "UNPK_B16"), "float32")
                        carry = T.simd.vexp(
                            T.simd.vmuls(
                                T.simd.vsub(
                                    T.simd.vld(raw_g_ub[last_row, col]),
                                    T.simd.vld(raw_g_ub[row, col]),
                                    mask,
                                ),
                                T.float32(LN2),
                                mask,
                            ),
                            mask,
                        )
                        T.simd.vsts(
                            kn_ub[row, col],
                            T.simd.vcvt(T.simd.vmul(k_value, carry, mask), "bfloat16"),
                            mask,
                            dist="PK_B32",
                        )
                for col in (0, 64):
                    last = T.simd.vld(raw_g_ub[last_row, col])
                    T.simd.vsts(
                        dt_bias_ub[col],
                        T.simd.vexp(T.simd.vmuls(last, T.float32(LN2), mask), mask),
                        mask,
                    )
            if aligned:
                T.copy(kn_ub, kg[0, base : base + BLOCK_T, head, :])
            else:
                if is_tail:
                    T.copy(kn_ub[:rows, :], kg[0, base : base + rows, head, :])
                else:
                    T.copy(kn_ub, kg[0, base : base + BLOCK_T, head, :])
            with T.SimdVF():
                mask = T.simd.pset(32)
                # One 64x64 Gram. Factors are exp(L - L_ref) and exp(L_ref - L)
                # about the 32nd token, so each side spans 32 decay steps.
                for row in T.serial(BLOCK_T):
                    for col in (0, 64):
                        gc_i = T.simd.vld(raw_g_ub[row, col])
                        gc_c = T.simd.vld(raw_g_ub[center_row, col])
                        up = T.simd.vexp(
                            T.simd.vmuls(T.simd.vsub(gc_i, gc_c, mask), T.float32(LN2), mask),
                            mask,
                        )
                        down = T.simd.vexp(
                            T.simd.vmuls(T.simd.vsub(gc_c, gc_i, mask), T.float32(LN2), mask),
                            mask,
                        )
                        q_value = T.simd.vcvt(T.simd.vld(qg_ub[row, col], "UNPK_B16"), "float32")
                        k_value = T.simd.vcvt(T.simd.vld(kp_ub[row, col], "UNPK_B16"), "float32")
                        T.simd.vsts(
                            qg_ub[row, col],
                            T.simd.vcvt(T.simd.vmul(q_value, up, mask), "bfloat16"),
                            mask,
                            dist="PK_B32",
                        )
                        T.simd.vsts(
                            kp_ub[row, col],
                            T.simd.vcvt(T.simd.vmul(k_value, up, mask), "bfloat16"),
                            mask,
                            dist="PK_B32",
                        )
                        T.simd.vsts(
                            kn_ub[row, col],
                            T.simd.vcvt(T.simd.vmul(k_value, down, mask), "bfloat16"),
                            mask,
                            dist="PK_B32",
                        )
            stage_pair(sid, kp_ub, kn_ub, a0_l1, b0_l1, a1_l1, b1_l1)
            mma(a0_l1, b0_l1, a_l0, b_l0, c64, transpose_a=False, transpose_b=False)
            with T.Task():
                T.copy(c64, gram, sub_blockid=0)
            mma(a1_l1, b1_l1, a_l0, b_l0, c64, transpose_a=False, transpose_b=False)
            with T.Task():
                T.copy(c64, gram, sub_blockid=1)
            with T.SimdVF():
                mask = T.simd.pset(32)
                columns = T.simd.vci(T.int32(0), "int32")
                for row in T.serial(BLOCK_T):
                    lower = T.simd.vcmps(columns, T.int32(row), mask, op="lt")
                    values = T.simd.vsel(
                        T.simd.vld(gram[row, 0]),
                        T.simd.vdup(T.float32(0.0), "float32", mask),
                        lower,
                    )
                    beta = T.simd.vcvt(T.simd.vld(beta_ub[row], "BRC_B16"), "float32")
                    scaled = T.simd.vmul(T.simd.vmuls(values, T.float32(-1.0), mask), beta, mask)
                    T.simd.vsts(p_fp32[1, row, 0], scaled, mask)
            stage_pair(sid, qg_ub, kn_ub, a0_l1, b0_l1, a1_l1, b1_l1)
            mma(a0_l1, b0_l1, a_l0, b_l0, c64, transpose_a=False, transpose_b=False)
            with T.Task():
                T.copy(c64, gram, sub_blockid=0)
            mma(a1_l1, b1_l1, a_l0, b_l0, c64, transpose_a=False, transpose_b=False)
            with T.Task():
                T.copy(c64, gram, sub_blockid=1)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for row in T.serial(BLOCK_T):
                    for col in (0, 64):
                        # qg/kp were stored as q/k * exp(L - L_ref). Multiply back
                        # exp(L_ref), which the epilogue left in the reference row.
                        to_center = T.simd.vexp(
                            T.simd.vmuls(T.simd.vld(raw_g_ub[center_row, col]), T.float32(LN2), mask),
                            mask,
                        )
                        q_value = T.simd.vcvt(T.simd.vld(qg_ub[row, col], "UNPK_B16"), "float32")
                        k_pos = T.simd.vcvt(T.simd.vld(kp_ub[row, col], "UNPK_B16"), "float32")
                        T.simd.vsts(
                            qg_ub[row, col],
                            T.simd.vcvt(T.simd.vmul(q_value, to_center, mask), "bfloat16"),
                            mask,
                            dist="PK_B32",
                        )
                        T.simd.vsts(
                            kp_ub[row, col],
                            T.simd.vcvt(T.simd.vmul(k_pos, to_center, mask), "bfloat16"),
                            mask,
                            dist="PK_B32",
                        )
            if aligned:
                T.copy(qg_ub, qg[0, base : base + BLOCK_T, head, :])
            else:
                if is_tail:
                    T.copy(qg_ub[:rows, :], qg[0, base : base + rows, head, :])
                else:
                    T.copy(qg_ub, qg[0, base : base + BLOCK_T, head, :])

            with T.SimdVF():
                mask = T.simd.pset(32)
                columns = T.simd.vci(T.int32(0), "int32")
                ones = T.simd.vdup(T.float32(1.0), "float32", mask)
                zeros = T.simd.vdup(T.float32(0.0), "float32", mask)
                acc = T.simd.alloc_local((4,), "float32")
                total = T.simd.alloc_local((1,), "float32")
                for row in T.serial(BLOCK_T):
                    for lane in (0, 1, 2, 3):
                        acc[lane] = T.simd.vdup(T.float32(0.0), "float32", mask)
                    for group in T.serial(row // 4):
                        prior = group * 4
                        for lane in (0, 1, 2, 3):
                            prev = T.simd.vld(p_fp32[0, prior + lane, 0])
                            coeff = T.simd.vld(p_fp32[1, row, prior + lane], "BRC_B32")
                            acc[lane] = T.simd.vadd(acc[lane], T.simd.vmul(prev, coeff, mask), mask)
                    total[0] = T.simd.vadd(
                        T.simd.vadd(acc[0], acc[1], mask),
                        T.simd.vadd(acc[2], acc[3], mask),
                        mask,
                    )
                    prior0 = (row // 4) * 4
                    for tail in T.serial(row - prior0):
                        prev = T.simd.vld(p_fp32[0, prior0 + tail, 0])
                        coeff = T.simd.vld(p_fp32[1, row, prior0 + tail], "BRC_B32")
                        total[0] = T.simd.vadd(total[0], T.simd.vmul(prev, coeff, mask), mask)
                    diagonal = T.simd.vcmps(columns, T.int32(row), mask, op="eq")
                    T.simd.vsts(p_fp32[0, row, 0], T.simd.vsel(ones, total[0], diagonal), mask)
                    T.simd.mem_bar("VST_VLD")
                for row in T.serial(BLOCK_T):
                    lower = T.simd.vcmps(columns, T.int32(row), mask, op="le")
                    solved = T.simd.vsel(T.simd.vld(p_fp32[0, row, 0]), zeros, lower)
                    T.simd.vsts(
                        p[0, row, 0],
                        T.simd.vcvt(solved, "bfloat16"),
                        mask,
                        dist="PK_B32",
                    )
                    beta_row = T.simd.vcvt(T.simd.vld(beta_ub[row], "BRC_B16"), "float32")
                    for col in (0, 64):
                        k_value = T.simd.vcvt(T.simd.vld(kp_ub[row, col], "UNPK_B16"), "float32")
                        T.simd.vsts(
                            rhs_128[row, col],
                            T.simd.vcvt(T.simd.vmul(k_value, beta_row, mask), "bfloat16"),
                            mask,
                            dist="PK_B32",
                        )

            stage_pair(sid, p[0, :, :], rhs_128, p0_l1, b0_l1, p1_l1, b1_l1)
            mma(p0_l1, b0_l1, p_l0, b64_l0, c128, transpose_a=False, transpose_b=True)
            # The aligned kernel fixpipes L0C straight to GM. A short fixpipe is
            # not a legal Cube store, and mixing it with a direct store makes the
            # two AIVs' MTE3 pipes depend on each other, so the varlen kernel
            # lands every W tile on one AIV.
            if aligned:
                T.copy(c128, w[0, base : base + BLOCK_T, head0, :])
            else:
                with T.Task():
                    T.copy(c128, qg_ub, sub_blockid=0)
                if sid == 0:
                    if is_tail:
                        T.copy(qg_ub[:rows, :], w[0, base : base + rows, head0, :])
                    else:
                        T.copy(qg_ub, w[0, base : base + BLOCK_T, head0, :])
            mma(p1_l1, b1_l1, p_l0, b64_l0, c128, transpose_a=False, transpose_b=True)
            if aligned:
                T.copy(c128, w[0, base : base + BLOCK_T, head1, :])
            else:
                with T.Task():
                    T.copy(c128, qg_ub, sub_blockid=0)
                if sid == 0:
                    if is_tail:
                        T.copy(qg_ub[:rows, :], w[0, base : base + rows, head1, :])
                    else:
                        T.copy(qg_ub, w[0, base : base + BLOCK_T, head1, :])

            if aligned:
                T.copy(v[0, base : base + BLOCK_T, head, :], kp_ub)
            else:
                if is_tail:
                    T.copy(v[0, base : base + rows, head, :], kp_ub[:rows, :])
                    T.fill(kp_ub[rows:BLOCK_T, :], T.bfloat16(0))
                else:
                    T.copy(v[0, base : base + BLOCK_T, head, :], kp_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for row in T.serial(BLOCK_T):
                    beta = T.simd.vcvt(T.simd.vld(beta_ub[row], "BRC_B16"), "float32")
                    for col in (0, 64):
                        values = T.simd.vcvt(T.simd.vld(kp_ub[row, col], "UNPK_B16"), "float32")
                        T.simd.vsts(
                            rhs_128[row, col],
                            T.simd.vcvt(T.simd.vmul(values, beta, mask), "bfloat16"),
                            mask,
                            dist="PK_B32",
                        )
            stage_pair(sid, p[0, :, :], rhs_128, p0_l1, b0_l1, p1_l1, b1_l1)
            mma(p0_l1, b0_l1, p_l0, b64_l0, c128, transpose_a=False, transpose_b=True)
            if aligned:
                T.copy(c128, u[0, base : base + BLOCK_T, head0, :])
            else:
                with T.Task():
                    T.copy(c128, qg_ub, sub_blockid=0)
                if sid == 0:
                    if is_tail:
                        T.copy(qg_ub[:rows, :], u[0, base : base + rows, head0, :])
                    else:
                        T.copy(qg_ub, u[0, base : base + BLOCK_T, head0, :])
            mma(p1_l1, b1_l1, p_l0, b64_l0, c128, transpose_a=False, transpose_b=True)
            if aligned:
                T.copy(c128, u[0, base : base + BLOCK_T, head1, :])
            else:
                with T.Task():
                    T.copy(c128, qg_ub, sub_blockid=0)
                if sid == 0:
                    if is_tail:
                        T.copy(qg_ub[:rows, :], u[0, base : base + rows, head1, :])
                    else:
                        T.copy(qg_ub, u[0, base : base + BLOCK_T, head1, :])
            with T.SimdVF():
                mask = T.simd.pset(32)
                columns = T.simd.vci(T.int32(0), "int32")
                zeros = T.simd.vdup(T.float32(0.0), "float32", mask)
                for row in T.serial(BLOCK_T):
                    lower = T.simd.vcmps(columns, T.int32(row), mask, op="le")
                    masked = T.simd.vsel(T.simd.vld(gram[row, 0]), zeros, lower)
                    T.simd.vsts(
                        p[0, row, 0],
                        T.simd.vcvt(masked, "bfloat16"),
                        mask,
                        dist="PK_B32",
                    )
            if aligned:
                T.copy(p[0, :, :], aqk[0, base : base + BLOCK_T, head, :])
            else:
                if is_tail:
                    T.copy(p[0, :rows, :], aqk[0, base : base + rows, head, :])
                else:
                    T.copy(p[0, :, :], aqk[0, base : base + BLOCK_T, head, :])
            T.copy(dt_bias_ub, dec[slot, head, :])

        @T.macro
        def prepare_chunk(
            base,
            slot,
            rows,
            is_tail,
            head,
            sid,
            head0,
            head1,
            q,
            k,
            v,
            raw_g,
            beta,
            dt_bias,
            w,
            u,
            qg,
            kg,
            aqk,
            dec,
            raw_g_ub,
            beta_ub,
            a_log_ub,
            dt_bias_ub,
            qg_ub,
            kp_ub,
            kn_ub,
            gram,
            p_fp32,
            rhs_128,
            p,
            a0_l1,
            b0_l1,
            a1_l1,
            b1_l1,
            p0_l1,
            p1_l1,
            a_l0,
            b_l0,
            p_l0,
            b64_l0,
            c64,
            c128,
        ):
            # Only the varlen specialization calls this. The tail predicate shares
            # the task with the full chunks, so the UB-to-L1 copies exist once.
            if is_tail:
                T.copy(raw_g[0, base : base + rows, head, :], raw_g_ub[:rows, :])
                T.fill(raw_g_ub[rows:BLOCK_T, :], T.float32(0))
                T.copy(beta[0, base : base + rows, head], beta_ub[:rows])
                T.fill(beta_ub[rows:BLOCK_T], T.bfloat16(0))
            else:
                T.copy(raw_g[0, base : base + BLOCK_T, head, :], raw_g_ub)
                T.copy(beta[0, base : base + BLOCK_T, head], beta_ub)
            T.copy(dt_bias[head * DIM_K : (head + 1) * DIM_K], dt_bias_ub)
            if is_tail:
                T.copy(q[0, base : base + rows, head, :], qg_ub[:rows, :])
                T.fill(qg_ub[rows:BLOCK_T, :], T.bfloat16(0))
                T.copy(k[0, base : base + rows, head, :], kp_ub[:rows, :])
                T.fill(kp_ub[rows:BLOCK_T, :], T.bfloat16(0))
            else:
                T.copy(q[0, base : base + BLOCK_T, head, :], qg_ub)
                T.copy(k[0, base : base + BLOCK_T, head, :], kp_ub)
            prepare_math(
                base,
                slot,
                rows,
                is_tail,
                head,
                sid,
                head0,
                head1,
                v,
                w,
                u,
                qg,
                kg,
                aqk,
                dec,
                raw_g_ub,
                beta_ub,
                a_log_ub,
                dt_bias_ub,
                qg_ub,
                kp_ub,
                kn_ub,
                gram,
                p_fp32,
                rhs_128,
                p,
                a0_l1,
                b0_l1,
                a1_l1,
                b1_l1,
                p0_l1,
                p1_l1,
                a_l0,
                b_l0,
                p_l0,
                b64_l0,
                c64,
                c128,
            )

        @T.prim_func
        def prepare(
            q: T.Tensor((1, total_tokens, heads, DIM_K), "bfloat16"),
            k: T.Tensor((1, total_tokens, heads, DIM_K), "bfloat16"),
            v: T.Tensor((1, total_tokens, value_heads, DIM_V), "bfloat16"),
            raw_g: T.Tensor((1, total_tokens, value_heads, DIM_K), "float32"),
            beta: T.Tensor((1, total_tokens, value_heads), "bfloat16"),
            cu_seqlens: T.Tensor((num_seqs + 1,), "int64"),
            a_log: T.Tensor((value_heads,), "float32"),
            dt_bias: T.Tensor((value_heads * DIM_K,), "float32"),
            w: T.Tensor((1, total_tokens, value_heads, DIM_K), "bfloat16"),
            u: T.Tensor((1, total_tokens, value_heads, DIM_V), "bfloat16"),
            qg: T.Tensor((1, total_tokens, value_heads, DIM_K), "bfloat16"),
            kg: T.Tensor((1, total_tokens, value_heads, DIM_K), "bfloat16"),
            aqk: T.Tensor((1, total_tokens, value_heads, BLOCK_T), "bfloat16"),
            dec: T.Tensor((chunk_bound, value_heads, DIM_K), "float32"),
        ):
            with T.MixedKernel(core_num, sids=2) as (block, sid):
                raw_g_ub = T.alloc_shared((BLOCK_T, DIM_K), "float32")
                beta_ub = T.alloc_shared((BLOCK_T,), "bfloat16")
                a_log_ub = T.alloc_shared((1,), "float32")
                dt_bias_ub = T.alloc_shared((DIM_K,), "float32")
                qg_ub = T.alloc_shared((BLOCK_T, DIM_K), "bfloat16")
                kp_ub = T.alloc_shared((BLOCK_T, DIM_K), "bfloat16")
                kn_ub = T.alloc_shared((BLOCK_T, DIM_K), "bfloat16")
                p_fp32 = T.view(raw_g_ub, (2, BLOCK_T, BLOCK_T), dtype="float32")
                rhs_128 = T.view(qg_ub, (BLOCK_T, DIM_V), dtype="bfloat16")
                gram = T.alloc_shared((BLOCK_T, BLOCK_T), "float32")
                p = T.view(kn_ub, (2, BLOCK_T, BLOCK_T), dtype="bfloat16")
                a0_l1 = T.alloc_l1((BLOCK_T, DIM_K), "bfloat16")
                b0_l1 = T.alloc_l1((BLOCK_T, DIM_K), "bfloat16")
                a1_l1 = T.alloc_l1((BLOCK_T, DIM_K), "bfloat16")
                b1_l1 = T.alloc_l1((BLOCK_T, DIM_K), "bfloat16")
                p0_l1 = T.alloc_l1((BLOCK_T, BLOCK_T), "bfloat16")
                p1_l1 = T.alloc_l1((BLOCK_T, BLOCK_T), "bfloat16")
                a_l0 = T.alloc_l0a((BLOCK_T, DIM_K), "bfloat16")
                b_l0 = T.alloc_l0b((BLOCK_T, DIM_K), "bfloat16")
                p_l0 = T.alloc_l0a((BLOCK_T, BLOCK_T), "bfloat16")
                b64_l0 = T.alloc_l0b((DIM_V, BLOCK_T), "bfloat16")
                c64 = T.alloc_l0c((BLOCK_T, BLOCK_T), "float32")
                c128 = T.alloc_l0c((BLOCK_T, DIM_V), "float32")
                T.annotate_buffer_versions(
                    {
                        raw_g_ub: 1,
                        beta_ub: 2,
                        a_log_ub: 1,
                        dt_bias_ub: 2,
                        qg_ub: 1,
                        kp_ub: 1,
                        kn_ub: 1,
                        gram: 1,
                        a0_l1: 1,
                        b0_l1: 1,
                        a1_l1: 1,
                        b1_l1: 1,
                        p0_l1: 1,
                        p1_l1: 1,
                        a_l0: 1,
                        b_l0: 1,
                        p_l0: 1,
                        b64_l0: 1,
                        c64: 1,
                        c128: 1,
                    }
                )
                windows = T.int32(value_heads // 2)
                task_num = count_chunks(cu_seqlens) * windows
                # Zero when this block owns nothing. A surrounding `if` would
                # keep both AIV branches of the Cube copies live in one schedule.
                trips = (task_num - block + T.int32(core_num - 1)) // T.int32(core_num)
                for step in T.serial(trips):
                    task = block + step * T.int32(core_num)
                    flat = task // windows
                    window = task % windows
                    seq = owner(cu_seqlens, flat)
                    local_chunk = flat - origin(cu_seqlens, seq)
                    base = T.cast(cu_seqlens[seq], "int32") + local_chunk * T.int32(BLOCK_T)
                    head0 = window * 2
                    head1 = head0 + 1
                    head = head0 + sid
                    T.copy(a_log[head : head + 1], a_log_ub)
                    if aligned:
                        T.copy(raw_g[0, base : base + BLOCK_T, head, :], raw_g_ub)
                        T.copy(beta[0, base : base + BLOCK_T, head], beta_ub)
                        T.copy(dt_bias[head * DIM_K : (head + 1) * DIM_K], dt_bias_ub)
                        T.copy(q[0, base : base + BLOCK_T, head, :], qg_ub)
                        T.copy(k[0, base : base + BLOCK_T, head, :], kp_ub)
                        prepare_math(
                            base,
                            flat,
                            T.int32(0),
                            False,
                            head,
                            sid,
                            head0,
                            head1,
                            v,
                            w,
                            u,
                            qg,
                            kg,
                            aqk,
                            dec,
                            raw_g_ub,
                            beta_ub,
                            a_log_ub,
                            dt_bias_ub,
                            qg_ub,
                            kp_ub,
                            kn_ub,
                            gram,
                            p_fp32,
                            rhs_128,
                            p,
                            a0_l1,
                            b0_l1,
                            a1_l1,
                            b1_l1,
                            p0_l1,
                            p1_l1,
                            a_l0,
                            b_l0,
                            p_l0,
                            b64_l0,
                            c64,
                            c128,
                        )
                    else:
                        span = T.cast(cu_seqlens[seq + 1] - cu_seqlens[seq], "int32")
                        whole = span // T.int32(BLOCK_T)
                        rows = span - whole * T.int32(BLOCK_T)
                        is_tail = local_chunk == whole
                        prepare_chunk(
                            base,
                            flat,
                            rows,
                            is_tail,
                            head,
                            sid,
                            head0,
                            head1,
                            q,
                            k,
                            v,
                            raw_g,
                            beta,
                            dt_bias,
                            w,
                            u,
                            qg,
                            kg,
                            aqk,
                            dec,
                            raw_g_ub,
                            beta_ub,
                            a_log_ub,
                            dt_bias_ub,
                            qg_ub,
                            kp_ub,
                            kn_ub,
                            gram,
                            p_fp32,
                            rhs_128,
                            p,
                            a0_l1,
                            b0_l1,
                            a1_l1,
                            b1_l1,
                            p0_l1,
                            p1_l1,
                            a_l0,
                            b_l0,
                            p_l0,
                            b64_l0,
                            c64,
                            c128,
                        )

        return prepare

    return build()


def state(
    value_heads: int = VALUE_HEADS,
    num_seqs: int = 1,
    aligned: bool = False,
    core_num: int = 32,
):
    """Stage 2. One block per AIC. A task is one sequence of one value tile."""

    _check_shape(value_heads, value_heads, num_seqs, core_num)
    aligned = bool(aligned)
    core_num = int(core_num)
    num_seqs = int(num_seqs)
    total_tokens, chunk_bound = _token_axis(num_seqs)

    def build():
        @T.macro
        def chunk_prefix(cu_seqlens, tile_cum):
            """Chunk-id prefix of each packed sequence. Both AIVs write their own UB."""

            tile_cum[0] = T.int32(0)
            for seq in T.serial(num_seqs):
                span = T.cast(cu_seqlens[seq + 1] - cu_seqlens[seq], "int32")
                count = (span + T.int32(BLOCK_T - 1)) // T.int32(BLOCK_T)
                tile_cum[seq + 1] = tile_cum[seq] + count

        @T.macro
        def store_vn(u_tile, vn_ub, scratch, vn, base, head, v0, rows, is_tail):
            """vn = U - W@S. Alignment picks a full store; the varlen tail stores valid rows."""

            with T.SimdVF():
                mask = T.simd.pset(32)
                for row in T.serial(BLOCK_T):
                    u_value = T.simd.vcvt(T.simd.vld(u_tile[row, 0], "UNPK_B16"), "float32")
                    diff = T.simd.vsub(u_value, T.simd.vld(scratch[row, 0]), mask)
                    T.simd.vsts(
                        vn_ub[row, 0],
                        T.simd.vcvt(diff, "bfloat16"),
                        mask,
                        dist="PK_B32",
                    )
            if aligned:
                T.copy(vn_ub, vn[0, base : base + BLOCK_T, head, v0 : v0 + 64])
            else:
                if is_tail:
                    T.copy(vn_ub[:rows, :], vn[0, base : base + rows, head, v0 : v0 + 64])
                else:
                    T.copy(vn_ub, vn[0, base : base + BLOCK_T, head, v0 : v0 + 64])

        @T.macro
        def state_chunk(
            c,
            whole,
            base,
            slot,
            head,
            sid,
            k0,
            v0,
            rows,
            is_tail,
            w,
            u,
            kg,
            dec,
            h,
            vn,
            state_ub,
            state_bf,
            w_ub,
            u_ub,
            kg_ub,
            vn_ub,
            scratch,
            update,
            dec_ub,
            wk_l1,
            st_l1,
            kg_l1,
            vn_l1,
            wk_l0,
            st_l0,
            kg_l0,
            vn_l0,
            c64,
            c128,
        ):
            """One chunk of the state walk. A full chunk reuses the prefetched W, U, and Kg.
            The varlen tail reloads only its valid rows and does not prefetch onward.
            """

            if not aligned:  # noqa: SIM102
                if is_tail:
                    T.copy(w[0, base : base + rows, head, k0 : k0 + 64], w_ub[:rows, :])
                    T.fill(w_ub[rows:BLOCK_T, :], T.bfloat16(0))
                    T.copy(kg[0, base : base + rows, head, k0 : k0 + 64], kg_ub[:rows, :])
                    T.fill(kg_ub[rows:BLOCK_T, :], T.bfloat16(0))
                    if sid == 0:
                        T.copy(u[0, base : base + rows, head, v0 : v0 + 64], u_ub[:rows, :])
                        T.fill(u_ub[rows:BLOCK_T, :], T.bfloat16(0))
            T.copy(dec[slot, head, k0 : k0 + 64], dec_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for row in T.serial(64):
                    value = T.simd.vld(state_ub[row, 0])
                    T.simd.vsts(
                        state_bf[row, 0],
                        T.simd.vcvt(value, "bfloat16"),
                        mask,
                        dist="PK_B32",
                    )
            T.copy(state_bf, h[slot, head, k0 : k0 + 64, v0 : v0 + 64])
            T.copy(w_ub, wk_l1[:, k0 : k0 + 64])
            T.copy(state_bf, st_l1[k0 : k0 + 64, :])
            mma(wk_l1, st_l1, wk_l0, st_l0, c64, transpose_a=False, transpose_b=True)
            with T.Task():
                T.copy(c64, scratch, sub_blockid=0)
            if sid == 0:
                store_vn(u_ub, vn_ub, scratch, vn, base, head, v0, rows, is_tail)
                T.copy(vn_ub, vn_l1)
            if c + 1 < whole:
                T.copy(w[0, base + BLOCK_T : base + BLOCK_T + BLOCK_T, head, k0 : k0 + 64], w_ub)
                if sid == 0:
                    T.copy(u[0, base + BLOCK_T : base + BLOCK_T + BLOCK_T, head, v0 : v0 + 64], u_ub)
            T.copy(kg_ub, kg_l1[:, k0 : k0 + 64])
            mma(kg_l1, vn_l1, kg_l0, vn_l0, c128, transpose_a=True, transpose_b=True)
            T.dual_copy(c128, update)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for row in T.serial(64):
                    scale = T.simd.vld(dec_ub[row], "BRC_B32")
                    updated = T.simd.vadd(
                        T.simd.vmul(T.simd.vld(state_ub[row, 0]), scale, mask),
                        T.simd.vld(update[row, 0]),
                        mask,
                    )
                    T.simd.vsts(state_ub[row, 0], updated, mask)
            if c + 1 < whole:
                T.copy(kg[0, base + BLOCK_T : base + BLOCK_T + BLOCK_T, head, k0 : k0 + 64], kg_ub)

        @T.prim_func
        def state(
            w: T.Tensor((1, total_tokens, value_heads, DIM_K), "bfloat16"),
            u: T.Tensor((1, total_tokens, value_heads, DIM_V), "bfloat16"),
            kg: T.Tensor((1, total_tokens, value_heads, DIM_K), "bfloat16"),
            dec: T.Tensor((chunk_bound, value_heads, DIM_K), "float32"),
            h0: T.Tensor((num_seqs, value_heads, DIM_K, DIM_V), "float32"),
            cu_seqlens: T.Tensor((num_seqs + 1,), "int64"),
            h: T.Tensor((chunk_bound, value_heads, DIM_K, DIM_V), "bfloat16"),
            vn: T.Tensor((1, total_tokens, value_heads, DIM_V), "bfloat16"),
            final: T.Tensor((num_seqs, value_heads, DIM_K, DIM_V), "float32"),
        ):
            with T.MixedKernel(core_num, sids=2) as (block, sid):
                k0 = sid * 64
                state_ub = T.alloc_shared((64, 64), "float32")
                state_bf = T.alloc_shared((64, 64), "bfloat16")
                w_ub = T.alloc_shared((BLOCK_T, 64), "bfloat16")
                u_ub = T.alloc_shared((BLOCK_T, 64), "bfloat16")
                kg_ub = T.alloc_shared((BLOCK_T, 64), "bfloat16")
                vn_ub = T.alloc_shared((BLOCK_T, 64), "bfloat16")
                scratch = T.alloc_shared((BLOCK_T, 64), "float32")
                update = T.alloc_shared((64, 64), "float32")
                dec_ub = T.alloc_shared((64,), "float32")
                wk_l1 = T.alloc_l1((BLOCK_T, DIM_K), "bfloat16")
                st_l1 = T.alloc_l1((DIM_K, 64), "bfloat16")
                kg_l1 = T.alloc_l1((BLOCK_T, DIM_K), "bfloat16")
                vn_l1 = T.alloc_l1((BLOCK_T, 64), "bfloat16")
                wk_l0 = T.alloc_l0a((BLOCK_T, DIM_K), "bfloat16")
                st_l0 = T.alloc_l0b((64, DIM_K), "bfloat16")
                kg_l0 = T.alloc_l0a((DIM_K, BLOCK_T), "bfloat16")
                vn_l0 = T.alloc_l0b((64, BLOCK_T), "bfloat16")
                c64 = T.alloc_l0c((BLOCK_T, 64), "float32")
                c128 = T.alloc_l0c((DIM_K, 64), "float32")
                T.annotate_buffer_versions(
                    {
                        state_ub: 1,
                        state_bf: 1,
                        w_ub: 1,
                        u_ub: 1,
                        kg_ub: 1,
                        vn_ub: 1,
                        scratch: 1,
                        update: 1,
                        dec_ub: 1,
                        wk_l1: 1,
                        st_l1: 1,
                        kg_l1: 1,
                        vn_l1: 1,
                        wk_l0: 1,
                        st_l0: 1,
                        kg_l0: 1,
                        vn_l0: 1,
                        c64: 1,
                        c128: 1,
                    }
                )
                tile_cum = T.alloc_shared((num_seqs + 1,), "int32")
                chunk_prefix(cu_seqlens, tile_cum)
                tiles = T.int32(value_heads * 2)
                task_num = num_seqs * value_heads * 2
                trips = (task_num - block + T.int32(core_num - 1)) // T.int32(core_num)
                for step in T.serial(trips):
                    task = block + step * T.int32(core_num)
                    seq = task // tiles
                    tile = task % tiles
                    head = tile // 2
                    v0 = (tile % 2) * 64
                    bos = T.cast(cu_seqlens[seq], "int32")
                    span = T.cast(cu_seqlens[seq + 1] - cu_seqlens[seq], "int32")
                    whole = span // T.int32(BLOCK_T)
                    chunk0 = tile_cum[seq]
                    T.copy(h0[seq, head, k0 : k0 + 64, v0 : v0 + 64], state_ub)
                    if whole > 0:
                        T.copy(w[0, bos : bos + BLOCK_T, head, k0 : k0 + 64], w_ub)
                        T.copy(kg[0, bos : bos + BLOCK_T, head, k0 : k0 + 64], kg_ub)
                        if sid == 0:
                            T.copy(u[0, bos : bos + BLOCK_T, head, v0 : v0 + 64], u_ub)
                    if aligned:
                        chunk_count = whole
                        rows = T.int32(0)
                    else:
                        rows = span - whole * T.int32(BLOCK_T)
                        chunk_count = (span + T.int32(BLOCK_T - 1)) // T.int32(BLOCK_T)
                    for c in T.serial(chunk_count):
                        base = bos + c * BLOCK_T
                        slot = chunk0 + c
                        if aligned:
                            is_tail = False
                        else:
                            is_tail = c == whole
                        state_chunk(
                            c,
                            whole,
                            base,
                            slot,
                            head,
                            sid,
                            k0,
                            v0,
                            rows,
                            is_tail,
                            w,
                            u,
                            kg,
                            dec,
                            h,
                            vn,
                            state_ub,
                            state_bf,
                            w_ub,
                            u_ub,
                            kg_ub,
                            vn_ub,
                            scratch,
                            update,
                            dec_ub,
                            wk_l1,
                            st_l1,
                            kg_l1,
                            vn_l1,
                            wk_l0,
                            st_l0,
                            kg_l0,
                            vn_l0,
                            c64,
                            c128,
                        )
                    T.copy(state_ub, final[seq, head, k0 : k0 + 64, v0 : v0 + 64])

        return state

    return build()


def output(
    value_heads: int = VALUE_HEADS,
    num_seqs: int = 1,
    aligned: bool = False,
    core_num: int = 32,
):
    """Stage 3. One block per AIC. Each block stripes chunk × head tasks."""

    _check_shape(value_heads, value_heads, num_seqs, core_num)
    aligned = bool(aligned)
    core_num = int(core_num)
    num_seqs = int(num_seqs)
    total_tokens, chunk_bound = _token_axis(num_seqs)

    def build():
        count_chunks = live_chunks(num_seqs)
        owner = owning_sequence(num_seqs)
        origin = chunk_origin(num_seqs)

        @T.macro
        def output_epilogue(first, second, out_ub):
            """Scale qg@h, add Aqk@vn, and pack the bf16 tile. GM store stays outside."""

            with T.SimdVF():
                mask = T.simd.pset(32)
                for row in T.serial(BLOCK_T):
                    qh = T.simd.vmuls(T.simd.vld(first[row, 0]), T.float32(ATTN_SCALE), mask)
                    summed = T.simd.vadd(qh, T.simd.vld(second[row, 0]), mask)
                    T.simd.vsts(
                        out_ub[row, 0],
                        T.simd.vcvt(summed, "bfloat16"),
                        mask,
                        dist="PK_B32",
                    )

        @T.macro
        def output_compute(
            head,
            sid,
            q_ub,
            a_ub,
            h_ub,
            vn_ub,
            qh,
            av,
            out_ub,
            q0_l1,
            h0_l1,
            a0_l1,
            vn0_l1,
            q1_l1,
            h1_l1,
            a1_l1,
            vn1_l1,
            q_l0,
            h_l0,
            a_l0,
            vn_l0,
            c_qh,
            c_av,
        ):
            """Two GEMMs and the epilogue. The caller stores the valid rows."""

            stage_pair(sid, q_ub, h_ub, q0_l1, h0_l1, q1_l1, h1_l1)
            mma(q0_l1, h0_l1, q_l0, h_l0, c_qh, transpose_a=False, transpose_b=True)
            with T.Task():
                T.copy(c_qh, qh, sub_blockid=0)
            mma(q1_l1, h1_l1, q_l0, h_l0, c_qh, transpose_a=False, transpose_b=True)
            with T.Task():
                T.copy(c_qh, qh, sub_blockid=1)
            stage_pair(sid, a_ub, vn_ub, a0_l1, vn0_l1, a1_l1, vn1_l1)
            # vn is [token, V]. The gemm contracts B's last axis, so transpose
            # it; otherwise a square 64x64 tile reduces over V instead of token.
            mma(a0_l1, vn0_l1, a_l0, vn_l0, c_av, transpose_a=False, transpose_b=True)
            with T.Task():
                T.copy(c_av, av, sub_blockid=0)
            mma(a1_l1, vn1_l1, a_l0, vn_l0, c_av, transpose_a=False, transpose_b=True)
            with T.Task():
                T.copy(c_av, av, sub_blockid=1)
            output_epilogue(qh, av, out_ub)

        @T.prim_func
        def output(
            qg: T.Tensor((1, total_tokens, value_heads, DIM_K), "bfloat16"),
            aqk: T.Tensor((1, total_tokens, value_heads, BLOCK_T), "bfloat16"),
            h: T.Tensor((chunk_bound, value_heads, DIM_K, DIM_V), "bfloat16"),
            vn: T.Tensor((1, total_tokens, value_heads, DIM_V), "bfloat16"),
            cu_seqlens: T.Tensor((num_seqs + 1,), "int64"),
            out: T.Tensor((1, total_tokens, value_heads, DIM_V), "bfloat16"),
        ):
            with T.MixedKernel(core_num, sids=2) as (block, sid):
                v0 = sid * 64
                q_ub = T.alloc_shared((BLOCK_T, DIM_K), "bfloat16")
                a_ub = T.alloc_shared((BLOCK_T, BLOCK_T), "bfloat16")
                h_ub = T.alloc_shared((DIM_K, 64), "bfloat16")
                vn_ub = T.alloc_shared((BLOCK_T, 64), "bfloat16")
                qh = T.alloc_shared((BLOCK_T, 64), "float32")
                av = T.alloc_shared((BLOCK_T, 64), "float32")
                out_ub = T.alloc_shared((BLOCK_T, 64), "bfloat16")
                q0_l1 = T.alloc_l1((BLOCK_T, DIM_K), "bfloat16")
                h0_l1 = T.alloc_l1((DIM_K, 64), "bfloat16")
                a0_l1 = T.alloc_l1((BLOCK_T, BLOCK_T), "bfloat16")
                vn0_l1 = T.alloc_l1((BLOCK_T, 64), "bfloat16")
                q1_l1 = T.alloc_l1((BLOCK_T, DIM_K), "bfloat16")
                h1_l1 = T.alloc_l1((DIM_K, 64), "bfloat16")
                a1_l1 = T.alloc_l1((BLOCK_T, BLOCK_T), "bfloat16")
                vn1_l1 = T.alloc_l1((BLOCK_T, 64), "bfloat16")
                q_l0 = T.alloc_l0a((BLOCK_T, DIM_K), "bfloat16")
                h_l0 = T.alloc_l0b((64, DIM_K), "bfloat16")
                a_l0 = T.alloc_l0a((BLOCK_T, BLOCK_T), "bfloat16")
                vn_l0 = T.alloc_l0b((64, BLOCK_T), "bfloat16")
                c_qh = T.alloc_l0c((BLOCK_T, 64), "float32")
                c_av = T.alloc_l0c((BLOCK_T, 64), "float32")
                T.annotate_buffer_versions(
                    {
                        q_ub: 1,
                        a_ub: 1,
                        h_ub: 1,
                        vn_ub: 1,
                        qh: 1,
                        av: 1,
                        out_ub: 1,
                        q0_l1: 1,
                        h0_l1: 1,
                        a0_l1: 1,
                        vn0_l1: 1,
                        q1_l1: 1,
                        h1_l1: 1,
                        a1_l1: 1,
                        vn1_l1: 1,
                        q_l0: 1,
                        h_l0: 1,
                        a_l0: 1,
                        vn_l0: 1,
                        c_qh: 1,
                        c_av: 1,
                    }
                )
                heads = T.int32(value_heads)
                task_num = count_chunks(cu_seqlens) * heads
                trips = (task_num - block + T.int32(core_num - 1)) // T.int32(core_num)
                for step in T.serial(trips):
                    task = block + step * T.int32(core_num)
                    flat = task // heads
                    head = task % heads
                    seq = owner(cu_seqlens, flat)
                    local_chunk = flat - origin(cu_seqlens, seq)
                    base = T.cast(cu_seqlens[seq], "int32") + local_chunk * T.int32(BLOCK_T)
                    if aligned:
                        T.copy(qg[0, base : base + BLOCK_T, head, :], q_ub)
                        T.copy(aqk[0, base : base + BLOCK_T, head, :], a_ub)
                        T.copy(vn[0, base : base + BLOCK_T, head, v0 : v0 + 64], vn_ub)
                    else:
                        span = T.cast(cu_seqlens[seq + 1] - cu_seqlens[seq], "int32")
                        whole = span // T.int32(BLOCK_T)
                        rows = span - whole * T.int32(BLOCK_T)
                        is_tail = local_chunk == whole
                        if is_tail:
                            T.copy(qg[0, base : base + rows, head, :], q_ub[:rows, :])
                            T.fill(q_ub[rows:BLOCK_T, :], T.bfloat16(0))
                            T.copy(aqk[0, base : base + rows, head, :], a_ub[:rows, :])
                            T.fill(a_ub[rows:BLOCK_T, :], T.bfloat16(0))
                            T.copy(vn[0, base : base + rows, head, v0 : v0 + 64], vn_ub[:rows, :])
                            T.fill(vn_ub[rows:BLOCK_T, :], T.bfloat16(0))
                        else:
                            T.copy(qg[0, base : base + BLOCK_T, head, :], q_ub)
                            T.copy(aqk[0, base : base + BLOCK_T, head, :], a_ub)
                            T.copy(vn[0, base : base + BLOCK_T, head, v0 : v0 + 64], vn_ub)
                    T.copy(h[flat, head, :, v0 : v0 + 64], h_ub)
                    output_compute(
                        head,
                        sid,
                        q_ub,
                        a_ub,
                        h_ub,
                        vn_ub,
                        qh,
                        av,
                        out_ub,
                        q0_l1,
                        h0_l1,
                        a0_l1,
                        vn0_l1,
                        q1_l1,
                        h1_l1,
                        a1_l1,
                        vn1_l1,
                        q_l0,
                        h_l0,
                        a_l0,
                        vn_l0,
                        c_qh,
                        c_av,
                    )
                    if aligned:
                        T.copy(out_ub, out[0, base : base + BLOCK_T, head, v0 : v0 + 64])
                    else:
                        if is_tail:
                            T.copy(out_ub[:rows, :], out[0, base : base + rows, head, v0 : v0 + 64])
                        else:
                            T.copy(out_ub, out[0, base : base + BLOCK_T, head, v0 : v0 + 64])

        return output

    return build()


def lower_host_only(heads: int = 2, num_seqs: int = 1, aligned: bool = False, core_num: int = 32) -> dict[str, object]:
    """Lower all three stages without running the vendor device compiler."""

    from tvm.target import Target

    target = Target(TARGET)
    stages = {
        "prepare": prepare(heads, heads, num_seqs, aligned, core_num),
        "state": state(heads, num_seqs, aligned, core_num),
        "output": output(heads, num_seqs, aligned, core_num),
    }
    with tilelang.transform.PassContext(opt_level=3), target:
        return {name: tilelang.lower(func, target=target, enable_device_compile=False) for name, func in stages.items()}


def _compile_passes():
    return {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


def _device_core_count() -> int:
    """AIC count for the launch grid. Host-only lowering keeps the 32-core fallback."""

    try:
        from tilelang.carver.arch.driver.ascend_driver import get_num_cube_cores

        count = int(get_num_cube_cores())
    except Exception:
        return 32
    return count if count > 0 else 32


def build(heads: int, num_seqs: int, aligned: bool, core_num: int | None = None):
    """Compile the three stages. ``core_num`` is part of the cache key."""

    resolved = _device_core_count() if core_num is None else int(core_num)
    return _build_cached(heads, num_seqs, aligned, resolved)


@functools.cache
def _build_cached(heads: int, num_seqs: int, aligned: bool, core_num: int) -> tuple[object, object, object]:
    """Compile the three stages for one head count, sequence count, alignment, and grid."""

    common = dict(target=TARGET, pass_configs=_compile_passes())
    return (
        tilelang.compile(
            prepare(heads, heads, num_seqs, aligned, core_num),
            out_idx=[-6, -5, -4, -3, -2, -1],
            **common,
        ),
        tilelang.compile(
            state(heads, num_seqs, aligned, core_num),
            out_idx=[-3, -2, -1],
            **common,
        ),
        tilelang.compile(
            output(heads, num_seqs, aligned, core_num),
            out_idx=[-1],
            **common,
        ),
    )


def _as_fp32(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        return tensor
    if tensor.dtype == torch.bfloat16:
        return tensor.float()
    raise TypeError(f"{name} must be float32 or bfloat16")


def normalize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
):
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("q must be packed BSND with B=1")
    total = q.shape[1]
    heads = q.shape[2]
    if k.shape != q.shape:
        raise ValueError(f"k must have shape {tuple(q.shape)}")
    if v.shape != (1, total, heads, DIM_V) or q.shape[3] != DIM_K:
        raise ValueError(f"q/k/v must have K=V={DIM_K} and H=HV")
    if raw_g.shape != (1, total, heads, DIM_K):
        raise ValueError(f"raw_g must have shape {(1, total, heads, DIM_K)}")
    if beta.shape != (1, total, heads):
        raise ValueError(f"beta must have shape {(1, total, heads)}")
    if heads % 2:
        raise ValueError("value-head count must be even")
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, beta)):
        raise TypeError("q, k, v, and beta must be torch.bfloat16")
    if raw_g.dtype != torch.float32:
        raise TypeError("raw_g must be torch.float32")
    if a_log.shape != (heads,) or dt_bias.shape != (heads * DIM_K,):
        raise ValueError(f"a_log/dt_bias must have shapes {(heads,)} and {(heads * DIM_K,)}")
    a_log = _as_fp32(a_log, "a_log")
    dt_bias = _as_fp32(dt_bias, "dt_bias")
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, total], dtype=torch.int64, device=q.device)
    if cu_seqlens.dtype != torch.int64 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be an int64 vector of length num_seqs + 1")
    num_seqs = cu_seqlens.numel() - 1
    if num_seqs > MAX_SEQS:
        raise ValueError(f"num_seqs must be <= {MAX_SEQS}")
    host_cu = [int(value) for value in cu_seqlens.detach().to("cpu").tolist()]
    if host_cu[0] != 0 or host_cu[-1] != total:
        raise ValueError("cu_seqlens must start at 0 and end at total_tokens")
    if any(host_cu[i] > host_cu[i + 1] for i in range(num_seqs)):
        raise ValueError("cu_seqlens must be nondecreasing")
    if cu_seqlens.device != q.device:
        cu_seqlens = cu_seqlens.to(device=q.device)
    if initial_state is None:
        initial_state = torch.zeros((num_seqs, heads, DIM_K, DIM_V), dtype=torch.float32, device=q.device)
    elif initial_state.shape != (num_seqs, heads, DIM_K, DIM_V) or initial_state.dtype != torch.float32:
        raise ValueError(f"initial_state must be float32 with shape {(num_seqs, heads, DIM_K, DIM_V)}")
    tensors = (q, k, v, raw_g, beta, a_log, dt_bias, cu_seqlens, initial_state)
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all inputs must be contiguous")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all inputs must reside on the same device")
    aligned = all((host_cu[index + 1] - host_cu[index]) % BLOCK_T == 0 for index in range(num_seqs))
    return a_log, dt_bias, cu_seqlens, initial_state, heads, num_seqs, aligned


class KDAChunkPrefill:
    """KDA chunked prefill: preparation, the state walk, and the output.

    ``heads`` and ``value_heads`` are fixed for the instance. Sequence count and
    alignment are taken from ``cu_seqlens`` on each call. ``cu_seqlens=None`` is
    the single sequence ``[0, S]``.
    """

    def __init__(self, heads: int, value_heads: int | None = None, core_num: int | None = None) -> None:
        if value_heads is None:
            value_heads = heads
        _check_shape(heads, value_heads, 1, core_num or 1)
        self.heads = heads
        self.value_heads = value_heads
        self.core_num = core_num

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_g: torch.Tensor,
        beta: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        *,
        debug: bool = False,
    ):
        """Run the three stages. Final state is ``[num_seqs, HV, K, V]``.

        With ``debug=True``, also return the tensors each stage writes.
        """

        a_log, dt_bias, cu_seqlens, initial_state, heads, num_seqs, aligned = normalize(
            q, k, v, raw_g, beta, a_log, dt_bias, cu_seqlens, initial_state
        )
        if heads != self.heads:
            raise ValueError(f"expected H=HV={self.heads}, got {heads}")
        prepare_kernel, state_kernel, output_kernel = build(self.heads, num_seqs, aligned, self.core_num)
        w, u, qg, kg, aqk, dec = prepare_kernel(q, k, v, raw_g, beta, cu_seqlens, a_log, dt_bias)
        h, vn, final = state_kernel(w, u, kg, dec, initial_state, cu_seqlens)
        attn = output_kernel(qg, aqk, h, vn, cu_seqlens)
        if not debug:
            return attn, final
        return (
            attn,
            final,
            {
                "w": w,
                "u": u,
                "qg": qg,
                "kg": kg,
                "aqk": aqk,
                "dec": dec,
                "h": h,
                "vn": vn,
                "final": final,
                "o": attn,
            },
        )
