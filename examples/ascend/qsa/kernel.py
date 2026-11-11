"""QSA (Qwen sparse attention) grouped-query forward on Ascend.

24 query heads share 2 KV heads with head_dim = 256; BF16 in/out, FP32
accumulation and online-softmax state. Each query selects top_k whole
4-token KV blocks (quads), so one scalar index load plus one 4-burst
DMA covers a quad. Invalid slots hold -1 in Indices and 0 in Valid; the
host passes clamped indices and the (valid - 1) * 1e30 pen mask drops
their softmax contribution. Q is pre-grouped per KV head and zero-padded
to MP = 32 rows so each AIV owns one 16-row cube fractal; O keeps that
layout and the host slices the first group rows per head. With
dynamic_shape=True, num_tokens and kv_length are symbolic (T.dynamic)
and one compiled kernel serves all shapes.
"""

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout
from tvm import tirx


NUM_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
PAD_M = 32
BLOCK_I = 128


def _copy_kv_rows(dst, src, n_rows, row_stride, row_bytes):
    """Gather n_rows consecutive BF16 KV rows in one DMA."""
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.ascend_copy_gm_to_ubuf"),
        dst,
        src,
        0,
        n_rows,
        row_bytes,
        0,
        0,
        0,
        0,
        row_stride * row_bytes,
        row_bytes,
    )


def qsa_fwd(total_q, total_kv, top_k=2176, sm_scale=None, *, dynamic_shape=False):
    """Build the QSA forward kernel. total_q / total_kv are num_tokens and
    kv_length; dynamic_shape=True makes them symbolic dimensions."""
    if top_k <= 0 or top_k % BLOCK_I:
        raise ValueError(f"top_k must be a positive multiple of {BLOCK_I}")
    MP = PAD_M
    D = HEAD_DIM
    BI = BLOCK_I
    VL = 64
    ROWS = MP // 2
    ROWS_PER_AIV = BI // 2
    if dynamic_shape:
        total_q = T.dynamic("num_tokens")
        total_kv = T.dynamic("kv_length")
    NI = top_k // BI
    # 32-row chunks halve the scatter/ub2l1 flag traffic, but with NI <= 2
    # the loop runs sequentially and 32-row scatters hit a precision bug.
    CHUNK_ROWS = BI // 4 if NI > 2 else 16
    total_tasks = total_q * NUM_KV_HEADS
    dtype = "bfloat16"
    accum_dtype = "float32"
    scale = sm_scale if sm_scale is not None else D**-0.5
    # K and V share one L1 buffer (disjoint lifetimes); 3 versions decouple
    # the next gather from current consumers.
    KV_L1_VERSIONS = 3 if NI > 1 else 1

    @T.macro
    def gather_kv(qk_head, sid, KV, indices_ub, KV_ub, KV_nz_ub, KV_l1):
        # One quad = 4 consecutive tokens (block contract), so a single
        # scalar load plus one 4-burst DMA covers it.
        for chunk in T.Serial(ROWS_PER_AIV // CHUNK_ROWS):
            row_start = chunk * CHUNK_ROWS
            for quad in T.Serial(CHUNK_ROWS // 4):
                r = row_start + 4 * quad
                base = T.int32(indices_ub[sid * ROWS_PER_AIV + r])
                T.assume(base >= 0)
                T.assume(base < total_kv)
                if base + 4 <= total_kv:
                    _copy_kv_rows(
                        T.access_ptr(KV_ub[4 * quad, 0], "w"),
                        T.access_ptr(KV[base, qk_head, 0], "r"),
                        4,
                        NUM_KV_HEADS,
                        D * 2,
                    )
                else:
                    # Tail quad crosses the KV boundary; gather per row.
                    for i in T.Serial(4):
                        idx = T.int32(indices_ub[sid * ROWS_PER_AIV + r + i])
                        T.assume(idx >= 0)
                        T.assume(idx < total_kv)
                        T.copy(KV[idx, qk_head, :], KV_ub[4 * quad + i, :])
            T.copy(KV_ub, KV_nz_ub[:CHUNK_ROWS, :])
            l1_row = sid * ROWS_PER_AIV + row_start
            T.copy(KV_nz_ub[:CHUNK_ROWS, :], KV_l1[l1_row : l1_row + CHUNK_ROWS, :])

    @T.macro
    def softmax_pack(S_ub, Val_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub):
        # Column mask pen = (valid - 1) * 1e30: valid -> 0, invalid -> -1e30.
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            one = S.pset(32, "PAT_VL1")
            b16 = S.pset(16, "PAT_VL128")
            v_lo = S.vcvt(S.vld(Val_ub[0], "NORM"), "float32", full)
            v_hi = S.vcvt(S.vld(Val_ub[VL], "NORM"), "float32", full)
            pen_lo = S.vmuls(S.vadds(v_lo, T.float32(-1.0), full), T.float32(1e30), full)
            pen_hi = S.vmuls(S.vadds(v_hi, T.float32(-1.0), full), T.float32(1e30), full)

            p_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BI),
                dtype,
            )
            nz_stride = T.int32(((ROWS + 1) << 16) | 1)

            for r in range(ROWS):
                s0 = S.vmuls(S.vld(S_ub[r, 0], "NORM"), T.float32(scale), full)
                s1 = S.vmuls(S.vld(S_ub[r, VL], "NORM"), T.float32(scale), full)
                s0 = S.vadd(s0, pen_lo, full)
                s1 = S.vadd(s1, pen_hi, full)
                S.vsts(S_ub[r, 0], s0, full, "NORM_B32")
                S.vsts(S_ub[r, VL], s1, full, "NORM_B32")
                m_all = S.vmax(s0, s1, full)
                if NI > 1:
                    old = S.vld(m_ub[r], "BRC_B32")
                    S.vsts(old_m_ub[r], old, one, "ONEPT_B32")
                    m_all = S.vmax(m_all, old, full)
                mrow = S.vcmax(m_all, full)
                S.vsts(m_ub[r], mrow, one, "ONEPT_B32")

            S.mem_bar("VST_VLD")

            for r in range(ROWS):
                mrow = S.vld(m_ub[r], "BRC_B32")
                e0, e1 = S.vld2(S_ub[r, 0], "DINTLV_B32")
                p0 = S.vexpdif(e0, mrow, full)
                p1 = S.vexpdif(e1, mrow, full)
                sumall = S.vadd(p0, p1, full)
                b0 = S.vcvt(p0, dtype, full, sat=False, part=0)
                b1 = S.vcvt(p1, dtype, full, sat=False, part=1)
                packed = S.vor(
                    T.reinterpret(b0, "uint16x128"),
                    T.reinterpret(b1, "uint16x128"),
                    b16,
                )
                p_nz_ptr = S.vsstb(
                    T.reinterpret(packed, "bfloat16x128"),
                    p_nz_ptr,
                    nz_stride,
                    b16,
                    update=True,
                )
                psum = S.vcadd(sumall, full)
                if NI > 1:
                    alpha = S.vexpdif(S.vld(old_m_ub[r], "BRC_B32"), mrow, full)
                    S.vsts(alpha_ub[r], alpha, one, "ONEPT_B32")
                    lnew = S.vadd(
                        S.vmul(S.vld(l_ub[r], "BRC_B32"), alpha, full),
                        psum,
                        one,
                    )
                    S.vsts(l_ub[r], lnew, one, "ONEPT_B32")
                else:
                    S.vsts(l_ub[r], psum, one, "ONEPT_B32")

    @T.prim_func
    def main(
        Q: T.Buffer((total_q, NUM_KV_HEADS, MP, D), dtype),
        K: T.Buffer((total_kv, NUM_KV_HEADS, D), dtype),
        V: T.Buffer((total_kv, NUM_KV_HEADS, D), dtype),
        Indices: T.Buffer((total_q, top_k), "int32"),
        Valid: T.Buffer((total_q, top_k), "int32"),
        O: T.Buffer((total_q, NUM_KV_HEADS, MP, D), dtype),
    ):
        with T.MixedKernel(total_tasks) as (tid, sid):
            grp = tid // NUM_KV_HEADS
            g = tid % NUM_KV_HEADS

            Q_l1 = T.alloc_l1((MP, D), dtype)
            KV_l1 = T.alloc_l1((BI, D), dtype)
            P_l1 = T.alloc_l1((MP, BI), dtype)
            qk_l0c = T.alloc_l0c((MP, BI), accum_dtype)
            pv_l0c = T.alloc_l0c((MP, D), accum_dtype)

            KV_ub = T.alloc_shared((CHUNK_ROWS, D), dtype)
            KV_nz_ub = T.alloc_shared((CHUNK_ROWS + 1, D), dtype)
            S_ub = T.alloc_shared((ROWS, BI), accum_dtype)
            P_nz_ub = T.alloc_shared((ROWS + 1, BI), dtype)
            idx_ub = T.alloc_shared((BI,), "int32")
            val_ub = T.alloc_shared((BI,), "int32")
            m_ub = T.alloc_shared((VL,), accum_dtype)
            l_ub = T.alloc_shared((VL,), accum_dtype)
            alpha_ub = T.alloc_shared((VL,), accum_dtype)
            old_m_ub = T.alloc_shared((VL,), accum_dtype)
            O_ub = T.alloc_shared((ROWS, D), accum_dtype)
            O_tmp_ub = T.alloc_shared((ROWS, D), accum_dtype)
            O_out_ub = T.alloc_shared((ROWS, D), dtype)

            T.annotate_layout(
                {
                    KV_nz_ub: make_ascend_compact_nz_layout(KV_nz_ub),
                    P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub),
                }
            )
            T.annotate_buffer_versions(
                {
                    KV_l1: KV_L1_VERSIONS,
                    P_l1: 2,
                    qk_l0c: 2,
                    pv_l0c: 2,
                    S_ub: 2,
                    KV_ub: 2,
                    KV_nz_ub: 2,
                    P_nz_ub: 2,
                    idx_ub: 2,
                    val_ub: 2,
                    O_ub: 1,
                    O_tmp_ub: 1,
                }
            )
            if NI > 1:
                T.annotate_buffer_versions({m_ub: 1, l_ub: 1})

            for task_block in T.Pipelined(
                NI,
                num_stages=2 if NI > 2 else 1,
                annotations={"enable_offset": True},
            ):
                b = task_block * BI
                if task_block == 0:
                    T.copy(Q[grp, g, :, :], Q_l1, l2_cache_ctrl=4)
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        S.vsts(m_ub[0], S.vdup(T.float32(-(2.0**30)), "float32"), full, "NORM_B32")
                        S.vsts(l_ub[0], S.vdup(T.float32(0.0), "float32"), full, "NORM_B32")
                T.copy(Indices[grp, b : b + BI], idx_ub)
                T.copy(Valid[grp, b : b + BI], val_ub)
                gather_kv(g, sid, K, idx_ub, KV_ub, KV_nz_ub, KV_l1)
                T.gemm(Q_l1, KV_l1, qk_l0c, transpose_B=True, clear_accum=True)
                T.dual_copy(qk_l0c, S_ub)
                gather_kv(g, sid, V, idx_ub, KV_ub, KV_nz_ub, KV_l1)
                softmax_pack(S_ub, val_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub)
                T.dual_copy(P_nz_ub[:ROWS, :], P_l1)
                T.gemm(P_l1, KV_l1, pv_l0c, transpose_B=False, clear_accum=True)
                if task_block == 0:
                    T.dual_copy(pv_l0c, O_ub)
                else:
                    T.dual_copy(pv_l0c, O_tmp_ub)
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        for r in range(ROWS):
                            a = S.vld(alpha_ub[r], "BRC_B32")
                            for c in range(0, D, VL):
                                o = S.vld(O_ub[r, c], "NORM")
                                t = S.vld(O_tmp_ub[r, c], "NORM")
                                S.vsts(O_ub[r, c], S.vadd(S.vmul(o, a, full), t, full), full, "NORM_B32")
                if task_block == NI - 1:
                    with T.SimdVF():
                        # Guard l == 0 (fully invalid query) to 1 to avoid
                        # NaN, then cache 1/l.
                        full = S.pset(32, "PAT_ALL")
                        b16 = S.pset(16, "PAT_VL128")
                        lv = S.vld(l_ub[0], "NORM")
                        ones = S.vadds(S.vmuls(lv, T.float32(0.0), full), T.float32(1.0), full)
                        guard = S.vcmps(lv, T.float32(0.0), full, op="le")
                        denom = S.vsel(ones, lv, guard)
                        recip = S.vdiv(ones, denom, full)
                        S.vsts(l_ub[0], recip, full, "NORM_B32")
                        S.mem_bar("VST_VLD")
                        for r in range(ROWS):
                            rp = S.vld(l_ub[r], "BRC_B32")
                            for c in range(0, D, 2 * VL):
                                o0, o1 = S.vld2(O_ub[r, c], "DINTLV_B32")
                                f0 = S.vmul(o0, rp, full)
                                f1 = S.vmul(o1, rp, full)
                                c0 = S.vcvt(f0, dtype, full, sat=False, part=0)
                                c1 = S.vcvt(f1, dtype, full, sat=False, part=1)
                                S.vsts(
                                    O_out_ub[r, c],
                                    T.reinterpret(
                                        S.vor(T.reinterpret(c0, "uint16x128"), T.reinterpret(c1, "uint16x128"), b16), "bfloat16x128"
                                    ),
                                    b16,
                                    "NORM_B16",
                                )
                    # SimdVF blocks cannot touch GM; dual_copy stores AIV
                    # sid's half.
                    T.dual_copy(O_out_ub, O[grp, g, :, :])

    return main
