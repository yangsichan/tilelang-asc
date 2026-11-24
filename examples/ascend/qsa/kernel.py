"""Persistent mixed-core QSA schedule for Qwen3.8-Flash-Next on Ascend 950."""

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_major_k_layout

NUM_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
GROUP_SIZE = NUM_HEADS // NUM_KV_HEADS
LOGICAL_TOP_K = 2051
PHYSICAL_TOP_K = 2176
PAD_M = 16
PIPELINE_STAGES = 3
SUPPORTED_AIC_COUNTS = (28, 32, 36)

# SIMD reduction state uses one 64-element register per field.
MAX, SUM, ALPHA, OLD_MAX = 0, 64, 128, 192
INDEX, VALID = 0, 256


@T.macro
def _initialize(f, acc):
    with T.SimdVF():
        full = S.pset(32)
        zero = S.vdup(0.0, "float32", full)
        S.vsts(f[MAX], S.vdup(-1073741824.0, "float32", full), full)
        S.vsts(f[SUM], zero, full)
        for row in range(8):
            for col in range(4):
                S.vsts(acc[row, col * 64], zero, full)


@T.macro
def _softmax(f, ints, scores, packed, parity):
    with T.SimdVF():
        full = S.pset(32)
        one = S.pset(32, "PAT_VL1")
        b16 = S.pset(16, "PAT_VL128")
        valid0 = S.vcvt(S.vld(ints[VALID + parity * 128]), "float32", full)
        valid1 = S.vcvt(S.vld(ints[VALID + parity * 128 + 64]), "float32", full)
        pen0 = S.vmuls(S.vadds(valid0, -1.0, full), 1e30, full)
        pen1 = S.vmuls(S.vadds(valid1, -1.0, full), 1e30, full)
        dst = S.make_ubuf_ptr(T.access_ptr(packed[parity, 0, 0], "w", extent=1152), "bfloat16")
        for row in range(8):
            s0 = S.vadd(S.vmuls(S.vld(scores[parity, row, 0]), 0.0625, full), pen0, full)
            s1 = S.vadd(S.vmuls(S.vld(scores[parity, row, 64]), 0.0625, full), pen1, full)
            S.vsts(scores[parity, row, 0], s0, full)
            S.vsts(scores[parity, row, 64], s1, full)
            old = S.vld(f[MAX + row], "BRC_B32")
            S.vsts(f[OLD_MAX + row], old, one, "ONEPT_B32")
            m = S.vcmax(S.vmax(S.vmax(s0, s1, full), old, full), full)
            S.vsts(f[MAX + row], m, one, "ONEPT_B32")
        S.mem_bar("VST_VLD")
        for row in range(8):
            m = S.vld(f[MAX + row], "BRC_B32")
            even, odd = S.vld2(scores[parity, row, 0], "DINTLV_B32")
            e0 = S.vexpdif(even, m, full)
            e1 = S.vexpdif(odd, m, full)
            pair_sum = S.vadd(e0, e1, full)
            b0 = S.vcvt(e0, "bfloat16", full, sat=False, part=0)
            b1 = S.vcvt(e1, "bfloat16", full, sat=False, part=1)
            bits = S.vor(T.reinterpret(b0, "uint16x128"), T.reinterpret(b1, "uint16x128"), b16)
            dst = S.vsstb(T.reinterpret(bits, "bfloat16x128"), dst, (9 << 16) | 1, b16, update=True)
            psum = S.vcadd(pair_sum, full)
            alpha = S.vexpdif(S.vld(f[OLD_MAX + row], "BRC_B32"), m, full)
            S.vsts(f[ALPHA + row], alpha, one, "ONEPT_B32")
            new_sum = S.vadd(S.vmul(S.vld(f[SUM + row], "BRC_B32"), alpha, full), psum, one)
            S.vsts(f[SUM + row], new_sum, one, "ONEPT_B32")


@T.macro
def _accumulate(f, acc, pv, parity):
    with T.SimdVF():
        full = S.pset(32)
        for row in range(8):
            alpha = S.vld(f[ALPHA + row], "BRC_B32")
            for col in range(4):
                prev = S.vld(acc[row, col * 64])
                cur = S.vld(pv[parity, row, col * 64])
                S.vsts(acc[row, col * 64], S.vadd(S.vmul(prev, alpha, full), cur, full), full)


@T.macro
def _normalize(f, acc, output):
    with T.SimdVF():
        full = S.pset(32)
        b16 = S.pset(16, "PAT_VL128")
        sums = S.vld(f[SUM])
        ones = S.vadds(S.vmuls(sums, 0.0, full), 1.0, full)
        denom = S.vsel(ones, sums, S.vcmps(sums, 0.0, full, "le"))
        S.vsts(f[SUM], S.vdiv(ones, denom, full), full)
        S.mem_bar("VST_VLD")
        for row in range(8):
            recip = S.vld(f[SUM + row], "BRC_B32")
            for col in range(2):
                even, odd = S.vld2(acc[row, col * 128], "DINTLV_B32")
                b0 = S.vcvt(S.vmul(even, recip, full), "bfloat16", full, sat=False, part=0)
                b1 = S.vcvt(S.vmul(odd, recip, full), "bfloat16", full, sat=False, part=1)
                bits = S.vor(T.reinterpret(b0, "uint16x128"), T.reinterpret(b1, "uint16x128"), b16)
                S.vsts(output[row, col * 128], T.reinterpret(bits, "bfloat16x128"), b16, "NORM_B16")


@T.macro
def _cube(core, cores, tiles, q, kv_l1, q_l1, p_l1, qa, pa, kb, vb, qk_c, pv_c, scores, pv):
    T.ascend_set_flag("M_MTE1", 0)
    T.ascend_set_flag("M_MTE1", 1)
    T.ascend_set_flag("MTE1_MTE2", 0)
    # One PV is issued one tick behind its QK. The last tick only drains PV.
    for tick in range(tiles + 1):
        next_slot = tick & 1
        prev_slot = (tick + 1) & 1
        if tick < tiles:
            T.ascend_wait_flag("MTE1_MTE2", 0)
            if tick % 17 == 0:
                task = core + tick // 17 * cores
                T.assume(task < q.shape[0] * 2)
                T.copy(q[task // 2, task % 2, :, :], q_l1, l2_cache_ctrl=4)
            T.ascend_set_flag("MTE2_MTE1", 0)
            for part in range(2):
                if part == 0:
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 22 + next_slot)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 6 + next_slot)
                T.ascend_wait_flag("M_MTE1", 0)
                T.copy(kv_l1[tick % 3, :, part * 128 : (part + 1) * 128], kb)
                if part == 1:
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 4 + next_slot)
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 20 + next_slot)
                if part == 0:
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                T.copy(q_l1[:, part * 128 : (part + 1) * 128], qa)
                T.ascend_set_flag("MTE1_M", 0)
                if part == 1:
                    T.ascend_set_flag("MTE1_MTE2", 0)
                T.ascend_wait_flag("MTE1_M", 0)
                T.gemm(qa, kb, qk_c[next_slot, :, :], transpose_B=True, clear_accum=(part == 0))
                T.ascend_set_flag("M_MTE1", 0)
                if part == 1:
                    T.ascend_set_flag("M_FIX", next_slot)
        if tick >= 1:
            for part in range(2):
                if part == 0:
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 18 + prev_slot)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 2 + prev_slot)
                T.ascend_wait_flag("M_MTE1", 1)
                T.copy(kv_l1[(tick + 2) % 3, part * 64 : (part + 1) * 64, :], vb, transpose=True)
                if part == 0:
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 25)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 9)
                T.copy(p_l1[prev_slot, :, part * 64 : (part + 1) * 64], pa)
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 1)
                T.gemm(pa, vb, pv_c[prev_slot, :, :], transpose_B=True, clear_accum=(part == 0))
                T.ascend_set_flag("M_MTE1", 1)
                if part == 1:
                    T.ascend_set_flag("M_FIX", 2)
            T.ascend_wait_flag("M_FIX", 2)
            T.dual_copy(pv_c[prev_slot, :, :], pv[prev_slot, :, :])
            T.ascend_cross_core_set_flag(4, "PIPE_FIX", 8)
            T.ascend_cross_core_set_flag(4, "PIPE_FIX", 24)
        if tick < tiles:
            T.ascend_wait_flag("M_FIX", next_slot)
            T.dual_copy(qk_c[next_slot, :, :], scores[next_slot, :, :])
            T.ascend_cross_core_set_flag(4, "PIPE_FIX", next_slot)
            T.ascend_cross_core_set_flag(4, "PIPE_FIX", next_slot + 16)
    T.ascend_wait_flag("M_MTE1", 0)
    T.ascend_wait_flag("M_MTE1", 1)
    T.ascend_wait_flag("MTE1_MTE2", 0)


@T.macro
def _relay(is_key, tile, core, sid, last_quad, kv, ints, nd, nz, kv_l1):
    parity = tile & 1
    for chunk in range(2):
        # Each relay issues exactly two chunks, so its ND and NZ slots always
        # start at zero and advance together. The 33rd NZ row is bank padding.
        tail = tile % 17 == 16 and chunk == 1
        for quad in range(8):
            if is_key:
                if chunk == 0 and quad == 0:
                    T.ascend_wait_flag("MTE2_S", parity)
                if tail and quad == 0:
                    T.ascend_wait_flag("MTE2_S", parity + 2)
            offset = parity * 128 + sid * 64 + chunk * 32 + quad * 4
            base = ints[INDEX + offset]
            all_valid = T.alloc_var("int32", init=1)
            if tail:
                for row in range(4):
                    all_valid = all_valid & ints[VALID + offset + row]
            if quad == 0:
                T.ascend_wait_flag("V_MTE2", chunk)
            if base <= last_quad and all_valid != 0:
                # Prepared indices are nonnegative, including masked slots.
                T.assume(base >= 0)
                start = T.cast(base, "int64")
                T.copy(kv[start : start + 4, core & 1, :], nd[chunk, quad * 4 : quad * 4 + 4, :])
            else:
                for row in range(4):
                    if not tail or ints[VALID + offset + row] != 0:
                        idx = ints[INDEX + offset + row]
                        T.assume(idx >= 0 and idx < kv.shape[0])
                        T.copy(kv[T.cast(idx, "int64"), core & 1, :], nd[chunk, quad * 4 + row, :])
            if quad == 7:
                T.ascend_set_flag("MTE2_V", chunk)
                if not is_key and chunk == 1:
                    T.ascend_set_flag("S_MTE2", parity)
        T.ascend_wait_flag("MTE3_V", chunk)
        T.ascend_wait_flag("MTE2_V", chunk)
        T.copy(nd[chunk, :, :], nz[chunk, :32, :])
        T.ascend_set_flag("V_MTE3", chunk)
        T.ascend_set_flag("V_MTE2", chunk)
        if not is_key and chunk == 0:
            T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", parity + 4)
        T.ascend_wait_flag("V_MTE3", chunk)
        row_start = sid * 64 + chunk * 32
        T.copy(nz[chunk, :32, :], kv_l1[tile % 3, row_start : row_start + 32, :])
        T.ascend_set_flag("MTE3_V", chunk)
        if chunk == 1:
            T.ascend_cross_core_set_flag(4, "PIPE_MTE3", parity + (6 if is_key else 2))


@T.macro
def _vector(
    core, sid, cores, tiles, last_quad, indices, valid, key, value, out, output, ints, f, acc, pv, scores, packed, nd, nz, kv_l1, p_l1
):
    for slot in (0, 1):
        T.ascend_set_flag("MTE3_V", slot)
    for slot in (0, 1, 2, 3):
        T.ascend_set_flag("V_MTE2", slot)
    for slot in (0, 1):
        T.ascend_set_flag("S_MTE2", slot)
    for tick in range(tiles + 5):
        if tick >= 5:
            T.ascend_wait_flag("V_MTE3", 2)
            if (tick - 5) % 17 == 16:
                task = core + (tick - 5) // 17 * cores
                T.assume(task < out.shape[0] * 2)
                T.copy(output, out[task // 2, task % 2, sid * 8 : sid * 8 + 8, :], l2_cache_ctrl=0)
                T.ascend_set_flag("MTE3_V", 4)
        if tick >= 4 and tick < tiles + 4:
            if (tick - 4) % 17 == 16:
                if tick > 20:
                    T.ascend_wait_flag("MTE3_V", 4)
                _normalize(f, acc, output)
            T.ascend_set_flag("V_MTE3", 2)
        if tick >= 3 and tick < tiles + 3:
            if (tick - 3) % 17 == 0:
                _initialize(f, acc)
            parity = (tick + 1) & 1
            T.ascend_cross_core_wait_flag(4, "PIPE_V", parity)
            _softmax(f, ints, scores, packed, parity)
            T.ascend_set_flag("V_MTE3", 3)
            T.ascend_set_flag("V_MTE2", parity + 2)
            T.ascend_wait_flag("V_MTE3", 3)
            T.copy(packed[parity, :8, :], p_l1[parity, sid * 8 : sid * 8 + 8, :])
            T.ascend_cross_core_set_flag(4, "PIPE_MTE3", 9)
        if tick >= 2 and tick < tiles + 2:
            _relay(False, tick - 2, core, sid, last_quad, value, ints, nd, nz, kv_l1)
        if tick >= 1 and tick < tiles + 1:
            tile = tick - 1
            parity = tile & 1
            query = (core + tile // 17 * cores) // 2
            T.ascend_wait_flag("V_MTE2", parity + 2)
            T.copy(valid[query, (tile % 17) * 128 : (tile % 17 + 1) * 128], ints[VALID + parity * 128 : VALID + (parity + 1) * 128])
            if tile % 17 == 16:
                T.ascend_set_flag("MTE2_S", parity + 2)
            _relay(True, tile, core, sid, last_quad, key, ints, nd, nz, kv_l1)
        if tick >= 3 and tick < tiles + 3:
            T.ascend_cross_core_wait_flag(4, "PIPE_V", 8)
            _accumulate(f, acc, pv, (tick + 1) & 1)
        if tick < tiles:
            parity = tick & 1
            query = (core + tick // 17 * cores) // 2
            T.ascend_wait_flag("S_MTE2", parity)
            T.copy(indices[query, (tick % 17) * 128 : (tick % 17 + 1) * 128], ints[INDEX + parity * 128 : INDEX + (parity + 1) * 128])
            T.ascend_set_flag("MTE2_S", parity)
    for slot in (0, 1):
        T.ascend_wait_flag("MTE3_V", slot)
    for slot in (0, 1, 2, 3):
        T.ascend_wait_flag("V_MTE2", slot)
    T.ascend_wait_flag("MTE3_V", 4)
    for slot in (0, 1):
        T.ascend_wait_flag("S_MTE2", slot)


def qsa_fwd(total_q, total_kv, *, aic_count=32, dynamic_shape=False):
    assert aic_count in SUPPORTED_AIC_COUNTS, "aic_count must be 28, 32, or 36"
    assert total_q > 0 and total_kv >= 4, "Q must be positive and KV must be at least four rows"
    if dynamic_shape:
        total_q = T.dynamic("num_tokens")
        padded_kv = T.dynamic("padded_kv_length")
    else:
        padded_kv = total_kv + 3
        assert max(total_q * 8192, padded_kv * 512) <= 2**31, "GM offsets must fit int32"

    @T.prim_func
    def main(
        Q: T.Buffer((total_q, NUM_KV_HEADS, PAD_M, HEAD_DIM), "bfloat16"),
        K: T.Buffer((padded_kv, NUM_KV_HEADS, HEAD_DIM), "bfloat16"),
        V: T.Buffer((padded_kv, NUM_KV_HEADS, HEAD_DIM), "bfloat16"),
        Indices: T.Buffer((total_q, PHYSICAL_TOP_K), "int32"),
        Valid: T.Buffer((total_q, PHYSICAL_TOP_K), "int32"),
        O: T.Buffer((total_q, NUM_KV_HEADS, PAD_M, HEAD_DIM), "bfloat16"),
    ):
        with T.Kernel(aic_count) as core:
            # Keep the allocation order of the manual pipeline. Each AIV owns
            # its UB tiles; the FIX engine splits each 16-row result in half.
            nz = T.alloc_shared((2, 33, 256), "bfloat16")
            nd = T.alloc_shared((2, 32, 256), "bfloat16")
            pv = T.alloc_shared((2, 8, 256), "float32")
            scores = T.alloc_shared((2, 8, 128), "float32")
            acc = T.alloc_shared((8, 256), "float32")
            packed = T.alloc_shared((2, 9, 128), "bfloat16")
            output = T.alloc_shared((8, 256), "bfloat16")
            ints = T.alloc_shared((512,), "int32")
            f = T.alloc_shared((256,), "float32")
            kv_l1 = T.alloc_l1((3, 128, 256), "bfloat16")
            q_l1 = T.alloc_l1((16, 256), "bfloat16")
            p_l1 = T.alloc_l1((2, 16, 128), "bfloat16")
            qa = T.alloc_l0a((16, 128), "bfloat16")
            pa = T.alloc_l0a((16, 64), "bfloat16")
            kb = T.alloc_l0b((128, 128), "bfloat16")
            vb = T.alloc_l0b((256, 64), "bfloat16")
            pv_c = T.alloc_l0c((2, 16, 256), "float32")
            qk_c = T.alloc_l0c((2, 16, 128), "float32")
            T.annotate_layout(
                {
                    nz: make_ascend_compact_nz_layout(nz),
                    packed: make_ascend_compact_nz_layout(packed),
                    kv_l1: make_ascend_major_k_layout(kv_l1),
                    q_l1: make_ascend_major_k_layout(q_l1),
                    p_l1: make_ascend_major_k_layout(p_l1),
                    qa: make_ascend_major_k_layout(qa),
                    pa: make_ascend_major_k_layout(pa),
                    kb: make_ascend_major_k_layout(kb),
                    vb: make_ascend_major_k_layout(vb),
                }
            )
            tasks = total_q * 2
            tiles = 17 * (T.truncdiv(tasks - 1 - core, aic_count) + 1)
            with T.Cube():
                if core < tasks:
                    _cube(core, aic_count, tiles, Q, kv_l1, q_l1, p_l1, qa, pa, kb, vb, qk_c, pv_c, scores, pv)
            with T.Vector() as sid:
                if core < tasks:
                    _vector(
                        core,
                        sid,
                        aic_count,
                        tiles,
                        padded_kv - 7,
                        Indices,
                        Valid,
                        K,
                        V,
                        O,
                        output,
                        ints,
                        f,
                        acc,
                        pv,
                        scores,
                        packed,
                        nd,
                        nz,
                        kv_l1,
                        p_l1,
                    )

    return main
