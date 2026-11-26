"""Sparse MLA Decode adapted from the shared-L0 Prefill kernel.

Cube and the two AIV subblocks follow the Prefill QK, SIMD softmax, PV and
SIMD accumulation stages. Decode gathers and dequantizes the selected KV
records before QK. The public Decode entry point maps each query to one
mixed block.
"""

import math
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache

import torch

H, D = 64, 512
BLOCK_K, FRAG_D, KV_TILE_ROWS = 64, 128, 128
SM_SCALE = 1.0 / math.sqrt(D)


@dataclass
class FlashMLASchedMeta:
    """Compatible scheduling metadata for the public Decode entry point."""

    have_initialized: bool = False
    config: tuple | None = None
    tile_scheduler_metadata: object = None
    num_splits: object = None


def get_mla_metadata(*args, **kwargs):
    return FlashMLASchedMeta(), None


def _decode_tiles(width, extra_topk):
    """KV tiles a single query visits (main + extra cache)."""
    block = BLOCK_K if max(width, extra_topk) <= BLOCK_K else KV_TILE_ROWS
    return (width + block - 1) // block + ((extra_topk + block - 1) // block if extra_topk else 0)


def _long_single_cache_stream(batch, width, extra_topk, cores):
    # Each query gets its own mixed block and amortizes the pipeline prologue
    # over at least sixteen full KV tiles. The index mask handles a half tail.
    return batch <= cores and extra_topk == 0 and width // KV_TILE_ROWS >= 16


def _short_single_cache_stream(batch, width, extra_topk, cores):
    # Up to two queries per block, each with two complete tiles.
    return cores <= batch < 2 * cores and extra_topk == 0 and width == 2 * KV_TILE_ROWS


def _auto_kv_pipeline(batch, width, extra_topk, cores, static_lengths, *, enable_batch_invariant=False):
    """Select pipelined gather or a static tile stream from launch geometry."""
    if extra_topk:
        # Device A/B covered four to five queries per core and one or two
        # complete packed tiles per query. Keep this first policy in that regime.
        kv_rows = width + extra_topk
        return (static_lengths and not enable_batch_invariant and batch >= 4 * cores
                and extra_topk % BLOCK_K == 0 and kv_rows % KV_TILE_ROWS == 0
                and kv_rows <= 2 * KV_TILE_ROWS)
    if (static_lengths and not enable_batch_invariant
            and (_long_single_cache_stream(batch, width, extra_topk, cores)
                 or _short_single_cache_stream(batch, width, extra_topk, cores))):
        return True
    # Static 128-row tiles share one stream across queries. With runtime lengths
    # the producer restarts per query, so keep the legacy multi-tile condition.
    return (batch >= 2 * cores
            and (_decode_tiles(width, 0) >= 2 or (static_lengths and width >= KV_TILE_ROWS)))


def _check_kv_split(batch, width, extra_topk, cores, kv_split):
    block = BLOCK_K if max(width, extra_topk) <= BLOCK_K else KV_TILE_ROWS
    main_tiles = (width + block - 1) // block
    extra_tiles = (extra_topk + block - 1) // block
    limit = max(1, min(cores // batch, main_tiles + extra_tiles))
    if type(kv_split) is not int or not 1 <= kv_split <= limit:
        raise ValueError(f"kv_split must be an integer in [1, {limit}]")


def _kv_dequantization_macros(stripe, fp8_pitch, direct_kv):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    kv_pitch = stripe + int(direct_kv)
    fp8_scales = D // 32
    fp4_scales = D // 16

    @T.macro
    def dequantize_fp8(raw_fp8, scale_ub, valid, first, nd):
        with T.SimdVF():
            full8 = S.pset(8, "PAT_ALL")
            full16 = S.pset(16, "PAT_ALL")
            full32 = S.pset(32, "PAT_ALL")
            lanes = T.reinterpret(S.vci(0, "int16"), "uint16x128")
            offsets = S.vadd(lanes, S.vmuls(S.vshrs(lanes, 4), T.uint16(fp8_pitch - fp8_scales)))
            # One gather decodes all 16 scale bytes from each of eight rows.
            for group in T.serial(stripe // 8):
                codes = S.vgather2(T.access_ptr(raw_fp8[group * 8, D], "r",
                                               extent=7 * fp8_pitch + fp8_scales), offsets, full16)
                bits = S.vshls(S.vmins(codes, T.uint16(224)), 7)
                S.vsts(scale_ub[group * 128], T.reinterpret(bits, "bfloat16x128"), full16, "NORM_B16")
            S.mem_bar("VST_VLD")
            high_bits = S.vdup(T.uint32(0xFFFF0000), "uint32", full32)
            if direct_kv:
                span = kv_pitch * 128
                nz0 = S.make_ubuf_ptr(T.access_ptr(nd[0, 0], "w", extent=span), "bfloat16")
                nz1 = S.make_ubuf_ptr(T.access_ptr(nd[0, 128], "w", extent=span), "bfloat16")
                nz2 = S.make_ubuf_ptr(T.access_ptr(nd[0, 256], "w", extent=span), "bfloat16")
                nz3 = S.make_ubuf_ptr(T.access_ptr(nd[0, 384], "w", extent=span), "bfloat16")
            for j in T.serial(stripe):
                flag = S.vld(valid[first + j], "BRC_B32")
                flag16 = T.reinterpret(S.vor(flag, S.vshls(flag, 16)), "int16x128")
                visible = S.vcmps(flag16, T.int16(0), full16, "ne")
                scales = T.alloc_local((4,), "bfloat16x128")
                first_scales = S.vld(scale_ub[j * fp8_scales], "E2B_B16")
                last_scales = S.vld(scale_ub[j * fp8_scales + 8], "E2B_B16")
                s0, s1 = S.vintlv(first_scales, first_scales)
                s2, s3 = S.vintlv(last_scales, last_scales)
                scales[0] = s0
                scales[1] = s1
                scales[2] = s2
                scales[3] = s3
                for chunk in T.unroll(D // 128):
                    col = chunk * 128
                    packed = T.reinterpret(S.vld(raw_fp8[j, col], "UNPK_B8"), "float8_e4m3fnx256")
                    even = S.vcvt(packed, "float32", full8, part=0)
                    odd = S.vcvt(packed, "float32", full8, part=2)
                    low = S.vshrs(T.reinterpret(even, "uint32x64"), 16)
                    high = S.vand(T.reinterpret(odd, "uint32x64"), high_bits)
                    values = T.reinterpret(S.vor(low, high), "bfloat16x128")
                    restored = S.vmul(values, scales[chunk], visible)
                    if direct_kv:
                        # Each 16-lane group lands at the next NZ column-group pitch.
                        stride = (kv_pitch << 16) | 1
                        if chunk == 0:
                            nz0 = S.vsstb(restored, nz0, stride, full16, update=True)
                        elif chunk == 1:
                            nz1 = S.vsstb(restored, nz1, stride, full16, update=True)
                        elif chunk == 2:
                            nz2 = S.vsstb(restored, nz2, stride, full16, update=True)
                        else:
                            nz3 = S.vsstb(restored, nz3, stride, full16, update=True)
                    else:
                        S.vsts(nd[j, col], restored, full16, "NORM_B16")

    @T.macro
    def dequantize_fp4(raw, scale_ub, valid, first, nd):
        with T.SimdVF():
            full16 = S.pset(16, "PAT_ALL")
            full32 = S.pset(32, "PAT_ALL")
            lanes = T.reinterpret(S.vci(0, "int16"), "uint16x128")
            offsets = S.vadd(lanes, S.vmuls(S.vshrs(lanes, 5), T.uint16(fp8_pitch - fp4_scales)))
            high_bits = S.vdup(T.uint32(0xFFFF0000), "uint32", full32)
            for group in T.serial(stripe // 4):
                codes = S.vgather2(T.access_ptr(raw[group * 4, D // 2], "r",
                                               extent=3 * fp8_pitch + fp4_scales), offsets, full16)
                packed = T.reinterpret(codes, "float8_e4m3fnx256")
                even = S.vcvt(packed, "float32", full16, part=0)
                odd = S.vcvt(packed, "float32", full16, part=2)
                low = S.vshrs(T.reinterpret(even, "uint32x64"), 16)
                high = S.vand(T.reinterpret(odd, "uint32x64"), high_bits)
                scales = T.reinterpret(S.vor(low, high), "bfloat16x128")
                S.vsts(scale_ub[group * 128], scales, full16, "NORM_B16")
            S.mem_bar("VST_VLD")
            if direct_kv:
                span = kv_pitch * 128
                nz0 = S.make_ubuf_ptr(T.access_ptr(nd[0, 0], "w", extent=span), "bfloat16")
                nz1 = S.make_ubuf_ptr(T.access_ptr(nd[0, 128], "w", extent=span), "bfloat16")
                nz2 = S.make_ubuf_ptr(T.access_ptr(nd[0, 256], "w", extent=span), "bfloat16")
                nz3 = S.make_ubuf_ptr(T.access_ptr(nd[0, 384], "w", extent=span), "bfloat16")
            for j in T.serial(stripe):
                flag = S.vld(valid[first + j], "BRC_B32")
                flag16 = T.reinterpret(S.vor(flag, S.vshls(flag, 16)), "int16x128")
                visible = S.vcmps(flag16, T.int16(0), full16, "ne")
                for chunk in T.unroll(D // 128):
                    col = chunk * 128
                    scale = S.vld(scale_ub[j * fp4_scales + chunk * 8], "E2B_B16")
                    packed = T.reinterpret(S.vld(raw[j, col // 2], "UNPK4_B8"), "float4_e2m1fnx512")
                    value = S.vcvt(packed, "bfloat16", full16, part=0)
                    restored = S.vmul(value, scale, visible)
                    if direct_kv:
                        stride = (kv_pitch << 16) | 1
                        if chunk == 0:
                            nz0 = S.vsstb(restored, nz0, stride, full16, update=True)
                        elif chunk == 1:
                            nz1 = S.vsstb(restored, nz1, stride, full16, update=True)
                        elif chunk == 2:
                            nz2 = S.vsstb(restored, nz2, stride, full16, update=True)
                        else:
                            nz3 = S.vsstb(restored, nz3, stride, full16, update=True)
                    else:
                        S.vsts(nd[j, col], restored, full16, "NORM_B16")

    return dequantize_fp8, dequantize_fp4


def build_decode_kernel(batch, nc, page_size, width, *, ne=0, extra_page_size=2,
                        extra_width=528, extra_topk=0, softmax_scale=SM_SCALE, cores=28,
                        kv_split=1, full_length=False, kv_pipeline=False, balance_rows=False,
                        packed_lookahead=False, packed_pair_dma=False):
    """Build mixed blocks over query rows or disjoint slices of their KV tiles.

    Opt in to paired DMA for the long packed schedule when comparing gather
    implementations.
    """
    if min(batch, nc, cores) <= 0 or page_size <= 1 or width <= 0 or width % BLOCK_K:
        raise ValueError("Require B,Nc,cores>0, P>1, T>0 and T%64==0")
    if extra_width not in (528, 288):
        raise ValueError("Extra cache bytes/token must be 528 or 288")
    if ne < 0 or extra_topk < 0 or bool(ne) != bool(extra_topk):
        raise ValueError("ne and extra_topk must either both be positive or both be zero")
    if ne and extra_page_size < 1:
        raise ValueError("Extra cache page size must be positive")
    if kv_pipeline and extra_topk and not (full_length and kv_split == 1):
        # The packed dual-cache producer can stream only with a static tile
        # table; otherwise the single-cache gather path would be wrong.
        raise ValueError("kv_pipeline with an extra KV cache requires full_length and kv_split=1")
    if balance_rows and (full_length or kv_split != 1):
        raise ValueError("Row balancing requires runtime lengths and kv_split=1")
    if packed_lookahead and (not extra_topk or full_length or kv_split != 1):
        raise ValueError("Packed lookahead requires two caches, runtime lengths and kv_split=1")
    scale = _check_scale(softmax_scale)
    _check_kv_split(batch, width, extra_topk, cores, kv_split)

    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S
    from tilelang.layout import (
        make_ascend_compact_nz_layout, make_ascend_l0c_layout, make_ascend_major_k_layout,
    )

    packed_tile = bool(extra_topk) and kv_split == 1
    direct_kv = packed_tile or kv_pipeline
    block = KV_TILE_ROWS if packed_tile or max(width, extra_topk) > BLOCK_K else BLOCK_K
    full_tiles = full_length and width % block == 0 and extra_topk % block == 0
    stream_tiles = kv_pipeline and full_length and kv_split == 1
    # Dual-cache (packed) tiles stream across queries through the same
    # producer/consumer chain: one 128-row tile per iteration, odd tails padded.
    stream_packed = packed_tile and full_length and kv_split == 1 and kv_pipeline
    short_stream = stream_tiles and _short_single_cache_stream(batch, width, extra_topk, cores)
    stream = (stream_tiles or stream_packed) and not short_stream
    # Target long, dynamic FP4 work with one or two queries per core.
    packed_staged = (packed_lookahead and balance_rows and cores < batch < 2 * cores
                     and extra_width == 288 and min(width, extra_topk) >= 8 * KV_TILE_ROWS)
    packed_halves = ((width + BLOCK_K - 1) // BLOCK_K
                     + ((extra_topk + BLOCK_K - 1) // BLOCK_K if extra_topk else 0))
    tiles_per_query = (width + block - 1) // block
    steps_per_query = (packed_halves + 1) // 2 if stream_packed else tiles_per_query
    q_slots, q_half, q_flag = 3, D // 2, 5
    tokens = nc * page_size
    extra_tokens = ne * extra_page_size if ne else 1
    tasks = batch * kv_split
    grid = min(tasks, cores)
    output_rows = tasks * H
    output_dtype = "float32" if kv_split > 1 else "bfloat16"
    half, pitch = H // 2, H // 2 + 1
    stripe = 16
    raw_lead = 2 if stream and block // (2 * stripe) > 2 else 1
    kv_pitch = stripe + int(direct_kv)
    workspace_shape = (1,) if direct_kv else (grid, 2, block // 2, D)
    fp8_scales = D // 32
    fp8_bytes = D + fp8_scales
    fp8_pitch = (fp8_bytes + 31) // 32 * 32
    invalid_offset = (1 << 31) - 1
    # Preserve per-record DMA in the long-query and two-tile schedules.
    long_query_stream = stream_tiles and _long_single_cache_stream(batch, width, extra_topk, cores)
    pair_dma = kv_pipeline and not (long_query_stream or short_stream) and tokens * fp8_bytes <= invalid_offset
    packed_pair_dma = (packed_pair_dma and packed_staged and tokens * fp8_bytes <= invalid_offset
                       and extra_tokens * extra_width <= invalid_offset)

    fp4_scales = D // 16

    dequantize_fp8, dequantize_fp4 = _kv_dequantization_macros(stripe, fp8_pitch, direct_kv)

    @T.macro
    def load_records(Cache, index_ub, raw, first, n, cache_tokens, record_bytes):
        for j in T.serial(stripe):
            col = first + j
            index = index_ub[col]
            if (col < n) & (index >= 0) & (index < cache_tokens):
                T.copy(Cache[T.cast(index, "int64"), :], raw[j, :record_bytes])

    @T.macro
    def prepare_indices(Indices, row, local_tile, n, rows, index_ub, valid, index_width, cache_tokens,
                        pair_offsets=None):
        start = local_tile * block
        if index_width % block == 0:
            T.copy(Indices[row, start:start + block], index_ub)
        else:
            tail = index_width % block
            if start + block <= index_width:
                T.copy(Indices[row, start:start + block], index_ub)
            else:
                with T.SimdVF():
                    full = S.pset(32, "PAT_ALL")
                    zero = S.vdup(T.int32(0), "int32", full)
                    for chunk in T.serial(block // BLOCK_K):
                        S.vsts(index_ub[chunk * BLOCK_K], zero, full)
                T.copy(Indices[row, start:start + tail], index_ub[:tail])
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            one = S.vdup(T.int32(1), "int32", full)
            zero = S.vdup(T.int32(0), "int32", full)
            for chunk in T.serial(rows // BLOCK_K):
                index = S.vld(index_ub[chunk * BLOCK_K])
                positions = S.vci(chunk * BLOCK_K, "int32")
                visible = S.vcmps(positions, n, full, "lt")
                lower = S.vcmps(index, T.int32(0), full, "ge")
                upper = S.vcmps(index, T.int32(min(cache_tokens - 1, 2147483647)), full, "le")
                mask = S.pand(visible, S.pand(lower, upper, full), full)
                S.vsts(valid[chunk * BLOCK_K], S.vsel(one, zero, mask), full)
                if pair_offsets is not None:
                    sentinel = S.vdup(T.int32(invalid_offset), "int32", full)
                    offsets = S.vsel(S.vmuls(index, T.int32(fp8_bytes), full), sentinel, mask)
                    even, odd = S.vdintlv(offsets, offsets)
                    pairs = S.pset(32, "PAT_VL32")
                    lo = S.vmin(even, odd, pairs)
                    hi = S.vmax(even, odd, pairs)
                    # A single valid token fills both slots. The validity mask
                    # suppresses the duplicate in its original invalid lane.
                    hi = S.vsel(lo, hi, S.vcmps(hi, T.int32(invalid_offset), pairs, "eq"))
                    S.vsts(pair_offsets[0, chunk * 32], lo, pairs, extent=32)
                    S.vsts(pair_offsets[1, chunk * 32], S.vsub(hi, lo, pairs), pairs, extent=32)

    @T.macro
    def load_fp8_pairs(Cache, offsets, raw_fp8, first):
        # When both rows are valid their order within the pair is immaterial
        # to QK/softmax/PV. Sorting enables one increasing-stride DMA.
        for pair in T.serial(stripe // 2):
            col = first // 2 + pair
            base = offsets[0, col]
            if base != invalid_offset:
                stride = T.alloc_var("int64", init=T.cast(offsets[1, col], "int64"))
                records = T.StridedTensor((tokens * fp8_bytes, 2, fp8_bytes), (1, stride, 1),
                                          "uint8", data=Cache.data)
                T.copy(records[base, :, :], raw_fp8[2 * pair:2 * pair + 2, :fp8_bytes])

    @T.macro
    def load_packed_pairs(Cache, offsets, raw, first, cache_tokens, record_bytes):
        for pair in T.serial(stripe // 2):
            col = first // 2 + pair
            base = T.cast(offsets[0, col], "int64")
            stride = T.alloc_var("int64", init=T.cast(offsets[1, col], "int64"))
            if stride > 0:
                records = T.StridedTensor((cache_tokens * record_bytes, 2, record_bytes), (1, stride, 1),
                                          "uint8", data=Cache.data)
                T.copy(records[base, :, :], raw[2 * pair:2 * pair + 2, :record_bytes])
            elif base != invalid_offset:
                # Zero is a repeated address. Negative codes identify the
                # valid lane so the record reaches its original slot.
                record = T.StridedTensor((cache_tokens * record_bytes, record_bytes), (1, 1),
                                         "uint8", data=Cache.data)
                if stride != -2:
                    T.copy(record[base, :], raw[2 * pair, :record_bytes])
                if stride != -1:
                    T.copy(record[base, :], raw[2 * pair + 1, :record_bytes])

    @T.macro
    def gather_fp8_to_l1(Cache, n, sid, index_ub, valid, raw_fp8, scale_ub, nd, kv_l1, pair_offsets):
        # Raw slots cover the DMA-to-dequantization distance. The NZ store
        # follows dequantization by one stripe and needs two ND slots.
        for group in T.Pipelined(block // (2 * stripe), num_stages=raw_lead + 2,
                                 annotations={"enable_offset": True}):
            # Alternate stripes across AIVs so a short prefix does not land
            # entirely on one subblock. Preserve their logical L1 row offsets.
            first = (2 * group + sid) * stripe
            with T.Stage(0):
                if pair_dma:
                    load_fp8_pairs(Cache, pair_offsets, raw_fp8, first)
                else:
                    load_records(Cache, index_ub, raw_fp8, first, n, tokens, fp8_bytes)
            with T.Stage(raw_lead):
                dequantize_fp8(raw_fp8, scale_ub, valid, first, nd)
            with T.Stage(raw_lead + 1):
                T.copy(nd[:stripe, :], kv_l1[first:first + stripe, :])

    @T.macro
    def gather(Cache, Indices, row, local_tile, n, rows, bx, sid, index_ub, valid,
               nd, raw_fp8, scale_ub, KVWorkspace, index_width, cache_tokens, record_bytes):
        prepare_indices(Indices, row, local_tile, n, rows, index_ub, valid, index_width, cache_tokens)
        for group in T.serial(rows // (2 * stripe)):
            offset = group * stripe
            first = sid * (rows // 2) + offset
            load_records(Cache, index_ub, raw_fp8, first, n, cache_tokens, record_bytes)
            if record_bytes == fp8_bytes:
                dequantize_fp8(raw_fp8, scale_ub, valid, first, nd)
            else:
                dequantize_fp4(raw_fp8, scale_ub, valid, first, nd)
            T.copy(nd, KVWorkspace[bx, sid, offset:offset + stripe, :])

    @T.macro
    def prepare_half(Indices, row, piece, n, half_start, index_ub, valid, index_width, cache_tokens,
                     record_bytes, pair_offsets):
        start = piece * BLOCK_K
        if n > 0:
            if index_width % BLOCK_K == 0 or start + BLOCK_K <= index_width:
                T.copy(Indices[row, start:start + BLOCK_K], index_ub[half_start:half_start + BLOCK_K])
            else:
                tail = index_width % BLOCK_K
                T.copy(Indices[row, start:start + tail], index_ub[half_start:half_start + tail])
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            index = S.vld(index_ub[half_start])
            visible = S.vcmps(S.vci(0, "int32"), n, full, "lt")
            lower = S.vcmps(index, T.int32(0), full, "ge")
            upper = S.vcmps(index, T.int32(min(cache_tokens - 1, 2147483647)), full, "le")
            mask = S.pand(visible, S.pand(lower, upper, full), full)
            one = S.vdup(T.int32(1), "int32", full)
            zero = S.vdup(T.int32(0), "int32", full)
            S.vsts(valid[half_start], S.vsel(one, zero, mask), full)
            if pair_offsets is not None:
                sentinel = S.vdup(T.int32(invalid_offset), "int32", full)
                offsets = S.vsel(S.vmuls(index, T.int32(record_bytes), full), sentinel, mask)
                even, odd = S.vdintlv(offsets, offsets)
                pairs = S.pset(32, "PAT_VL32")
                # Two valid lanes can exchange positions as one complete KV record each.
                lo = S.vmin(even, odd, pairs)
                hi = S.vmax(even, odd, pairs)
                missing = S.vcmps(hi, T.int32(invalid_offset), pairs, "eq")
                even_missing = S.vcmps(even, T.int32(invalid_offset), pairs, "eq")
                even_only = S.vdup(T.int32(-1), "int32", pairs)
                odd_only = S.vdup(T.int32(-2), "int32", pairs)
                single = S.vsel(odd_only, even_only, even_missing)
                stride = S.vsel(single, S.vsub(hi, lo, pairs), missing)
                S.vsts(pair_offsets[0, half_start // 2], lo, pairs, extent=BLOCK_K // 2)
                S.vsts(pair_offsets[1, half_start // 2], stride, pairs, extent=BLOCK_K // 2)

    @T.macro
    def prepare_packed_indices(Indices, ExtraIndices, row, tile, main_length, extra_length,
                               main_halves, rows, index_ub, valid, pair_offsets=None):
        for part in T.serial(rows // BLOCK_K):
            piece = tile * 2 + part
            if piece < main_halves:
                n = T.min(main_length - piece * BLOCK_K, BLOCK_K)
                prepare_half(Indices, row, piece, n, part * BLOCK_K, index_ub, valid,
                             width, tokens, fp8_bytes, pair_offsets)
            else:
                extra_piece = piece - main_halves
                n = T.max(T.min(extra_length - extra_piece * BLOCK_K, BLOCK_K), 0)
                prepare_half(ExtraIndices, row, extra_piece, n, part * BLOCK_K, index_ub, valid,
                             extra_topk, extra_tokens, extra_width, pair_offsets)

    @T.macro
    def gather_packed(Cache, Extra, Indices, ExtraIndices, row, tile, main_length, extra_length,
                      main_halves, rows, sid, index_ub, valid, raw, scale_ub, nd, kv_l1,
                      prepare=True, pair_offsets=None):
        if prepare:
            prepare_packed_indices(Indices, ExtraIndices, row, tile, main_length, extra_length,
                                   main_halves, rows, index_ub, valid, pair_offsets)
        # Each AIV gathers half of each 64-row piece. MTE2, Vector and MTE3
        # advance on different stripes with two raw slots and two BF16 slots.
        # Keep static group bounds for the NZ copy and the buffer reuse flags.
        for group in T.Pipelined(KV_TILE_ROWS // (2 * stripe), num_stages=3,
                                 annotations={"enable_offset": True}):
            part = group // (BLOCK_K // (2 * stripe))
            piece = tile * 2 + part
            from_main = piece < main_halves
            active = group * (2 * stripe) < rows
            # Alternate disjoint L1 stripes between the two AIVs.
            first = (2 * group + sid) * stripe
            with T.Stage(0):
                if active:
                    if from_main:
                        if pair_offsets is not None:
                            load_packed_pairs(Cache, pair_offsets, raw, first, tokens, fp8_bytes)
                        else:
                            end = part * BLOCK_K + T.min(main_length - piece * BLOCK_K, BLOCK_K)
                            load_records(Cache, index_ub, raw, first, end, tokens, fp8_bytes)
                    else:
                        if pair_offsets is not None:
                            load_packed_pairs(Extra, pair_offsets, raw, first, extra_tokens, extra_width)
                        else:
                            end = part * BLOCK_K + T.max(T.min(extra_length - (piece - main_halves) * BLOCK_K, BLOCK_K), 0)
                            load_records(Extra, index_ub, raw, first, end, extra_tokens, extra_width)
            with T.Stage(1):
                if active:
                    if extra_width == fp8_bytes or from_main:
                        dequantize_fp8(raw, scale_ub, valid, first, nd)
                    else:
                        dequantize_fp4(raw, scale_ub, valid, first, nd)
            with T.Stage(2):
                if active:
                    T.copy(nd[:stripe, :], kv_l1[first:first + stripe, :])

    @T.macro
    def prefetch_query(Q, row, q_l1, q_slot):
        for q_part in T.unroll(2, explicit=True):
            slot = (q_slot + q_part) % q_slots
            with T.Task():
                T.ascend_wait_flag("MTE1_MTE2", q_flag + slot)
                T.copy(Q[row * H:(row + 1) * H, q_part * q_half:(q_part + 1) * q_half],
                       q_l1[slot, :, :])
                T.ascend_set_flag("MTE2_MTE1", q_flag + slot)

    @T.macro
    def qk(q_l1, kv_l1, qa, kb, scores, score_ub, rows, q_slot=None, local_tile=0, steps=None):
        for part in T.serial(D // FRAG_D):
            if q_slot is None:
                T.copy(q_l1[:, part * FRAG_D:(part + 1) * FRAG_D], qa)
            else:
                panels = q_half // FRAG_D
                slot = (q_slot + part // panels) % q_slots
                column = (part % panels) * FRAG_D
                with T.Task():
                    if (local_tile == 0) & (part % panels == 0):
                        T.ascend_wait_flag("MTE2_MTE1", q_flag + slot)
                    T.copy(q_l1[slot, :, column:column + FRAG_D], qa)
                    last = tiles_per_query if steps is None else steps
                    if (local_tile == last - 1) & (part % panels == panels - 1):
                        T.ascend_set_flag("MTE1_MTE2", q_flag + slot)
            T.copy(kv_l1[:rows, part * FRAG_D:(part + 1) * FRAG_D], kb[:rows, :])
            T.gemm(qa, kb[:rows, :], scores[:, :rows], transpose_B=True, clear_accum=(part == 0))
        T.dual_copy(scores[:, :rows], score_ub[:, :rows])

    @T.macro
    def softmax(score_ub, valid, prob_ub, p_l1, m, ell, alpha, sink, stats, p_scale, rows,
                result_state=None):
        raw = T.Tensor((pitch * block,), "bfloat16", prob_ub.data, scope="shared.dyn")
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            heads = S.pset(32, "PAT_VL32")
            scalar = S.pset(32, "PAT_VL1")
            p_mask = S.pset(16, "PAT_VL64")
            zero = S.vdup(T.float32(0), "float32", full)
            one = S.vdup(T.float32(1), "float32", full)
            neg_inf = S.vdup(-T.infinity("float32"), "float32", full)
            valid0 = S.vcmps(S.vld(valid[0]), T.int32(0), full, "ne")
            if rows == KV_TILE_ROWS:
                valid1 = S.vcmps(S.vld(valid[BLOCK_K]), T.int32(0), full, "ne")
            for h in T.serial(half):
                first = S.vsel(S.vld(score_ub[h, 0]), neg_inf, valid0)
                if rows == KV_TILE_ROWS:
                    second = S.vsel(S.vld(score_ub[h, BLOCK_K]), neg_inf, valid1)
                    first = S.vmax(first, second, full)
                S.vsts(stats[h], S.vcmax(first, full), scalar, "ONEPT_B32")
            S.mem_bar("VST_VLD")

            # LSE tracks the KV-only state. Anchor the PV numerator to the
            # maximum of that state and the sink, as in the Prefill variant.
            old_m = S.vld(m[0])
            new_m = S.vmax(old_m, S.vmuls(S.vld(stats[0]), T.float32(scale), heads), heads)
            empty = S.vcmps(new_m, -T.infinity("float32"), heads, "eq")
            origin = S.vsel(zero, new_m, empty)
            old_empty = S.vcmps(old_m, -T.infinity("float32"), heads, "eq")
            factor = S.vsel(zero, S.vexp(S.vsub(old_m, origin, heads), heads), old_empty)
            sink_value = S.vld(sink[0])
            infinite_sink = S.vcmps(sink_value, T.infinity("float32"), heads, "eq")
            output_m = S.vmax(new_m, sink_value, heads)
            old_output_m = S.vmax(old_m, sink_value, heads)
            degenerate = S.por(infinite_sink,
                               S.vcmps(output_m, -T.infinity("float32"), heads, "eq"), heads)
            output_origin = S.vsel(zero, output_m, degenerate)
            output_alpha = S.vsel(
                one, S.vexp(S.vsub(old_output_m, output_origin, heads), heads), degenerate)
            output_scale = S.vsel(
                zero, S.vexp(S.vsub(new_m, output_origin, heads), heads), infinite_sink)
            S.vsts(p_scale[0], output_scale, heads)
            S.vsts(m[0], new_m, full)
            S.vsts(alpha[0], output_alpha, heads)
            S.vsts(stats[0], origin, full)
            S.mem_bar("VST_VLD")

            p0_ptr = S.make_ubuf_ptr(T.access_ptr(raw[0], "w", extent=4 * pitch * 16), "bfloat16")
            if rows == KV_TILE_ROWS:
                offset = 4 * pitch * 16
                p1_ptr = S.make_ubuf_ptr(T.access_ptr(raw[offset], "w", extent=offset), "bfloat16")
            for h in T.serial(half):
                origin_h = S.vld(stats[h], "BRC_B32")
                scale_h = S.vld(p_scale[h], "BRC_B32")
                score0 = S.vmuls(S.vld(score_ub[h, 0]), T.float32(scale), full)
                p0 = S.vsel(S.vexp(S.vsub(score0, origin_h, full), full), zero, valid0)
                row_prob = p0
                if rows == KV_TILE_ROWS:
                    score1 = S.vmuls(S.vld(score_ub[h, BLOCK_K]), T.float32(scale), full)
                    p1 = S.vsel(S.vexp(S.vsub(score1, origin_h, full), full), zero, valid1)
                    row_prob = S.vadd(p0, p1, full)
                S.vsts(stats[h], S.vcadd(row_prob, full), scalar, "ONEPT_B32")
                bf0 = S.vcvt(S.vmul(p0, scale_h, full), "bfloat16", full, sat=False, part=0)
                packed0 = T.reinterpret(S.vpack(T.reinterpret(bf0, "uint32x64"), 0), "bfloat16x128")
                p0_ptr = S.vsstb(packed0, p0_ptr, (pitch << 16) | 1, p_mask, update=True)
                if rows == KV_TILE_ROWS:
                    bf1 = S.vcvt(S.vmul(p1, scale_h, full), "bfloat16", full, sat=False, part=0)
                    packed1 = T.reinterpret(S.vpack(T.reinterpret(bf1, "uint32x64"), 0), "bfloat16x128")
                    p1_ptr = S.vsstb(packed1, p1_ptr, (pitch << 16) | 1, p_mask, update=True)
            S.mem_bar("VST_VLD")
            total = S.vadd(S.vmul(factor, S.vld(ell[0]), heads), S.vld(stats[0]), heads)
            S.vsts(ell[0], total, full)
            if result_state is not None:
                final_m, final_ell = result_state
                S.vsts(final_m[0], new_m, heads)
                S.vsts(final_ell[0], total, heads)
        T.dual_copy(prob_ub[:half, :rows], p_l1[:, :rows])

    @T.macro
    def pv_fragment(kv_l1, p_l1, pa, vb, product_parts, rows, part):
        T.copy(p_l1[:, :rows], pa[:, :rows])
        T.copy(kv_l1[:rows, part * FRAG_D:(part + 1) * FRAG_D], vb[:, :rows], transpose=True)
        T.gemm(pa[:, :rows], vb[:, :rows], product_parts[0 if stream else part, :, :],
               transpose_B=True, clear_accum=True)

    @T.macro
    def pv(kv_l1, p_l1, pa, vb, product_parts, rows):
        for part in T.serial(D // FRAG_D):
            pv_fragment(kv_l1, p_l1, pa, vb, product_parts, rows, part)

    @T.macro
    def update_output(tile, product_ub, accum, alpha, first=0):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            for h in T.serial(half):
                if tile == 0:
                    for chunk in T.serial(product_ub.shape[1] // BLOCK_K):
                        col = chunk * BLOCK_K
                        S.vsts(accum[h, first + col], S.vld(product_ub[h, col]), full)
                else:
                    factor = S.vld(alpha[h], "BRC_B32")
                    for chunk in T.serial(product_ub.shape[1] // BLOCK_K):
                        col = chunk * BLOCK_K
                        previous = S.vld(accum[h, first + col])
                        current = S.vld(product_ub[h, col])
                        S.vsts(accum[h, first + col], S.vadd(S.vmul(previous, factor, full), current, full), full)

    @T.macro
    def stream_pv(tile, kv_l1, p_l1, pa, vb, product_parts, product_ub, accum, alpha, rows):
        # FIXP publishes each product panel while MTE1 and Cube advance to
        # the next one. Vector consumes the previous panel in the other UB slot.
        for part in T.Pipelined(D // FRAG_D, num_stages=2, annotations={"enable_offset": True}):
            with T.Stage(0):
                pv_fragment(kv_l1, p_l1, pa, vb, product_parts, rows, part)
                T.dual_copy(product_parts[0, :, :], product_ub)
            with T.Stage(1):
                update_output(tile, product_ub, accum, alpha, part * FRAG_D)

    @T.macro
    def normalize_output(accum, out_ub, inv_ub, first, full, slot=None):
        for h in T.serial(half):
            factor = S.vld(inv_ub[h], "BRC_B32")
            for chunk in T.serial(out_ub.shape[-1] // BLOCK_K):
                col = chunk * BLOCK_K
                output = S.vmul(S.vld(accum[h, first + col]), factor, full)
                if kv_split == 1:
                    packed = S.vcvt(output, "bfloat16", full, sat=False, part=0)
                    if slot is None:
                        S.vsts(out_ub[h, col], packed, full, "PK_B32")
                    else:
                        S.vsts(out_ub[slot, h, col], packed, full, "PK_B32")
                else:
                    S.vsts(out_ub[h, col], output, full)

    @T.macro
    def finish(row, O, LSE, accum, out_ub, m, ell, sink, lse_ub, inv_ub, sid=0):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            heads = S.pset(32, "PAT_VL32")
            zero = S.vdup(T.float32(0), "float32", full)
            one = S.vdup(T.float32(1), "float32", full)
            inf = S.vdup(T.infinity("float32"), "float32", full)
            maximum = S.vld(m[0])
            total = S.vld(ell[0])
            nonempty = S.vcmps(total, T.float32(0), heads, "gt")
            safe_max = S.vsel(maximum, zero, nonempty)
            safe_total = S.vsel(total, one, nonempty)
            empty_lse = inf if kv_split == 1 else S.vdup(-T.infinity("float32"), "float32", full)
            lse = S.vsel(S.vadd(safe_max, S.vln(safe_total, heads), heads), empty_lse, nonempty)
            if kv_split == 1:
                sink_value = S.vld(sink[0])
                infinite_sink = S.vcmps(sink_value, T.infinity("float32"), heads, "eq")
                output_max = S.vsel(zero, S.vmax(safe_max, sink_value, heads), infinite_sink)
                kv_term = S.vmul(safe_total, S.vexp(S.vsub(safe_max, output_max, heads), heads), heads)
                denominator = S.vadd(kv_term, S.vexp(S.vsub(sink_value, output_max, heads), heads), heads)
            else:
                denominator = safe_total
            inverse = S.vsel(S.vdiv(one, denominator, heads, precision="exact"), zero, nonempty)
            S.vsts(lse_ub[0], lse, heads, extent=half)
            S.vsts(inv_ub[0], inverse, heads, extent=half)
            S.mem_bar("VST_VLD")
            if not stream:
                normalize_output(accum, out_ub, inv_ub, 0, full)
        if stream:
            for part in T.Pipelined(D // FRAG_D, num_stages=2, annotations={"enable_offset": True}):
                first = part * FRAG_D
                with T.Stage(0):
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        normalize_output(accum, out_ub, inv_ub, first, full, part % 2)
                with T.Stage(1):
                    head = row * H + sid * half
                    T.copy(out_ub[part % 2, :, :], O[head:head + half, first:first + FRAG_D])
        else:
            T.dual_copy(out_ub, O[row * H:(row + 1) * H, :])
        T.dual_copy(lse_ub, LSE[row * H:(row + 1) * H])

    @T.macro
    def row_work(Lengths, ExtraLengths, row):
        # Count the Cube rows issued in units of a half tile for packed and
        # non-pipelined paths, or in full tiles for the pipelined FP8 gather.
        main = T.ceildiv(Lengths[row], block if kv_pipeline else BLOCK_K)
        extra = T.ceildiv(ExtraLengths[row], BLOCK_K) if packed_tile else 0
        return T.max(main + extra, 1)

    @T.macro
    def row_bytes(Lengths, ExtraLengths, row):
        main = T.cast(Lengths[row], "int64") * fp8_bytes
        extra = T.cast(ExtraLengths[row], "int64") * extra_width if packed_tile else 0
        return main + extra

    @T.macro
    def count_row_pairs(work, payload, work_limit, byte_limit):
        pairs = T.alloc_var("int32", init=0)
        next_row = T.alloc_var("int32", init=0)
        for row in T.serial(batch - 1):
            if (row == next_row) & (work[row] + work[row + 1] <= work_limit) & (
                    payload[row] + payload[row + 1] <= byte_limit):
                pairs = pairs + 1
                next_row = row + 2
            else:
                next_row = T.max(next_row, row + 1)
        return pairs

    @T.macro
    def balance_single_and_double_rows(Lengths, ExtraLengths, bx, row_range):
        # With one or two queries per core, only the locations of the adjacent
        # pairs can change. The first eligible pair leaves the longest suffix,
        # so a greedy pass finds the maximum number of disjoint pairs.
        work = T.alloc_local((batch,), "int32")
        payload = T.alloc_local((batch,), "int64")
        pairs_needed = batch - grid
        lower = T.alloc_var("int32", init=0)
        upper = T.alloc_var("int32", init=0)
        byte_limit = T.alloc_var("int64", init=0)
        for row in T.serial(batch):
            work[row] = row_work(Lengths, ExtraLengths, row)
            payload[row] = row_bytes(Lengths, ExtraLengths, row)
            lower = T.max(lower, work[row])
            if row < 2 * pairs_needed:
                if row % 2 == 1:
                    upper = T.max(upper, work[row - 1] + work[row])
                    byte_limit = T.max(byte_limit, payload[row - 1] + payload[row])
            else:
                upper = T.max(upper, work[row])
                byte_limit = T.max(byte_limit, payload[row])
        old_peak = upper + 0
        for step in T.serial((2 * packed_halves).bit_length()):
            if lower < upper:
                limit = lower + (upper - lower) // 2
                if count_row_pairs(work, payload, limit, byte_limit) >= pairs_needed:
                    upper = limit
                else:
                    lower = limit + 1
        if upper < old_peak:
            pair_budget = T.alloc_var("int32", init=pairs_needed)
            assigned = T.alloc_var("int32", init=0)
            next_row = T.alloc_var("int32", init=0)
            for row in T.serial(batch):
                if (row == next_row) & (assigned <= bx):
                    size = T.alloc_var("int32", init=1)
                    if (pair_budget > 0) & (row + 1 < batch):
                        if (work[row] + work[row + 1] <= upper) & (
                                payload[row] + payload[row + 1] <= byte_limit):
                            size = 2
                            pair_budget = pair_budget - 1
                    if assigned == bx:
                        row_range[0] = row
                        row_range[1] = size
                    assigned = assigned + 1
                    next_row = row + size

    @T.macro
    def balance_pair(Lengths, ExtraLengths, first, middle, last, cut):
        total_work = T.alloc_var("int64", init=0)
        total_bytes = T.alloc_var("int64", init=0)
        left_work = T.alloc_var("int64", init=0)
        left_bytes = T.alloc_var("int64", init=0)
        for row in T.serial(first, last):
            work = row_work(Lengths, ExtraLengths, row)
            payload = row_bytes(Lengths, ExtraLengths, row)
            total_work = total_work + work
            total_bytes = total_bytes + payload
            if row < middle:
                left_work = left_work + work
                left_bytes = left_bytes + payload
        best = T.alloc_var("int64", init=T.max(left_work, total_work - left_work))
        byte_limit = T.max(left_bytes, total_bytes - left_bytes)
        left_work = 0
        left_bytes = 0
        for row in T.serial(first, last - 1):
            left_work = left_work + row_work(Lengths, ExtraLengths, row)
            left_bytes = left_bytes + row_bytes(Lengths, ExtraLengths, row)
            peak = T.max(left_work, total_work - left_work)
            if (peak < best) & (T.max(left_bytes, total_bytes - left_bytes) <= byte_limit):
                best = peak
                cut[0] = row + 1

    @T.prim_func
    def main(Q: T.Tensor((batch * H, D), "bfloat16"),
             Cache: T.Tensor((tokens, 528), "uint8"),
             Indices: T.Tensor((batch, width), "int32"),
             Lengths: T.Tensor((batch,), "int32"),
             Extra: T.Tensor((extra_tokens, extra_width), "uint8"),
             ExtraIndices: T.Tensor((batch, max(extra_topk, 1)), "int32"),
             ExtraLengths: T.Tensor((batch,), "int32"),
             Sink: T.Tensor((H,), "float32"),
             O: T.Tensor((output_rows, D), output_dtype),
             KVWorkspace: T.Tensor(workspace_shape, "bfloat16"),
             LSE: T.Tensor((output_rows,), "float32")):
        with T.MixedKernel(grid, sids=2) as (bx, sid):
            if stream:
                q_l1 = T.alloc_l1((q_slots, H, q_half), "bfloat16")
            else:
                q_l1 = T.alloc_l1((H, D), "bfloat16")
            kv_l1 = T.alloc_l1((block, D), "bfloat16")
            p_l1 = T.alloc_l1((H, block), "bfloat16")
            qa = T.alloc_l0a((H, FRAG_D), "bfloat16")
            kb = T.alloc_l0b((KV_TILE_ROWS, FRAG_D), "bfloat16")
            vb = T.Tensor((FRAG_D, KV_TILE_ROWS), "bfloat16", kb.data, scope="shared.l0b")
            scores = T.alloc_l0c((H, block), "float32")
            product_parts = T.alloc_l0c((1 if stream else D // FRAG_D, H, FRAG_D), "float32")
            if not stream:
                product = T.Tensor((H, D), "float32", product_parts.data, scope="shared.l0c")
                T.annotate_layout({product: make_ascend_l0c_layout(product)})
            nd = T.alloc_shared((kv_pitch, D), "bfloat16")
            raw_fp8 = T.alloc_shared((stripe, fp8_pitch), "uint8")
            scale_ub = T.alloc_shared((stripe * (fp4_scales if extra_topk and extra_width == 288 else fp8_scales),),
                                      "bfloat16")
            index_ub = T.alloc_shared((block,), "int32")
            valid = T.alloc_shared((block,), "int32")
            pair_offsets = T.alloc_shared((2, block // 2), "int32") if pair_dma or packed_pair_dma else None
            score_ub = T.alloc_shared((half, block), "float32")
            prob_ub = T.alloc_shared((pitch, block), "bfloat16")
            product_ub = T.alloc_shared((half, FRAG_D if stream else D), "float32")
            accum = T.alloc_shared((half, D), "float32")
            if stream:
                out_ub = T.alloc_shared((2, half, FRAG_D), output_dtype)
            if stream or packed_staged:
                final_m = T.alloc_shared((BLOCK_K,), "float32")
                final_ell = T.alloc_shared((BLOCK_K,), "float32")
            if short_stream and batch > cores:
                # An output DMA can still read this slot when the next query
                # starts updating its accumulator.
                out_ub = T.alloc_shared((half, D), output_dtype)
            elif not stream:
                out_ub = T.Tensor((half, D), output_dtype, accum.data, scope="shared.dyn")
            m = T.alloc_shared((BLOCK_K,), "float32")
            ell = T.alloc_shared((BLOCK_K,), "float32")
            alpha = T.alloc_shared((half,), "float32")
            stats = T.alloc_shared((BLOCK_K,), "float32")
            p_scale = T.alloc_shared((BLOCK_K,), "float32")
            sink = T.alloc_shared((BLOCK_K,), "float32")
            lse_ub = T.alloc_shared((half,), "float32")
            inv_ub = T.alloc_shared((half,), "float32")
            T.annotate_layout({prob_ub: make_ascend_compact_nz_layout(prob_ub)})
            T.annotate_buffer_versions({qa: (2, "counter"), kb: (2, "counter"), vb: (2, "counter")})
            if stream:
                T.annotate_manual_multi_buffer({q_l1: q_slots, out_ub: 2})
                # Prepare indices one tile ahead of the raw KV DMA. AIV can
                # fill one metadata slot while the Scalar pipe reads the other.
                T.annotate_buffer_versions({index_ub: (2, "iteration")})
                if pair_dma:
                    T.annotate_buffer_versions({pair_offsets: (2, "iteration")})
                # P, rescaling and the final stats stay live through delayed PV.
                # The FP32 product only needs two fragment-sized UB slots.
                T.annotate_buffer_versions({kv_l1: (3, "iteration"), valid: (3, "iteration"),
                                            score_ub: (2, "iteration"), prob_ub: (2, "iteration"),
                                            p_l1: (2, "iteration"), alpha: (2, "iteration"),
                                            final_m: (2, "iteration"), final_ell: (2, "iteration"),
                                            product_parts: (2, "iteration"), product_ub: (2, "iteration")})
            if packed_staged:
                # KV stays live from QK through PV, and the validity mask is
                # produced one stage earlier for gather and softmax.
                T.annotate_buffer_versions({
                    (pair_offsets if packed_pair_dma else index_ub): (2, "iteration"),
                    valid: (3, "iteration"), kv_l1: (3, "iteration"), score_ub: (2, "iteration"),
                    prob_ub: (2, "iteration"), p_l1: (2, "iteration"), alpha: (2, "iteration"),
                    final_m: (2, "iteration"), final_ell: (2, "iteration"),
                })
            elif short_stream or packed_lookahead:
                T.annotate_buffer_versions({kv_l1: (2, "iteration"), valid: (2, "iteration")})
            if direct_kv:
                T.annotate_layout({nd: make_ascend_compact_nz_layout(nd),
                                   kv_l1: make_ascend_major_k_layout(kv_l1)})
                T.annotate_buffer_versions({raw_fp8: (raw_lead + 1, "iteration"), nd: (2, "iteration")})
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                sink_init = T.float32(0) if kv_split == 1 else -T.infinity("float32")
                S.vsts(sink[0], S.vdup(sink_init, "float32", full), full)
                T.fill(raw_fp8, 0)
            if kv_split == 1:
                T.copy(Sink[sid * half:(sid + 1) * half], sink[:half])
            per = tasks // grid
            remainder = tasks % grid
            if balance_rows and grid < batch < 2 * grid:
                row_range = T.alloc_local((2,), "int32")
                row_range[0] = bx * per + T.min(bx, remainder)
                row_range[1] = per + T.if_then_else(bx < remainder, 1, 0)
                balance_single_and_double_rows(Lengths, ExtraLengths, bx, row_range)
                start = row_range[0]
                count = row_range[1]
            elif balance_rows:
                pair = bx // 2 * 2
                after_pair = T.min(pair + 2, grid)
                first = pair * per + T.min(pair, remainder)
                middle = (pair + 1) * per + T.min(pair + 1, remainder)
                last = after_pair * per + T.min(after_pair, remainder)
                cut = T.alloc_local((1,), "int32")
                cut[0] = middle
                if (pair + 1 < grid) & (last - first > 2):
                    balance_pair(Lengths, ExtraLengths, first, middle, last, cut)
                start = T.if_then_else(bx % 2 == 0, first, cut[0])
                end = T.if_then_else(bx % 2 == 0, cut[0], last)
                count = end - start
            else:
                start = bx * per + T.min(bx, remainder)
                count = per + T.if_then_else(bx < remainder, 1, 0)
            if stream:
                # Three Q half-slots fit beside three KV tiles and two P tiles.
                # One half of the next query can arrive before the old query
                # releases its first half. Synchronize those lifetimes on MTE1.
                with T.Stage(0):
                    for slot in T.serial(q_slots):
                        T.ascend_set_flag("MTE1_MTE2", q_flag + slot)
                for stream_tile in T.Pipelined(count * steps_per_query, num_stages=4,
                                              annotations={"enable_offset": True}):
                    T.assume_no_conflict(q_l1)
                    query = stream_tile // steps_per_query
                    row = start + query
                    local_tile = stream_tile % steps_per_query
                    q_slot = (2 * query) % q_slots
                    if stream_packed:
                        main_halves = (width + BLOCK_K - 1) // BLOCK_K
                        n = block
                    else:
                        n = block if full_tiles else T.min(width - local_tile * block, block)
                    with T.Stage(0):
                        if not stream_packed:
                            prepare_indices(Indices, row, local_tile, n, block, index_ub, valid,
                                            width, tokens, pair_offsets)
                    with T.Stage(1):
                        if local_tile == 0:
                            prefetch_query(Q, row, q_l1, q_slot)
                        if stream_packed:
                            # Odd tails pad the second half with masked rows.
                            gather_packed(Cache, Extra, Indices, ExtraIndices, row, local_tile,
                                          width, extra_topk, main_halves, block, sid, index_ub,
                                          valid, raw_fp8, scale_ub, nd, kv_l1)
                        else:
                            gather_fp8_to_l1(Cache, n, sid, index_ub, valid, raw_fp8, scale_ub, nd,
                                             kv_l1, pair_offsets)
                    with T.Stage(2):
                        qk(q_l1, kv_l1, qa, kb, scores, score_ub, block, q_slot, local_tile,
                           steps_per_query)
                        if local_tile == 0:
                            with T.SimdVF():
                                full = S.pset(32, "PAT_ALL")
                                S.vsts(m[0], S.vdup(-T.infinity("float32"), "float32", full), full)
                                S.vsts(ell[0], S.vdup(T.float32(0), "float32", full), full)
                        softmax(score_ub, valid, prob_ub, p_l1, m, ell, alpha, sink, stats, p_scale, block,
                                (final_m, final_ell))
                    with T.Stage(3):
                        stream_pv(local_tile, kv_l1, p_l1, qa, vb, product_parts, product_ub, accum,
                                  alpha, block)
                        if local_tile + 1 == steps_per_query:
                            finish(row, O, LSE, accum, out_ub, final_m, final_ell, sink, lse_ub, inv_ub, sid)
                with T.Stage(0):
                    for slot in T.serial(q_slots):
                        T.ascend_wait_flag("MTE1_MTE2", q_flag + slot)
            elif short_stream:
                # Keep one query's state in the consumer while the producer
                # fills the other KV slot, including across query boundaries.
                for stream_tile in T.Pipelined(count * steps_per_query, num_stages=2,
                                              annotations={"enable_offset": True}):
                    row = start + stream_tile // steps_per_query
                    local_tile = stream_tile % steps_per_query
                    with T.Stage(0):
                        prepare_indices(Indices, row, local_tile, block, block, index_ub, valid, width, tokens)
                        gather_fp8_to_l1(Cache, block, sid, index_ub, valid, raw_fp8, scale_ub, nd, kv_l1, None)
                    with T.Stage(1):
                        if local_tile == 0:
                            T.copy(Q[row * H:(row + 1) * H, :], q_l1)
                            with T.SimdVF():
                                full = S.pset(32, "PAT_ALL")
                                S.vsts(m[0], S.vdup(-T.infinity("float32"), "float32", full), full)
                                S.vsts(ell[0], S.vdup(T.float32(0), "float32", full), full)
                        qk(q_l1, kv_l1, qa, kb, scores, score_ub, block)
                        softmax(score_ub, valid, prob_ub, p_l1, m, ell, alpha, sink, stats, p_scale, block)
                        pv(kv_l1, p_l1, qa, vb, product_parts, block)
                        T.dual_copy(product, product_ub)
                        update_output(local_tile, product_ub, accum, alpha)
                        if local_tile + 1 == steps_per_query:
                            finish(row, O, LSE, accum, out_ub, m, ell, sink, lse_ub, inv_ub)
            elif packed_staged:
                for row in T.serial(start, start + count):
                    T.copy(Q[row * H:(row + 1) * H, :], q_l1)
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        S.vsts(m[0], S.vdup(-T.infinity("float32"), "float32", full), full)
                        S.vsts(ell[0], S.vdup(T.float32(0), "float32", full), full)
                    main_length = Lengths[row]
                    extra_length = ExtraLengths[row]
                    main_halves = T.ceildiv(main_length, BLOCK_K)
                    half_count = main_halves + T.ceildiv(extra_length, BLOCK_K)
                    iterations = T.max(T.ceildiv(half_count, 2), 1)
                    for tile in T.Pipelined(
                        iterations, num_stages=4,
                        annotations={
                            "enable_offset": True,
                            "multi_buffer_eligible": [pair_offsets if packed_pair_dma else index_ub, valid, kv_l1],
                        },
                    ):
                        rows = T.if_then_else(tile * 2 + 1 < half_count, block, BLOCK_K)
                        with T.Stage(0):
                            prepare_packed_indices(Indices, ExtraIndices, row, tile, main_length, extra_length,
                                                   main_halves, rows, index_ub, valid, pair_offsets)
                        with T.Stage(1):
                            gather_packed(Cache, Extra, Indices, ExtraIndices, row, tile, main_length, extra_length,
                                          main_halves, rows, sid, index_ub, valid, raw_fp8, scale_ub, nd, kv_l1,
                                          prepare=False, pair_offsets=pair_offsets)
                            if rows == BLOCK_K:
                                qk(q_l1, kv_l1, qa, kb, scores, score_ub, BLOCK_K)
                            else:
                                qk(q_l1, kv_l1, qa, kb, scores, score_ub, block)
                        with T.Stage(2):
                            if rows == BLOCK_K:
                                softmax(score_ub, valid, prob_ub, p_l1, m, ell, alpha, sink, stats, p_scale,
                                        BLOCK_K, (final_m, final_ell))
                            else:
                                softmax(score_ub, valid, prob_ub, p_l1, m, ell, alpha, sink, stats, p_scale,
                                        block, (final_m, final_ell))
                        with T.Stage(3):
                            if rows == BLOCK_K:
                                pv(kv_l1, p_l1, qa, vb, product_parts, BLOCK_K)
                            else:
                                pv(kv_l1, p_l1, qa, vb, product_parts, block)
                            T.dual_copy(product, product_ub)
                            update_output(tile, product_ub, accum, alpha)
                            if tile + 1 == iterations:
                                finish(row, O, LSE, accum, out_ub, final_m, final_ell, sink, lse_ub, inv_ub)
            else:
                for task in T.serial(start, start + count):
                    row = task // kv_split if kv_split > 1 else task
                    T.copy(Q[row * H:(row + 1) * H, :], q_l1)
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        S.vsts(m[0], S.vdup(-T.infinity("float32"), "float32", full), full)
                        S.vsts(ell[0], S.vdup(T.float32(0), "float32", full), full)
                    main_length = width if full_length else Lengths[row]
                    if extra_topk:
                        extra_length = extra_topk if full_length else ExtraLengths[row]
                    if packed_tile:
                        main_halves = T.ceildiv(main_length, BLOCK_K)
                        half_count = main_halves + T.ceildiv(extra_length, BLOCK_K)
                        iterations = T.max(T.ceildiv(half_count, 2), 1)
                    elif extra_topk:
                        extra_tiles = T.ceildiv(extra_length, block)
                    else:
                        extra_tiles = 0
                    if not packed_tile:
                        # Keep one masked tile when both prefixes are empty so
                        # the first PV update initializes the accumulator.
                        main_tiles = T.max(T.ceildiv(main_length, block), T.cast(extra_tiles == 0, "int32"))
                        tile_count = main_tiles + extra_tiles
                        if kv_split > 1:
                            piece = task % kv_split
                            begin = piece * tile_count // kv_split
                            end = (piece + 1) * tile_count // kv_split
                            piece_tiles = end - begin
                            # Empty slices emit a masked partial using the same DMA schedule.
                            iterations = piece_tiles if full_length else T.max(piece_tiles, 1)
                        else:
                            iterations = tile_count
                    for tile in (T.Pipelined(iterations, num_stages=2, annotations={"enable_offset": True})
                                 if packed_lookahead else T.serial(iterations)):
                        if packed_tile:
                            # A lone half tile fits in the first 64 L1 rows. An
                            # empty row also needs one masked half to initialize PV.
                            n = T.if_then_else(tile * 2 + 1 < half_count, block, BLOCK_K)
                            with T.Stage(0) if packed_lookahead else nullcontext():
                                gather_packed(Cache, Extra, Indices, ExtraIndices, row, tile, main_length, extra_length,
                                              main_halves, n, sid, index_ub, valid, raw_fp8, scale_ub, nd, kv_l1)
                        else:
                            if kv_split > 1:
                                global_tile = begin + tile if full_length else T.min(begin + tile, tile_count - 1)
                            else:
                                global_tile = tile
                            from_main = global_tile < main_tiles
                            extra_tile = global_tile - main_tiles
                            if full_tiles:
                                n = block
                            else:
                                n = main_length - global_tile * block
                                if extra_topk:
                                    n = T.if_then_else(from_main, n, extra_length - extra_tile * block)
                                n = T.min(n, block)
                                if kv_split > 1 and not full_length:
                                    n = T.if_then_else(piece_tiles == 0, 0, n)
                            rows = block if kv_pipeline or block == BLOCK_K else T.if_then_else(n <= BLOCK_K, BLOCK_K, block)
                            if kv_pipeline:
                                # Fixed L1 offsets also cover a masked final tile.
                                prepare_indices(Indices, row, global_tile, n, block,
                                                index_ub, valid, width, tokens, pair_offsets)
                                gather_fp8_to_l1(Cache, n, sid, index_ub, valid, raw_fp8, scale_ub, nd,
                                                kv_l1, pair_offsets)
                            elif not extra_topk or from_main:
                                gather(Cache, Indices, row, global_tile, n, rows, bx, sid, index_ub, valid,
                                       nd, raw_fp8, scale_ub, KVWorkspace, width, tokens, fp8_bytes)
                            else:
                                gather(Extra, ExtraIndices, row, extra_tile, n, rows, bx, sid, index_ub, valid,
                                       nd, raw_fp8, scale_ub, KVWorkspace, extra_topk, extra_tokens, extra_width)
                        # One KV slot feeds Cube while the other is gathered by AIV.
                        # Keep the online softmax and PV recurrence in one stage.
                        with T.Stage(1) if packed_lookahead else nullcontext():
                            if block == BLOCK_K or (not kv_pipeline and n <= BLOCK_K):
                                if not direct_kv:
                                    T.copy(KVWorkspace[bx, 0, :BLOCK_K // 2, :], kv_l1[:BLOCK_K // 2, :])
                                    T.copy(KVWorkspace[bx, 1, :BLOCK_K // 2, :], kv_l1[BLOCK_K // 2:BLOCK_K, :])
                                qk(q_l1, kv_l1, qa, kb, scores, score_ub, BLOCK_K)
                                softmax(score_ub, valid, prob_ub, p_l1, m, ell, alpha, sink, stats, p_scale, BLOCK_K)
                                pv(kv_l1, p_l1, qa, vb, product_parts, BLOCK_K)
                            else:
                                if not direct_kv:
                                    T.copy(KVWorkspace[bx, 0, :, :], kv_l1[:block // 2, :])
                                    T.copy(KVWorkspace[bx, 1, :, :], kv_l1[block // 2:, :])
                                qk(q_l1, kv_l1, qa, kb, scores, score_ub, block)
                                softmax(score_ub, valid, prob_ub, p_l1, m, ell, alpha, sink, stats, p_scale, block)
                                pv(kv_l1, p_l1, qa, vb, product_parts, block)
                            T.dual_copy(product, product_ub)
                            update_output(tile, product_ub, accum, alpha)
                            # Keep the output in the same task loop as its PV producer.
                            # The scheduler can otherwise move it across the nested loop.
                            if tile + 1 == iterations:
                                finish(task, O, LSE, accum, out_ub, m, ell, sink, lse_ub, inv_ub)

    return main


def build_decode_merge_kernel(batch, split, cores):
    """Merge groups of heads using one strided DMA across all pieces."""
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    rows = batch * H
    # Narrower groups share the bandwidth across more AIVs at high split
    # counts and keep partial-output staging within 128 KiB.
    max_heads = 8 if split <= 4 else 4
    heads = min(max_heads, 1 << max(0, (64 // split).bit_length() - 1))
    groups_per_query = H // heads
    groups = batch * groups_per_query
    grid = min(groups, 2 * cores)

    @T.macro
    def reduce_lse(logs, sink_ub, weights, lse_ub):
        with T.SimdVF():
            mask = S.pset(32, f"PAT_VL{heads}")
            zero = S.vdup(T.float32(0), "float32", mask)
            one = S.vdup(T.float32(1), "float32", mask)
            neg_inf = S.vdup(-T.infinity("float32"), "float32", mask)
            inf = S.vdup(T.infinity("float32"), "float32", mask)
            anchor = T.alloc_var("float32x64", init=neg_inf)
            for piece in T.serial(split):
                anchor = S.vmax(anchor, S.vld(logs[piece, 0]), mask)
            nonempty = S.vcmps(anchor, -T.infinity("float32"), mask, "ne")
            safe_anchor = S.vsel(anchor, zero, nonempty)
            total = T.alloc_var("float32x64", init=zero)
            for piece in T.serial(split):
                weight = S.vexp(S.vsub(S.vld(logs[piece, 0]), safe_anchor, mask), mask)
                total = S.vadd(total, weight, mask)
            safe_total = S.vsel(total, one, nonempty)
            kv_lse = S.vadd(safe_anchor, S.vln(safe_total, mask), mask)
            sink = S.vld(sink_ub[0])
            infinite_sink = S.vcmps(sink, T.infinity("float32"), mask, "eq")
            origin = S.vmax(safe_anchor, sink, mask)
            safe_origin = S.vsel(zero, origin, infinite_sink)
            active = S.pand(nonempty, S.vcmps(sink, T.infinity("float32"), mask, "ne"), mask)
            kv_scale = S.vsel(S.vexp(S.vsub(safe_anchor, safe_origin, mask), mask), zero, active)
            sink_weight = S.vsel(one, S.vexp(S.vsub(sink, safe_origin, mask), mask), infinite_sink)
            denominator = S.vadd(S.vmul(total, kv_scale, mask), sink_weight, mask)
            safe_denominator = S.vsel(denominator, one,
                                      S.vcmps(denominator, T.float32(0), mask, "gt"))
            factor = S.vdiv(kv_scale, safe_denominator, mask, precision="exact")
            for piece in T.serial(split):
                weight = S.vexp(S.vsub(S.vld(logs[piece, 0]), safe_anchor, mask), mask)
                S.vsts(weights[piece, 0], S.vmul(weight, factor, mask), mask, extent=heads)
            S.vsts(lse_ub[0], S.vsel(kv_lse, inf, nonempty), mask, extent=heads)

    @T.macro
    def reduce_output(values, weights, out):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0), "float32", full)
            for head in T.serial(heads):
                for col in T.serial(D // BLOCK_K):
                    result = T.alloc_var("float32x64", init=zero)
                    for piece in T.serial(split):
                        value = S.vld(values[piece, head * D + col * BLOCK_K])
                        factor = S.vld(weights[piece, head], "BRC_B32")
                        S.vmula(result, value, factor, full)
                    packed = S.vcvt(result, "bfloat16", full, sat=False, part=0)
                    S.vsts(out[head * D + col * BLOCK_K], packed, full, "PK_B32")

    @T.prim_func
    def decode_merge(Partial: T.Tensor((rows * split, D), "float32"),
             PartialLSE: T.Tensor((rows * split,), "float32"),
             Sink: T.Tensor((H,), "float32"),
             O: T.Tensor((rows, D), "bfloat16"),
             LSE: T.Tensor((rows,), "float32")):
        with T.Kernel(grid) as bx:
            values = T.alloc_shared((split, heads * D), "float32")
            logs = T.alloc_shared((split, BLOCK_K), "float32")
            weights = T.alloc_shared((split, BLOCK_K), "float32")
            sink_ub = T.alloc_shared((BLOCK_K,), "float32")
            out = T.alloc_shared((heads * D,), "bfloat16")
            lse_ub = T.alloc_shared((BLOCK_K,), "float32")
            partial = T.Tensor((batch, split, H * D), "float32", Partial.data)
            partial_lse = T.Tensor((batch, split, H), "float32", PartialLSE.data)
            output = T.Tensor((rows * D,), "bfloat16", O.data)
            for step in T.serial(T.ceildiv(groups, grid)):
                group = step * grid + bx
                if group < groups:
                    query = group // groups_per_query
                    head = group % groups_per_query * heads
                    row = group * heads
                    T.copy(partial[query, :, head * D:(head + heads) * D], values)
                    T.copy(partial_lse[query, :, head:head + heads], logs[:, :heads])
                    T.copy(Sink[head:head + heads], sink_ub[:heads])
                    reduce_lse(logs, sink_ub, weights, lse_ub)
                    reduce_output(values, weights, out)
                    T.copy(out, output[row * D:(row + heads) * D])
                    T.copy(lse_ub[:heads], LSE[row:row + heads])
    return decode_merge


@lru_cache(maxsize=32)
def _compile_decode(batch, nc, p, width, ne, pe, ce, te, scale, cores, *, kv_split=1,
                    full_length=False, kv_pipeline=False, balance_rows=False, packed_lookahead=False):
    import tilelang
    from tvm.target import Target
    program = build_decode_kernel(batch, nc, p, width, ne=ne, extra_page_size=pe,
                                   extra_width=ce, extra_topk=te, softmax_scale=scale, cores=cores,
                                   kv_split=kv_split, full_length=full_length, kv_pipeline=kv_pipeline,
                                   balance_rows=balance_rows, packed_lookahead=packed_lookahead)
    with Target("ascend") as target:
        attention = tilelang.compile(program, target=target, execution_backend="tvm_ffi", out_idx=[8, 9, 10])
        if kv_split == 1:
            return attention
        merge = tilelang.compile(build_decode_merge_kernel(batch, kv_split, cores), target=target,
                                 execution_backend="tvm_ffi", out_idx=[3, 4])

    def run_split(*inputs):
        partial, workspace, partial_lse = attention(*inputs)
        output, lse = merge(partial, partial_lse, inputs[-1])
        return output, workspace, lse

    return run_split


def _check_tensor(name, value, shape, dtype, device):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != tuple(shape) or value.dtype != dtype:
        raise ValueError(f"{name}: expected shape={shape}, dtype={dtype}; got {value.shape}, {value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")


def _check_scale(scale):
    scale = float(scale)
    if not math.isfinite(scale) or not math.isclose(scale, SM_SCALE, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("This model specification requires scale=1/sqrt(512)")
    return scale


def _validate_length_values(lengths, width, name="topk_length"):
    # Check the range on the host without dispatching NPU math kernels.
    host_lengths = lengths.detach().cpu()
    if bool(((host_lengths < 0) | (host_lengths > width)).any()):
        raise ValueError(f"{name} must be in [0, {width}]")
    return host_lengths


def _check_indices_lengths(indices, lengths, rows, width, device, validate_values):
    if width <= 0 or width % BLOCK_K:
        raise ValueError("Index width must be a positive multiple of 64")
    _check_tensor("indices", indices, (rows, 1, width), torch.int32, device)
    if lengths is not None:
        _check_tensor("topk_length", lengths, (rows,), torch.int32, device)
        if validate_values:
            return _validate_length_values(lengths, width)
    return None


def _require_npu(q):
    if q.device.type != "npu":
        raise ValueError("The TileLang implementation requires NPU tensors; use reference functions on CPU")


def _check_cache(name, cache, device, main=False):
    min_page_size = 2 if main else 1
    if (cache.ndim != 4 or cache.shape[0] <= 0 or cache.shape[1] < min_page_size
            or cache.shape[2] != 1):
        shape = "[Nc,P,1,C], Nc>0, P>1" if main else "[Ne,Pe,1,C], Ne>0, Pe>=1"
        raise ValueError(f"{name} must have shape {shape}")
    allowed_widths = (528,) if main else (528, 288)
    if cache.shape[-1] not in allowed_widths:
        raise ValueError(f"{name}: bytes/token must be in {allowed_widths}")
    if cache.dtype not in (torch.float8_e4m3fn, torch.int8, torch.uint8):
        raise ValueError(f"{name} must be an FP8/int8/uint8 byte container")
    if cache.device != device:
        raise ValueError(f"{name} must be on {device}")


_CONSTANT_CACHE: dict[tuple, torch.Tensor] = {}
# Compact views of the public tensors. A cached view keeps its base storage
# alive, so a data_ptr in the key can never be recycled by another tensor.
_VIEW_CACHE: dict[tuple, torch.Tensor] = {}
# get_device_properties() is a per-call device query; the core count is fixed.
_CORES_CACHE: dict[int, int] = {}


def _core_count(q=None):
    """Count the Cube cores on the tensor's NPU or the current NPU."""
    if not hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401; registers torch.npu lazily.
        except ImportError as exc:
            raise RuntimeError("Automatic core detection requires torch_npu; pass cores explicitly for offline source generation") from exc
    device = q.device if q is not None else torch.npu.current_device()
    cores = int(torch.npu.get_device_properties(device).cube_core_num)
    if cores <= 0:
        raise RuntimeError(f"Expected a positive Cube core count for {device}, got {cores}")
    return cores


def _cached_cores(q):
    index = q.device.index
    cores = _CORES_CACHE.get(index)
    if cores is None:
        cores = _core_count(q)
        _CORES_CACHE[index] = cores
    return cores


_VIEW_CACHE_BUDGET = 8 << 20  # only memoise compact views; a held view pins
# its storage, so large caches must be re-viewed instead of retained.


def _cached_view(key, build):
    view = _VIEW_CACHE.get(key)
    if view is None:
        view = build()
        if view.untyped_storage().nbytes() <= _VIEW_CACHE_BUDGET:
            if len(_VIEW_CACHE) > 32:
                _VIEW_CACHE.clear()
            _VIEW_CACHE[key] = view
    return view


def _constant_on_device(shape, value, dtype, device):
    # Build metadata and sink defaults on CPU, then transfer their bytes.
    return torch.full(shape, value, dtype=dtype, device="cpu").to(device)


def _cached_constant(shape, value, dtype, device):
    """Read-only default tensor reused across calls.

    The kernel only reads lengths/extra_lengths/sink, so a per-(shape, value,
    dtype, device) cache removes three CPU->NPU copies (~38 us on B=1) from
    every call.
    """
    key = (tuple(shape), float(value), dtype, str(device))
    tensor = _CONSTANT_CACHE.get(key)
    if tensor is None:
        tensor = _constant_on_device(shape, value, dtype, device)
        _CONSTANT_CACHE[key] = tensor
    return tensor


def flash_mla_with_kvcache(
    q, k_cache, block_table, cache_seqlens, head_dim_v,
    tile_scheduler_metadata, num_splits=None, softmax_scale=None,
    causal=False, is_fp8_kvcache=True, indices=None, attn_sink=None,
    extra_k_cache=None, extra_indices_in_kvcache=None,
    topk_length=None, extra_topk_length=None, enable_batch_invariant=False,
    *, validate_values=True,
):
    """[B,1,64,512] BF16 -> O[B,1,64,512], lse[B,64,1].

    Accepts metadata from this module's get_mla_metadata() or the original
    FlashMLA get_mla_metadata(). In the latter case its contents are preserved;
    the Ascend calculation does not consume CUDA scheduler tensors. Each query
    runs on one block. Decode selects pipelining from the input geometry,
    length mode and available cores. Long, variable-length KV can rebalance
    query rows between cores. With an extra cache, each tile packs up to two
    64-row pieces and gathers them directly to L1.
    With validate_values=True, check lengths and reject non--1 out-of-range
    physical indices in the consumed prefix (requires host synchronization).
    With False, those ranges are a caller precondition; loads still have bounds
    guards. block_table/cache_seqlens are ignored, as in the source interface.
    """
    if q.ndim != 4 or tuple(q.shape[1:]) != (1, H, D) or q.shape[0] <= 0:
        raise ValueError("q must have shape [B,1,64,512], B>0")
    _check_tensor("q", q, q.shape, torch.bfloat16, q.device)
    if head_dim_v != D or num_splits is not None or causal or not is_fp8_kvcache:
        raise ValueError("Require head_dim_v=512, num_splits=None, causal=False, is_fp8_kvcache=True")
    if tile_scheduler_metadata is None:
        raise TypeError("Pass the metadata object returned by get_mla_metadata()[0]")
    if not isinstance(tile_scheduler_metadata, FlashMLASchedMeta):
        cls = type(tile_scheduler_metadata)
        if cls.__name__ != "FlashMLASchedMeta" or cls.__module__ != "flash_mla.flash_mla_interface":
            raise TypeError("tile_scheduler_metadata must be local or FlashMLA FlashMLASchedMeta")
    _check_cache("k_cache", k_cache, q.device, main=True)
    if indices is None or indices.ndim != 3:
        raise ValueError("indices with shape [B,1,T] is required")
    b, nc, p, width = q.shape[0], k_cache.shape[0], k_cache.shape[1], indices.shape[-1]
    if validate_values:
        _check_indices_lengths(indices, topk_length, b, width, q.device, True)
    elif width <= 0 or width % BLOCK_K:
        # Same index-width rule, without the per-tensor contract checks: with
        # validate_values=False the caller already promised the ranges.
        raise ValueError("Index width must be a positive multiple of 64")
    scale = _check_scale(SM_SCALE if softmax_scale is None else softmax_scale)
    if attn_sink is not None:
        _check_tensor("attn_sink", attn_sink, (H,), torch.float32, q.device)
    static_lengths = topk_length is None and extra_topk_length is None
    ne, pe, ce, te = 0, 2, 528, 0
    if extra_k_cache is not None:
        _check_cache("extra_k_cache", extra_k_cache, q.device)
        if extra_indices_in_kvcache is None or extra_indices_in_kvcache.ndim != 3:
            raise ValueError("extra_indices_in_kvcache is required with extra_k_cache")
        ne, pe, _, ce = extra_k_cache.shape
        te = extra_indices_in_kvcache.shape[-1]
        if te <= 0:
            raise ValueError("Te must be positive")
        _check_tensor("extra_indices_in_kvcache", extra_indices_in_kvcache, (b, 1, te), torch.int32, q.device)
        if extra_topk_length is not None:
            _check_tensor("extra_topk_length", extra_topk_length, (b,), torch.int32, q.device)
            if validate_values:
                _validate_length_values(extra_topk_length, te, "extra_topk_length")
    elif extra_indices_in_kvcache is not None or extra_topk_length is not None:
        raise ValueError("Extra indices/lengths require extra_k_cache")

    if validate_values:
        for idx, lens, tokens in ((indices, topk_length, nc * p),
                                  (extra_indices_in_kvcache, extra_topk_length, ne * pe)):
            if idx is not None:
                host_idx = idx.detach().cpu()
                invalid = (host_idx != -1) & ((host_idx < 0) | (host_idx >= tokens))
                if lens is not None:
                    host_lens = lens.detach().cpu()
                    invalid &= torch.arange(idx.shape[-1], device="cpu")[None, None, :] < host_lens[:, None, None]
                if bool(invalid.any()):
                    raise ValueError("Decode physical index must be -1 or in [0, N*P) within the valid prefix")
    _require_npu(q)
    lengths = (_cached_constant((b,), width, torch.int32, q.device)
               if topk_length is None else topk_length.contiguous())
    extra_lengths = (_cached_constant((b,), te, torch.int32, q.device)
                     if extra_topk_length is None else extra_topk_length.contiguous())
    sink = (_cached_constant((H,), -math.inf, torch.float32, q.device)
            if attn_sink is None else attn_sink.contiguous())
    cache = _cached_view(
        ("cache", k_cache.data_ptr(), k_cache.shape, k_cache.stride(), k_cache.dtype),
        lambda: k_cache.contiguous().view(torch.uint8).view(nc * p, 528))
    idx = _cached_view(
        ("idx", indices.data_ptr(), indices.shape, indices.stride(), indices.dtype),
        lambda: indices.contiguous().view(b, width))
    extra = (cache[:1] if extra_k_cache is None else
             _cached_view(
                 ("extra", extra_k_cache.data_ptr(), extra_k_cache.shape,
                  extra_k_cache.stride(), extra_k_cache.dtype),
                 lambda: extra_k_cache.contiguous().view(torch.uint8).view(ne * pe, ce)))
    extra_idx = (extra_lengths.view(b, 1) if extra_indices_in_kvcache is None else
                 extra_indices_in_kvcache.contiguous().view(b, te))
    config = (b, 1, H, p, 1, False, True, width, pe if ne else None, te if ne else None,
              bool(enable_batch_invariant))
    if isinstance(tile_scheduler_metadata, FlashMLASchedMeta):
        if tile_scheduler_metadata.have_initialized and tile_scheduler_metadata.config != config:
            raise ValueError("Metadata shape/config changed; create a fresh get_mla_metadata()[0]")
    device = q.device
    if torch.npu.current_device() != device.index:
        torch.npu.set_device(device)
    cores = _cached_cores(q)
    pipeline = _auto_kv_pipeline(b, width, te, cores, static_lengths,
                                 enable_batch_invariant=enable_batch_invariant)
    full_length = pipeline and static_lengths
    # Rebalancing needs multiple queries on at least one core and enough
    # KV work per query to amortize the length reads.
    kv_rows = width + (te + BLOCK_K - 1) // BLOCK_K * BLOCK_K
    balance_rows = (not static_lengths and not enable_batch_invariant and 1 < cores < b
                    and kv_rows > KV_TILE_ROWS)
    packed_lookahead = bool(te and not static_lengths and not enable_batch_invariant and kv_rows > KV_TILE_ROWS)
    kernel = _compile_decode(b, nc, p, width, ne, pe, ce, te, scale, cores, kv_split=1,
                             full_length=full_length, kv_pipeline=pipeline, balance_rows=balance_rows,
                             packed_lookahead=packed_lookahead)
    flat_q = _cached_view(("q", q.data_ptr(), q.shape, q.stride(), q.dtype),
                          lambda: q.contiguous().view(b * H, D))
    o, _, lse = kernel(flat_q, cache, idx, lengths, extra, extra_idx, extra_lengths, sink)
    if isinstance(tile_scheduler_metadata, FlashMLASchedMeta):
        tile_scheduler_metadata.have_initialized = True
        tile_scheduler_metadata.config = config
    return o.view(b, 1, H, D), lse.view(b, H, 1)
