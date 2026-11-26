"""Sparse MLA prefill with shared, double-buffered Cube L0 operands.

Variant of mla.py: QK and PV share two L0B slots. Each slot holds one K=128
panel; alternating slots lets MTE1 load the next panel while Cube consumes
the current panel. The two L0B slots occupy 64 KiB. The direct KV path
publishes full tiles from UB to L1 and keeps P in L0A across PV fragments.
Public entry points and benchmark options match mla.py.
The same planner selects streaming and KV partitioning for prepared benchmarks
and ordinary calls. KV partitions address the original tensors directly and
emit FP32 KV-only partials; one vector kernel merges them with the sink once.

The two public entry points preserve the coworker handoff signatures. Every
supported shape uses the kernels defined in this file. Sparse KV tiles contain
128 entries (64 for TopK=64); partial final tiles are padded and masked.
Requirements: PyTorch, torch_npu for execution, and TileLang Ascend with the
Ascend 950 compiler. Compilation and execution are lazy.
"""

import math
from bisect import bisect_left, bisect_right
from functools import lru_cache
from itertools import accumulate

import torch

H, D, BLOCK_K, FRAG_D = 64, 512, 64, 128
SM_SCALE = 1.0 / math.sqrt(D)
TASK_STRIDE = 6  # row, tile, valid rows, first, last, Q half-slot
KV_TILE_ROWS = 128
KV_WORKSPACE_SLOTS = 3
# Keep the benchmark on the public default's masked path. Full-valid is an
# explicit caller contract, not a property inferred only by the test harness.
BENCHMARK_AUTO_FULL_VALID_KV = False

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


def _check_fixed_offset(value):
    """None keeps online softmax; a finite FP32 value selects the bounded fast path."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or abs(value) > torch.finfo(torch.float32).max:
        raise ValueError("fixed_offset must be a finite FP32 value or None")
    return value


def _validate_length_values(lengths, width, name="topk_length"):
    # Range checks are host work, not additional torch_npu math kernels.
    host_lengths = lengths.detach().cpu()
    if bool(((host_lengths < 0) | (host_lengths > width)).any()):
        raise ValueError(f"{name} must be in [0, {width}]")
    return host_lengths


def _constant_on_device(shape, value, dtype, device):
    # Build metadata/sink defaults on CPU, then transfer their bytes.
    return torch.full(shape, value, dtype=dtype, device="cpu").to(device)


def _max_tasks_per_core(sq, width, grid):
    tiles = (width + 127) // 128
    return (sq * tiles + grid - 1) // grid + tiles


def _meta_cols(sq, width, grid):
    # Header: query start, query count, task count. Then TASK_STRIDE fields
    # per S2 task so the kernel never carries a query/tile cursor.
    return 3 + TASK_STRIDE * _max_tasks_per_core(sq, width, grid)


def _packed_kv_workspace(width, full_length, pipeline_queries, full_valid_kv):
    return full_length and pipeline_queries and (width % KV_TILE_ROWS == 0 or full_valid_kv)


def _use_direct_kv_l1(width, full_length, pipeline_queries, full_valid_kv, pair_dma, kv_split):
    return (full_length and pipeline_queries and width % KV_TILE_ROWS == 0
            and not full_valid_kv and not pair_dma and kv_split == 1)


def _balance_streaming_ranges(lengths, tile_counts, tile_prefix, ranges):
    """Minimize peak tile count, then peak KV import rows, for contiguous queries."""
    grid, rows = len(ranges), len(lengths)
    if grid == 1 or grid == rows:
        return ranges

    # Match load_kv_tile's import geometry in units of 16 rows, including
    # the masked tile for length 0.
    kv_groups = []
    for length, tiles in zip(lengths, tile_counts):
        full_rows = (tiles - 1) * KV_TILE_ROWS
        tail = length - full_rows
        tail_import = 16 if tail <= 1 else BLOCK_K if tail <= BLOCK_K else KV_TILE_ROWS
        kv_groups.append((full_rows + tail_import) // 16)
    kv_prefix = list(accumulate(kv_groups, initial=0))
    peak_tiles = max(tile_prefix[start + count] - tile_prefix[start] for start, count in ranges)
    peak_kv = max(kv_prefix[start + count] - kv_prefix[start] for start, count in ranges)
    # Bound the Query startup and output work using the original partition.
    peak_queries = max(count for _, count in ranges)

    def suffix_start(end, tile_cap, kv_cap):
        return max(end - peak_queries,
                   bisect_left(tile_prefix, tile_prefix[end] - tile_cap, hi=end),
                   bisect_left(kv_prefix, kv_prefix[end] - kv_cap, hi=end))

    def fits(tile_cap, kv_cap):
        # Each core takes the longest fitting suffix of the remaining queries.
        # Removing this suffix leaves the fewest queries for the other cores.
        end = rows
        for _ in ranges:
            end = suffix_start(end, tile_cap, kv_cap)
            if end == 0:
                return True
        return False

    # The original partition gives feasible upper bounds for both searches.
    tile_floor = max(max(tile_counts), (tile_prefix[-1] + grid - 1) // grid)
    tile_cap = tile_floor + bisect_left(
        range(tile_floor, peak_tiles), True, key=lambda cap: fits(cap, peak_kv))
    kv_floor = max(max(kv_groups), (kv_prefix[-1] + grid - 1) // grid)
    kv_cap = kv_floor + bisect_left(
        range(kv_floor, peak_kv), True, key=lambda cap: fits(tile_cap, cap))
    if tile_cap == peak_tiles and kv_cap == peak_kv:
        return ranges

    # Keep the original cut when possible while reserving a feasible nonempty
    # suffix for the remaining cores.
    earliest = [rows]
    for _ in range(grid - 1):
        earliest.append(suffix_start(earliest[-1], tile_cap, kv_cap))
    balanced = []
    start = 0
    for core, (old_start, old_count) in enumerate(ranges[:-1]):
        remaining = grid - core - 1
        low = max(start + 1, earliest[remaining])
        high = min(rows - remaining, start + peak_queries,
                   bisect_right(tile_prefix, tile_prefix[start] + tile_cap) - 1,
                   bisect_right(kv_prefix, kv_prefix[start] + kv_cap) - 1)
        end = min(max(old_start + old_count, low), high)
        balanced.append((start, end - start))
        start = end
    balanced.append((start, rows - start))
    return balanced


def _cost_meta(host_lengths, rows, grid, full_lengths, width, *, task_pipeline=False):
    """Describe each core's Query range, plus S2 tasks only for varying lengths.

    Full-length inputs return the two-column (start, count) header only: every
    Query covers the same static tile count, so the kernel derives the row,
    tile, query boundary and Q slot arithmetically. Explicit lengths that were
    host-validated to equal T take this path too, without another D2H read.
    Variable-length inputs keep the cost-balanced ranges and, for the streaming
    schedule, a flat per-tile descriptor table.
    """
    if full_lengths:
        per, extra = divmod(rows, grid)
        ranges = [(core * per + min(core, extra), per + (core < extra))
                  for core in range(grid)]
        return torch.tensor(ranges, dtype=torch.int32)
    if host_lengths is None:
        # No host-visible lengths and not known full: the per-query fallback
        # uses two columns, while a task-stream kernel keeps its table width.
        cols = _meta_cols(rows, width, grid) if task_pipeline else 2
        return torch.full((grid, cols), -1, dtype=torch.int32)
    lengths = [int(length) for length in host_lengths.tolist()]
    tile_counts = [max((length + 127) // 128, 1) for length in lengths]
    prefix = list(accumulate(tile_counts, initial=0))
    ranges = []
    start = 0
    for core in range(1, grid):
        target = core * prefix[-1] // grid
        end = min(max(bisect_left(prefix, target), start + 1), rows - (grid - core))
        if end > start + 1 and abs(prefix[end - 1] - target) <= abs(prefix[end] - target):
            end -= 1
        ranges.append((start, end - start))
        start = end
    ranges.append((start, rows - start))
    if not task_pipeline:
        return torch.tensor(ranges, dtype=torch.int32)

    ranges = _balance_streaming_ranges(lengths, tile_counts, prefix, ranges)
    cols = _meta_cols(rows, width, grid)
    limit = _max_tasks_per_core(rows, width, grid)
    out = torch.full((grid, cols), -1, dtype=torch.int32)
    for core, (start, count) in enumerate(ranges):
        tasks = []
        for i in range(start, start + count):
            tiles = tile_counts[i]
            for tile in range(tiles):
                valid = min(128, max(0, lengths[i] - tile * 128))
                tasks.append((i, tile, valid, int(tile == 0),
                              int(tile + 1 == tiles), (2 * i) % 3))
        if len(tasks) > limit:
            raise ValueError("per-core task table exceeds its static bound")
        out[core, 0] = start
        out[core, 1] = count
        out[core, 2] = len(tasks)
        out[core, 3:3 + len(tasks) * TASK_STRIDE] = torch.tensor(tasks, dtype=torch.int32).flatten()
    return out


def _benchmark_query_pipeline_eligible(inputs, cores):
    sq, _, width = inputs["indices"].shape
    return _resolve_query_pipeline(sq, width, cores, True, None)


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


def _core_count(q=None):
    """Use the tensor's NPU, or the current NPU when building without a tensor."""
    if not hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401; registers torch.npu lazily.
        except ImportError as exc:
            raise RuntimeError("Automatic core detection requires torch_npu; pass cores explicitly for offline source generation") from exc
    device = q.device if q is not None else torch.npu.current_device()
    # Each mixed block uses one Cube core and its two Vector subblocks.
    cores = int(torch.npu.get_device_properties(device).cube_core_num)
    if cores <= 0:
        raise RuntimeError(f"Expected a positive Cube core count for {device}, got {cores}")
    return cores


def _build_multi_tile(sq, skv, width, scale, cores, *, pipeline_stages,
                      pair_dma, prefetch_kv, pipeline_queries,
                      full_valid_kv, fixed_offset, full_length=False,
                      workspace_l2_cache="NORMAL_FV", unroll_pairs=False,
                      input_sq=None, kv_split=1):
    """Online attention over contiguous Query ranges.

    Full-length runs give each core only (start, count); row, tile, the query
    boundary and the Q slot are derived at task time, so no per-tile metadata
    is built, uploaded or read. Variable-length runs keep a cost-balanced
    range and, for the streaming schedule, a flat per-tile descriptor.
    Direct KV gathers alternate 16-row groups between AIVs and publish them
    into L1. In the workspace path, AIV0 gathers the first ceil(N/2) rows and
    AIV1 the rest. Cube reads a packed full tile with one ND2NZ, or reads 32
    rows from each half for a short tail.
    The general schedule uses separate GM buffers and a validity mask in
    L1 row order. Variable short tails use one score vector per head and
    16/48/64-row Cube buckets. The 16-row bucket only loads AIV0's first
    KV group.
    """
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S
    from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_l0c_layout

    block, qk_d, pv_d = KV_TILE_ROWS, 128, 128
    input_sq = sq if input_sq is None else input_sq
    input_width = width * kv_split
    output_dtype = "float32" if kv_split > 1 else "bfloat16"
    q_half, q_slots = D // 2, 3
    half, gather_rows, pitch = H // 2, 16, H // 2 + 1
    grid = min(sq, cores)
    kv_slots = KV_WORKSPACE_SLOTS
    full_length = bool(full_length)
    tiles_per_query = (width + block - 1) // block
    # With full lengths and legal indices, both AIVs know their exact row
    # count for the 128-row tiles and the optional 64-row tail. full_blocks
    # separates the statically 128-wide case from the tail case.
    full_blocks = bool(full_length and width % block == 0)
    packed_kv = _packed_kv_workspace(width, full_length, pipeline_queries, full_valid_kv)
    packed_tail = bool(packed_kv and width % block)
    legal_indices = bool(full_valid_kv)
    static_full = bool(full_length and legal_indices)
    direct_kv_l1 = _use_direct_kv_l1(width, full_length, pipeline_queries,
                                     full_valid_kv, pair_dma, kv_split)
    meta_cols = 2 if (full_length or not pipeline_queries) else _meta_cols(sq, width, grid)
    # Match the baseline L1 budget for the pipelined path: three KV slots,
    # two P slots and the three Q half slots.
    stages = 2 if pipeline_stages == 2 and prefetch_kv else 1
    # Task stages address their buffers by logical task. In the nested schedule
    # the tile loop restarts for each Query, so those buffers use a counter.
    if pipeline_queries:
        kv_versions = (kv_slots, "iteration")
        p_versions = (2, "iteration")
    else:
        kv_versions = (3, "counter") if stages == 2 else 1
        p_versions = (2, "counter") if stages == 2 else 1
    max_pair_delta = (2**31 - 1) // (D * 2)

    def tail_rows_for(n):
        # AscendC-style tail: keep the Cube N at the real prefix instead of
        # always computing 64 rows. AIV1 publishes into the second 32-row
        # half, so n valid rows span 32 + n//2 rows (n itself when only
        # AIV0 has data). Round up to the 16-row MMA granularity.
        if static_full:
            return n
        return T.if_then_else(n <= 1, 16, T.if_then_else(n <= 33, 48, 64))

    @T.macro
    def zero_row(nd, token):
        with T.SimdVF():
            full16 = S.pset(16, "PAT_ALL")
            zero16 = S.vdup(T.bfloat16(0), "bfloat16", full16)
            for chunk in T.serial(D // 128):
                S.vsts(nd[token, chunk * 128], zero16, full16, "NORM_B16")

    @T.macro
    def copy_kv_pair(Cache, nd, token, first, second, first_valid=True, second_valid=True):
        start = T.min(first, second)
        delta = T.max(first, second) - start
        if first_valid & second_valid & (delta == 1):
            T.copy(Cache[start:start + 2, :], nd[token:token + 2, :])
        else:
            # Keep the same row order as the previous paired gather.
            swap = first_valid & second_valid & (first > second) & (delta <= max_pair_delta)
            if first_valid:
                T.copy(Cache[T.if_then_else(swap, second, first), :], nd[token, :])
            if second_valid:
                T.copy(Cache[T.if_then_else(swap, first, second), :], nd[token + 1, :])

    @T.macro
    def gather_group(Cache, index_ub, nd, src_start, real_rows, full_rows):
        # real_rows is clipped to the part of this group inside the prefix.
        # full_rows marks a statically complete 16-row group: no row guard,
        # padding write or index bounds check is needed.
        if full_rows:
            if pair_dma:
                # Explicit unrolling removes the loop-carried MTE2 barrier so
                # the first 16-row store can issue while the second slot still
                # gathers. Binary size grows ~17%; enable only when it wins.
                if unroll_pairs:
                    for pair in T.unroll(gather_rows // 2, explicit=True):
                        token = pair * 2
                        src = src_start + token
                        first = T.cast(index_ub[src], "int64")
                        second = T.cast(index_ub[src + 1], "int64")
                        copy_kv_pair(Cache, nd, token, first, second)
                else:
                    for pair in T.serial(gather_rows // 2):
                        token = pair * 2
                        src = src_start + token
                        first = T.cast(index_ub[src], "int64")
                        second = T.cast(index_ub[src + 1], "int64")
                        copy_kv_pair(Cache, nd, token, first, second)
            else:
                for token in T.serial(gather_rows):
                    index = T.cast(index_ub[src_start + token], "int64")
                    T.copy(Cache[index, :], nd[token, :])
        elif direct_kv_l1:
            # Full blocks contain whole groups. Copy or zero each row once so
            # the Vector pipe can run softmax while MTE2 gathers a later tile.
            for token in T.serial(gather_rows):
                index = T.cast(index_ub[src_start + token], "int64")
                if (index >= 0) & (index < skv):
                    T.copy(Cache[index, :], nd[token, :])
                else:
                    zero_row(nd, token)
        elif legal_indices:
            # The host validated every index, so only the prefix length
            # selects rows; no per-row [0, Skv) comparisons remain.
            if pair_dma:
                for pair in T.serial(gather_rows // 2):
                    token = pair * 2
                    if token < real_rows:
                        src = src_start + token
                        first = T.cast(index_ub[src], "int64")
                        second = T.cast(index_ub[T.min(src + 1, block - 1)], "int64")
                        copy_kv_pair(Cache, nd, token, first, second, True, token + 1 < real_rows)
            else:
                for token in T.serial(gather_rows):
                    if token < real_rows:
                        index = T.cast(index_ub[src_start + token], "int64")
                        T.copy(Cache[index, :], nd[token, :])
        else:
            if pair_dma:
                for pair in T.serial(gather_rows // 2):
                    token = pair * 2
                    if token < real_rows:
                        src = src_start + token
                        first = T.cast(index_ub[src], "int64")
                        second = T.cast(index_ub[T.min(src + 1, block - 1)], "int64")
                        ok0 = (first >= 0) & (first < skv)
                        ok1 = (token + 1 < real_rows) & (second >= 0) & (second < skv)
                        copy_kv_pair(Cache, nd, token, first, second, ok0, ok1)
            elif full_length:
                for token in T.serial(gather_rows):
                    if token < real_rows:
                        index = T.cast(index_ub[src_start + token], "int64")
                        if (index >= 0) & (index < skv):
                            T.copy(Cache[index, :], nd[token, :])
            else:
                # Keep the DMA loop separate from vector writes. A complete
                # group with legal indices needs neither zeroing loop.
                has_invalid = T.alloc_var("int32", init=0)
                for token in T.serial(real_rows):
                    index = T.cast(index_ub[src_start + token], "int64")
                    if (index >= 0) & (index < skv):
                        T.copy(Cache[index, :], nd[token, :])
                    else:
                        has_invalid = 1
                if has_invalid != 0:
                    for token in T.serial(real_rows):
                        index = index_ub[src_start + token]
                        if (index < 0) | (index >= skv):
                            zero_row(nd, token)
                for token in T.serial(real_rows, gather_rows):
                    zero_row(nd, token)
        if not (full_rows or direct_kv_l1) and (full_length or pair_dma or legal_indices):
            for token in T.serial(gather_rows):
                if legal_indices:
                    invalid = token >= real_rows
                else:
                    index = index_ub[T.min(src_start + token, block - 1)]
                    invalid = (token >= real_rows) | (index < 0) | (index >= skv)
                if invalid:
                    zero_row(nd, token)

    @T.macro
    def gather_half(Cache, KVWorkspace, index_ub, nd0, nd1, bx, slot,
                    dst_base, source_start, real_rows, group_pairs, full_rows, first_group_only=False):
        # Publish each 16-row half as soon as its MTE2 gathers finish. The next
        # half uses the other UB slot, so its MTE2 work can overlap this MTE3.
        # Keep the loop bound visible to dependency analysis when a tail and
        # a full block publish into the same 128-row storage layout.
        groups = 2 if packed_tail else group_pairs
        for group in T.serial(groups):
            if not packed_tail or group < group_pairs:
                offset = group * (2 * gather_rows)
                if full_rows:
                    gather_group(Cache, index_ub, nd0, source_start + offset, gather_rows, True)
                    T.copy(nd0, KVWorkspace[bx, slot, dst_base + offset:dst_base + offset + gather_rows, :],
                           l2_cache_ctrl=workspace_l2_cache)
                    gather_group(Cache, index_ub, nd1, source_start + offset + gather_rows,
                                 gather_rows, True)
                    T.copy(nd1, KVWorkspace[bx, slot,
                           dst_base + offset + gather_rows:dst_base + offset + 2 * gather_rows, :],
                           l2_cache_ctrl=workspace_l2_cache)
                else:
                    gather_group(Cache, index_ub, nd0, source_start + offset,
                                 T.max(T.min(gather_rows, real_rows - offset), 0), False)
                    T.copy(nd0, KVWorkspace[bx, slot, dst_base + offset:dst_base + offset + gather_rows, :],
                           l2_cache_ctrl=workspace_l2_cache)
                    if not first_group_only:
                        offset1 = offset + gather_rows
                        gather_group(Cache, index_ub, nd1, source_start + offset1,
                                     T.max(T.min(gather_rows, real_rows - offset1), 0), False)
                        T.copy(nd1, KVWorkspace[bx, slot, dst_base + offset1:dst_base + offset1 + gather_rows, :],
                               l2_cache_ctrl=workspace_l2_cache)

    @T.macro
    def gather(Cache, Indices, KVWorkspace0, KVWorkspace1, bx, row, tile,
               slot, sid, nd0, nd1, valid, n, tail64, kv_l1=None):
        index_ub = T.alloc_shared((block,), "int32")
        T.annotate_buffer_versions({index_ub: 1})
        input_row = row // kv_split
        column = (row % kv_split) * width + tile * block
        if width % block == 0:
            T.copy(Indices[input_row, column:column + block], index_ub)
        else:
            if tile * block + block <= width:
                T.copy(Indices[input_row, column:column + block], index_ub)
            else:
                T.copy(Indices[input_row, column:column + 64], index_ub[:64])
        if static_full:
            if width % block == 0:
                group_pairs = 2
                k0 = block // 2
            else:
                group_pairs = T.if_then_else(tail64, 1, 2)
                k0 = group_pairs * (2 * gather_rows)
            k1 = k0
        else:
            # n is the tile's real prefix; k0/k1 are the two AIVs' real counts.
            # The Cube reads 64 or 128 rows (tail64), so the stored order is
            # [0,k0) then [half, half+k1): AIV1 shifts its rows into its own
            # 64-row half, and the mask below follows that permutation.
            if full_blocks:
                # The block width is statically 128; only index legality can
                # still vary, so each half is a full 64-row scan and the
                # second half starts at an aligned 64-entry offset.
                k0 = block // 2
                k1 = block // 2
                group_pairs = 2
            else:
                # n is nonnegative. Integer division keeps the prefix bounds
                # in the compiler's arithmetic model during dependency analysis.
                k0 = (n + 1) // 2
                k1 = n - k0
                group_pairs = T.if_then_else(tail64, 1, 2)
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                one = S.vdup(T.int32(1), "int32", full)
                zero = S.vdup(T.int32(0), "int32", full)
                if full_blocks:
                    index0 = S.vld(index_ub[0])
                    vis0 = S.vcmps(index0, T.int32(0), full, "ge")
                    vis0 = S.pand(vis0, S.vcmps(index0, T.int32(skv), full, "lt"), full)
                    index1 = S.vld(index_ub[block // 2])
                    vis1 = S.vcmps(index1, T.int32(0), full, "ge")
                    vis1 = S.pand(vis1, S.vcmps(index1, T.int32(skv), full, "lt"), full)
                else:
                    lane = S.vci(0, "int32")
                    # The per-AIV prefix counts already clip the lanes to the
                    # valid span of this tile.
                    vis0 = S.vcmps(lane, k0, full, "lt")
                    vis1 = S.vcmps(lane, k1, full, "lt")
                    if not legal_indices:
                        index0 = S.vld(index_ub[0])
                        vis0 = S.pand(vis0, S.vcmps(index0, T.int32(0), full, "ge"), full)
                        vis0 = S.pand(vis0, S.vcmps(index0, T.int32(skv), full, "lt"), full)
                        # The second half can start at an unaligned offset.
                        split_indices = T.reinterpret(S.vci(k0, "int32"), "uint32x64")
                        if width % block == 0:
                            gather_mask = full
                        else:
                            # On the physical 64-column tail, k0 and k1 are
                            # at most 32. Only read the initialized prefix.
                            gather_mask = S.update_mask(k1, width=32)
                        index1 = S.vgather2(T.access_ptr(index_ub[0], "r", extent=block),
                                            split_indices, gather_mask)
                        vis1 = S.pand(vis1, S.vcmps(index1, T.int32(0), full, "ge"), full)
                        vis1 = S.pand(vis1, S.vcmps(index1, T.int32(skv), full, "lt"), full)
                valid0 = S.vsel(one, zero, vis0)
                valid1 = S.vsel(one, zero, vis1)
                if full_blocks:
                    S.vsts(valid[0], valid0, full)
                    S.vsts(valid[64], valid1, full)
                elif tail64:
                    first32 = S.pset(32, "PAT_VL32")
                    S.vsts(valid[0], valid0, first32, extent=32)
                    S.vsts(valid[32], valid1, first32, extent=32)
                    S.vsts(valid[64], zero, full)
                else:
                    S.vsts(valid[0], valid0, full)
                    S.vsts(valid[64], valid1, full)
        if direct_kv_l1:
            # Each dual copy joins two adjacent stripes. Gather in the same
            # order so the L1 rows still correspond to the original mask.
            for group in T.serial(block // (4 * gather_rows)):
                start0 = group * 4 * gather_rows
                start1 = start0 + 2 * gather_rows
                gather_group(Cache, index_ub, nd0, start0 + sid * gather_rows, gather_rows, False)
                T.dual_copy(nd0, kv_l1[start0:start0 + 2 * gather_rows, :])
                gather_group(Cache, index_ub, nd1, start1 + sid * gather_rows, gather_rows, False)
                T.dual_copy(nd1, kv_l1[start1:start1 + 2 * gather_rows, :])
        elif sid == 0:
            gather_half(Cache, KVWorkspace0, index_ub, nd0, nd1, bx, slot,
                        0, 0, k0, group_pairs, static_full, not full_length and n <= 1)
        elif full_length or n > 1:
            # Only AIV0 supplies KV for a 16-row Cube tile. AIV1 still
            # produces its validity mask and participates in Vec1 and Vec2.
            if packed_kv:
                gather_half(Cache, KVWorkspace0, index_ub, nd0, nd1, bx, slot,
                            block // 2, k0, k1, group_pairs, static_full)
            else:
                gather_half(Cache, KVWorkspace1, index_ub, nd0, nd1, bx, slot,
                            0, k0, k1, group_pairs, static_full)

    @T.macro
    def load_kv_tile(KVWorkspace0, KVWorkspace1, kv_l1, bx, slot, n, tail64):
        if not full_length and n <= 1:
            T.copy(KVWorkspace0[bx, slot, :gather_rows, :], kv_l1[:gather_rows, :])
        elif tail64:
            T.copy(KVWorkspace0[bx, slot, :32, :], kv_l1[:32, :])
            if packed_kv:
                T.copy(KVWorkspace0[bx, slot, block // 2:block // 2 + 32, :], kv_l1[32:64, :])
            else:
                T.copy(KVWorkspace1[bx, slot, :32, :], kv_l1[32:64, :])
        elif packed_kv:
            T.copy(KVWorkspace0[bx, slot, :, :], kv_l1)
        else:
            T.copy(KVWorkspace0[bx, slot, :, :], kv_l1[:64, :])
            T.copy(KVWorkspace1[bx, slot, :, :], kv_l1[64:128, :])

    @T.macro
    def load_kv(KVWorkspace0, KVWorkspace1, kv_l1, bx, slot, n, tail64):
        if full_length:
            load_kv_tile(KVWorkspace0, KVWorkspace1, kv_l1, bx, slot, n, tail64)
        else:
            # The mutually exclusive KV geometries share one MTE2 issue slot.
            with T.Task():
                load_kv_tile(KVWorkspace0, KVWorkspace1, kv_l1, bx, slot, n, tail64)

    @T.macro
    def load_q_half(Q, q_l1, row, slot, half_idx):
        # Each half is one 64x256 ND2NZ transfer; the GM row stride stays 512.
        input_row = row // kv_split
        T.copy(Q[input_row * H:(input_row + 1) * H, half_idx * q_half:(half_idx + 1) * q_half],
               q_l1[slot, :, :])

    @T.macro
    def qk(q_l1, kv_l1, qa, kb, scores, score_ub, slot0, slot1, tile_rows):
        # Both 256-wide Q halves advance the shared two-slot counter. Counter
        # mode keeps QK and PV aliases on one ring even when the outer task
        # pipeline reorders their disjoint, write-first inner-loop owners.
        for sub in T.serial(2):
            T.copy(q_l1[slot0, :, sub * qk_d:(sub + 1) * qk_d], qa)
            T.copy(kv_l1[:tile_rows, sub * qk_d:(sub + 1) * qk_d], kb[:tile_rows, :])
            T.gemm(qa, kb[:tile_rows, :], scores[:, :tile_rows], transpose_B=True,
                   clear_accum=(sub == 0))
        for sub in T.serial(2):
            T.copy(q_l1[slot1, :, sub * qk_d:(sub + 1) * qk_d], qa)
            T.copy(kv_l1[:tile_rows, (sub + 2) * qk_d:(sub + 3) * qk_d], kb[:tile_rows, :])
            T.gemm(qa, kb[:tile_rows, :], scores[:, :tile_rows], transpose_B=True,
                   clear_accum=False)
        T.dual_copy(scores[:, :tile_rows], score_ub[:, :tile_rows])

    @T.macro
    def softmax_tile(tile, score_ub, valid, raw, stats, p_scale, m, ell, alpha, sink_ub,
                     tail64, tail_rows, state, task_max, task_sum, masked, short):
        full = S.pset(32, "PAT_ALL")
        heads = S.pset(32, "PAT_VL32")
        scalar = S.pset(32, "PAT_VL1")
        p_mask = S.pset(16, "PAT_VL64")
        zero = S.vdup(T.float32(0), "float32", full)
        one = S.vdup(T.float32(1), "float32", full)
        neg_inf = S.vdup(-T.infinity("float32"), "float32", full)
        if tile == 0:
            S.vsts(m[state, 0], neg_inf, full)
            S.vsts(ell[state, 0], zero, full)
            S.mem_bar("VST_VLD")
        if short:
            if tail_rows < 64:
                # FIXP only initialized the active Cube columns. The
                # aligned vector load also reads the rest of this half.
                lanes = S.vci(0, "int32")
                padding = S.vcmps(lanes, tail_rows, full, "ge")
                for h in T.serial(half):
                    S.vsts(score_ub[h, 0], neg_inf, padding)
                S.mem_bar("VST_VLD")
        elif tail64:
            for h in T.serial(half):
                S.vsts(score_ub[h, 64], neg_inf, full)
            S.mem_bar("VST_VLD")
        if masked:
            v0 = S.vcmps(S.vld(valid[0]), T.int32(0), full, "ne")
            if not short:
                v1 = S.vcmps(S.vld(valid[64]), T.int32(0), full, "ne")
        elif width % block != 0:
            # Full-length tails have exactly 64 rows. Keep the unused
            # score half out of the sum even if the active scores are NaN.
            tail_lanes = S.vdup(T.cast(tail64, "int32"), "int32", full)
            v1 = S.vcmps(tail_lanes, T.int32(0), full, "eq")
        for h in T.serial(half):
            r0 = S.vld(score_ub[h, 0])
            if masked:
                r0 = S.vsel(r0, neg_inf, v0)
            if short:
                row_max = r0
            else:
                r1 = S.vld(score_ub[h, 64])
                if masked:
                    r1 = S.vsel(r1, neg_inf, v1)
                row_max = S.vmax(r0, r1, full)
            # The row maximum is taken on raw scores; scaling the single
            # row result is equivalent because scale is positive.
            S.vsts(stats[h], S.vcmax(row_max, full), scalar, "ONEPT_B32")
        S.mem_bar("VST_VLD")
        old_m = S.vld(m[state, 0])
        new_m = S.vmax(old_m, S.vmuls(S.vld(stats[0]), T.float32(scale), heads), heads)
        if fixed_offset is None:
            empty = S.vcmps(new_m, -T.infinity("float32"), heads, "eq")
            origin = S.vsel(zero, new_m, empty)
            old_empty = S.vcmps(old_m, -T.infinity("float32"), heads, "eq")
            factor = S.vsel(zero, S.vexp(S.vsub(old_m, origin, heads), heads), old_empty)
            sink = S.vld(sink_ub[0])
            sink_infinite = S.vcmps(sink, T.infinity("float32"), heads, "eq")
            output_max = S.vmax(new_m, sink, heads)
            old_output_max = S.vmax(old_m, sink, heads)
            degenerate = S.por(sink_infinite,
                               S.vcmps(output_max, -T.infinity("float32"), heads, "eq"), heads)
            output_origin = S.vsel(zero, output_max, degenerate)
            output_alpha = S.vsel(one, S.vexp(S.vsub(old_output_max, output_origin, heads), heads), degenerate)
            output_scale = S.vsel(zero, S.vexp(S.vsub(new_m, output_origin, heads), heads), sink_infinite)
            S.vsts(p_scale[0], output_scale, heads)
        else:
            origin = S.vdup(T.float32(fixed_offset), "float32", full)
            factor = one
            output_alpha = one
        S.vsts(m[state, 0], new_m, full)
        S.vsts(alpha[0], output_alpha, heads)
        S.vsts(stats[0], origin, full)
        S.mem_bar("VST_VLD")
        p0_ptr = S.make_ubuf_ptr(T.access_ptr(raw[0], "w", extent=4 * pitch * 16), "bfloat16")
        if not short:
            p1_ptr = S.make_ubuf_ptr(T.access_ptr(raw[4 * pitch * 16], "w", extent=4 * pitch * 16), "bfloat16")
        for h in T.serial(half):
            origin_h = S.vld(stats[h], "BRC_B32")
            # Reload and scale the QK scores without a scaled-score buffer.
            s0 = S.vmuls(S.vld(score_ub[h, 0]), T.float32(scale), full)
            if masked:
                s0 = S.vsel(s0, neg_inf, v0)
            p0 = S.vexp(S.vsub(s0, origin_h, full), full)
            if masked:
                p0 = S.vsel(p0, zero, v0)
            if short:
                row_prob = p0
            else:
                s1 = S.vmuls(S.vld(score_ub[h, 64]), T.float32(scale), full)
                if masked:
                    s1 = S.vsel(s1, neg_inf, v1)
                p1 = S.vexp(S.vsub(s1, origin_h, full), full)
                if masked or width % block != 0:
                    p1 = S.vsel(p1, zero, v1)
                row_prob = S.vadd(p0, p1, full)
            S.vsts(stats[h], S.vcadd(row_prob, full), scalar, "ONEPT_B32")
            if fixed_offset is None:
                # Keep the exact KV-only sum above. Cube receives weights
                # on the output scale that includes the sink anchor.
                scale_h = S.vld(p_scale[h], "BRC_B32")
                b0 = S.vcvt(S.vmul(p0, scale_h, full), "bfloat16", full, sat=False, part=0)
                if not short:
                    b1 = S.vcvt(S.vmul(p1, scale_h, full), "bfloat16", full, sat=False, part=0)
            else:
                b0 = S.vcvt(p0, "bfloat16", full, sat=False, part=0)
                if not short:
                    b1 = S.vcvt(p1, "bfloat16", full, sat=False, part=0)
            pack0 = T.reinterpret(S.vpack(T.reinterpret(b0, "uint32x64"), 0), "bfloat16x128")
            p0_ptr = S.vsstb(pack0, p0_ptr, (pitch << 16) | 1, p_mask, update=True)
            if not short:
                pack1 = T.reinterpret(S.vpack(T.reinterpret(b1, "uint32x64"), 0), "bfloat16x128")
                p1_ptr = S.vsstb(pack1, p1_ptr, (pitch << 16) | 1, p_mask, update=True)
        S.mem_bar("VST_VLD")
        total = S.vadd(S.vmul(factor, S.vld(ell[state, 0]), heads), S.vld(stats[0]), heads)
        S.vsts(ell[state, 0], total, full)
        if pipeline_queries:
            S.vsts(task_max[0, 0], new_m, heads)
            S.vsts(task_sum[0, 0], total, heads)

    @T.macro
    def softmax(tile, score_ub, valid, prob_ub, p_l1, m, ell, alpha, sink_ub, tail64, tail_rows,
                state, task_max, task_sum):
        stats = T.alloc_shared((64,), "float32")
        p_scale = T.alloc_shared((64,), "float32")
        raw = T.Tensor((pitch * block,), "bfloat16", prob_ub.data, scope="shared.dyn")
        T.annotate_buffer_versions({stats: 1, p_scale: 1})
        with T.SimdVF():
            if not full_length and tail64:
                softmax_tile(tile, score_ub, valid, raw, stats, p_scale, m, ell, alpha, sink_ub,
                             tail64, tail_rows, state, task_max, task_sum, True, True)
            else:
                softmax_tile(tile, score_ub, valid, raw, stats, p_scale, m, ell, alpha, sink_ub,
                             tail64, tail_rows, state, task_max, task_sum, not static_full, False)
        # Publish P on MTE3 after both scan passes.
        if full_length:
            T.dual_copy(prob_ub[:half, :], p_l1)
        else:
            with T.Task():
                if tail64:
                    T.dual_copy(prob_ub[:half, :64], p_l1[:, :64])
                else:
                    T.dual_copy(prob_ub[:half, :], p_l1)

    @T.macro
    def pv(kv_l1, p_l1, pa, vb, product_parts, product, product_ub, tile_rows):
        if direct_kv_l1:
            T.copy(p_l1[:, :tile_rows], pa[:, :tile_rows])
        for part in T.serial(D // pv_d):
            if not direct_kv_l1:
                # On the shared L0A ring each panel owns its operand writes.
                T.copy(p_l1[:, :tile_rows], pa[:, :tile_rows])
            T.copy(kv_l1[:tile_rows, part * pv_d:(part + 1) * pv_d], vb[:, :tile_rows], transpose=True)
            T.gemm(pa[:, :tile_rows], vb[:, :tile_rows], product_parts[part, :, :],
                   transpose_B=True, clear_accum=True)
        T.dual_copy(product, product_ub)

    @T.macro
    def qk_tail(q_l1, kv_l1, qa, kb, scores, score_ub, slot0, slot1, n):
        # Guard the static GEMM shapes with the prefix bounds directly.
        # Comparing the result of tail_rows_for() would put a nested select
        # inside each condition the scheduler and sync passes have to prove.
        if n <= 1:
            qk(q_l1, kv_l1, qa, kb, scores, score_ub, slot0, slot1, 16)
        elif n <= 33:
            qk(q_l1, kv_l1, qa, kb, scores, score_ub, slot0, slot1, 48)
        else:
            qk(q_l1, kv_l1, qa, kb, scores, score_ub, slot0, slot1, 64)

    @T.macro
    def pv_tail(kv_l1, p_l1, pa, vb, product_parts, product, product_ub, n):
        # PV contracts over the same real prefix, so invalid KV rows are
        # excluded from the GEMM instead of contributing 0 * KV.
        if n <= 1:
            pv(kv_l1, p_l1, pa, vb, product_parts, product, product_ub, 16)
        elif n <= 33:
            pv(kv_l1, p_l1, pa, vb, product_parts, product, product_ub, 48)
        else:
            pv(kv_l1, p_l1, pa, vb, product_parts, product, product_ub, 64)

    @T.macro
    def update_output(tile, product_ub, accum, alpha):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            for h in T.serial(half):
                if tile == 0:
                    for chunk in T.serial(D // 64):
                        value = S.vld(product_ub[h, chunk * 64])
                        S.vsts(accum[h, chunk * 64], value, full, "NORM_B32")
                else:
                    factor = S.vld(alpha[h], "BRC_B32")
                    for chunk in T.serial(D // 64):
                        previous = S.vld(accum[h, chunk * 64])
                        current = S.vld(product_ub[h, chunk * 64])
                        value = S.vadd(S.vmul(previous, factor, full), current, full)
                        S.vsts(accum[h, chunk * 64], value, full, "NORM_B32")

    @T.macro
    def finish(row, O, MaxLogits, LSE, accum, out_ub, m, ell, sink_ub,
               max_ub, lse_ub, inv_ub, sid, state):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            heads = S.pset(32, "PAT_VL32")
            zero = S.vdup(T.float32(0), "float32", full)
            one = S.vdup(T.float32(1), "float32", full)
            inf = S.vdup(T.infinity("float32"), "float32", full)
            maximum = S.vld(m[state, 0])
            total = S.vld(ell[state, 0])
            nonempty = S.vcmps(total, T.float32(0), heads, "gt")
            safe_total = S.vsel(total, one, nonempty)
            if fixed_offset is None:
                origin = S.vsel(maximum, zero, nonempty)
            else:
                origin = S.vdup(T.float32(fixed_offset), "float32", full)
            lse = S.vsel(S.vadd(origin, S.vln(safe_total, heads), heads), inf, nonempty)
            sink = S.vld(sink_ub[0])
            if fixed_offset is None:
                sink_infinite = S.vcmps(sink, T.infinity("float32"), heads, "eq")
                output_max = S.vsel(zero, S.vmax(origin, sink, heads), sink_infinite)
                kv_term = S.vmul(safe_total, S.vexp(S.vsub(origin, output_max, heads), heads), heads)
                sink_term = S.vexp(S.vsub(sink, output_max, heads), heads)
            else:
                kv_term = safe_total
                sink_term = S.vexp(S.vsub(sink, origin, heads), heads)
            inverse = S.vsel(S.vdiv(one, S.vadd(kv_term, sink_term, heads), heads,
                                   precision="exact"), zero, nonempty)
            S.vsts(max_ub[0], maximum, heads, extent=half)
            S.vsts(lse_ub[0], lse, heads, extent=half)
            S.vsts(inv_ub[0], inverse, heads, extent=half)
            S.mem_bar("VST_VLD")
            # Forward compaction is safe: each BF16 write lies before the
            # FP32 input bytes consumed by any following head or chunk.
            for h in T.serial(half):
                inverse_h = S.vld(inv_ub[h], "BRC_B32")
                for chunk in T.serial(D // 64):
                    value = S.vmul(S.vld(accum[h, chunk * 64]), inverse_h, full)
                    if kv_split > 1:
                        S.vsts(out_ub[h, chunk * 64], value, full, "NORM_B32")
                    else:
                        packed = S.vcvt(value, "bfloat16", full, sat=False, part=0)
                        S.vsts(out_ub[h, chunk * 64], packed, full, "PK_B32")
        if pipeline_queries:
            start = row * H + sid * half
            T.copy(out_ub, O[start:start + half, :])
            T.copy(max_ub, MaxLogits[start:start + half])
            T.copy(lse_ub, LSE[start:start + half])
        else:
            T.dual_copy(out_ub, O[row * H:(row + 1) * H, :])
            T.dual_copy(max_ub, MaxLogits[row * H:(row + 1) * H])
            T.dual_copy(lse_ub, LSE[row * H:(row + 1) * H])

    @T.prim_func
    def main(Q: T.Tensor((input_sq * H, D), "bfloat16"),
             Cache: T.Tensor((skv, D), "bfloat16"),
             Indices: T.Tensor((input_sq, input_width), "int32"),
             Lengths: T.Tensor((input_sq,), "int32"),
             Extra: T.Tensor((1, D), "bfloat16"),
             ExtraIndices: T.Tensor((input_sq, 1), "int32"),
             ExtraLengths: T.Tensor((input_sq,), "int32"),
             Sink: T.Tensor((H,), "float32"),
             Meta: T.Tensor((grid, meta_cols), "int32"),
             KVWorkspace0: T.Tensor((1,) if direct_kv_l1 else
                                    (grid, kv_slots, block if packed_kv else block // 2, D), "bfloat16"),
             KVWorkspace1: T.Tensor((1,) if packed_kv else (grid, kv_slots, block // 2, D), "bfloat16"),
             O: T.Tensor((sq * H, D), output_dtype),
             MaxLogits: T.Tensor((sq * H,), "float32"),
             LSE: T.Tensor((sq * H,), "float32")):
        with T.MixedKernel(grid, sids=2) as (bx, sid):
            # Keep the slot axis outside the NZ matrix so every half has its
            # own compact layout and 64-row NZ pitch.
            q_l1 = T.alloc_l1((q_slots, H, q_half), "bfloat16")
            kv_l1 = T.alloc_l1((block, D), "bfloat16")
            p_l1 = T.alloc_l1((H, block), "bfloat16")
            qa = T.alloc_l0a((H, qk_d), "bfloat16")
            if direct_kv_l1:
                # Two 16 KiB Q slots and one 16 KiB P slot fit in L0A.
                pa = T.alloc_l0a((H, block), "bfloat16")
                T.annotate_buffer_versions({pa: 1})
            else:
                pa = T.Tensor((H, block), "bfloat16", qa.data, scope="shared.l0a")
            kb = T.alloc_l0b((block, qk_d), "bfloat16")
            vb = T.Tensor((pv_d, block), "bfloat16", kb.data, scope="shared.l0b")
            scores = T.alloc_l0c((H, block), "float32")
            product_parts = T.alloc_l0c((D // pv_d, H, pv_d), "float32")
            product = T.Tensor((H, D), "float32", product_parts.data, scope="shared.l0c")
            nd0 = T.alloc_shared((gather_rows, D), "bfloat16")
            nd1 = T.alloc_shared((gather_rows, D), "bfloat16")
            valid = T.alloc_shared((1 if static_full else block,), "int32")
            score_ub = T.alloc_shared((half, block), "float32")
            prob_ub = T.alloc_shared((pitch, block), "bfloat16")
            product_ub = T.alloc_shared((half, D), "float32")
            accum = T.alloc_shared((half, D), "float32")
            out_ub = T.Tensor((half, D), output_dtype, accum.data, scope="shared.dyn")
            m = T.alloc_shared((2 if pipeline_queries else 1, 64), "float32")
            ell = T.alloc_shared((2 if pipeline_queries else 1, 64), "float32")
            if pipeline_queries:
                # Vec2 reads the previous task's snapshot while Vec1 updates
                # the recurrent statistics for the next task.
                task_max = T.alloc_shared((1, 64), "float32")
                task_sum = T.alloc_shared((1, 64), "float32")
                T.annotate_buffer_versions({task_max: (2, "iteration"), task_sum: (2, "iteration")})
            else:
                task_max = m
                task_sum = ell
            alpha = T.alloc_shared((64,), "float32")
            sink_ub = T.alloc_shared((64,), "float32")
            max_ub = T.alloc_shared((half,), "float32")
            lse_ub = T.alloc_shared((half,), "float32")
            inv_ub = T.alloc_shared((half,), "float32")
            T.annotate_layout({prob_ub: make_ascend_compact_nz_layout(prob_ub),
                               product: make_ascend_l0c_layout(product)})
            # QK's K loops and PV's N loop are disjoint owners of L0B.
            # Counter versioning alternates slots at every operand panel.
            T.annotate_buffer_versions({kv_l1: kv_versions, p_l1: p_versions,
                qa: (2, "counter"), kb: (2, "counter"),
                scores: 1, product_parts: 1, product: 1, nd0: 1, nd1: 1,
                valid: 1 if static_full else ((3, "iteration") if pipeline_queries else 1),
                score_ub: (2, "iteration") if pipeline_queries else 1, prob_ub: 1, product_ub: 1, accum: 1,
                out_ub: 1, m: 1, ell: 1, alpha: (2, "iteration") if pipeline_queries else 1,
                sink_ub: 1, max_ub: 1, lse_ub: 1, inv_ub: 1})
            # The manual Q ring is aggressive, but its row-based slot ids are
            # not a linear counter. In a full-length flat run with a 64-row
            # tail InsertSync cannot prove the cross-iteration ring, so that
            # combination uses compiler-managed Q synchronization instead.
            if not (pipeline_queries and full_length and width % block != 0):
                T.annotate_manual_multi_buffer(q_l1)
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                initial_sink = -T.infinity("float32") if kv_split > 1 else T.float32(0)
                S.vsts(sink_ub[0], S.vdup(initial_sink, "float32", full), full)
            if kv_split == 1:
                T.copy(Sink[sid * half:(sid + 1) * half], sink_ub[:half])
            # Cost-balanced contiguous query ownership: the wrapper supplies
            # each core's start/count and, for the task pipeline, a flat list
            # of its S2 tasks. A -1 header keeps the equal-count fallback.
            per = sq // grid
            extra = sq % grid
            fallback_start = bx * per + T.min(bx, extra)
            fallback_count = per + T.if_then_else(bx < extra, 1, 0)
            if pipeline_queries:
                # Flat S2 task stream. Full-length runs derive the row, tile,
                # query boundary and Q slot from the two-column core range, so
                # no per-tile table is uploaded or read. Variable-length runs
                # keep the descriptor table and reuse its prefix count. Manual
                # stages reproduce the baseline offsets: Vec0(i) -> LoadQK(i-1)
                # -> BMM1/Vec1(i-2) -> BMM2/Vec2(i-3). Direct KV combines
                # Vec0 and LoadQK in stage 1, within the three-slot L1 ring.
                if full_length:
                    start = T.if_then_else(Meta[bx, 0] >= 0, Meta[bx, 0], fallback_start)
                    count = T.if_then_else(Meta[bx, 1] >= 0, Meta[bx, 1], fallback_count)
                    start = T.if_then_else(count > 0, start, 0)
                    task_count = count * tiles_per_query
                else:
                    task_count = Meta[bx, 2]
                if task_count > 0:
                    if full_length:
                        first_row = start
                    else:
                        first_row = T.max(T.min(Meta[bx, 3], sq - 1), 0)
                    first_q = T.floormod(first_row * 2, q_slots)
                    first_q1 = T.floormod(first_row * 2 + 1, q_slots)
                    load_q_half(Q, q_l1, first_row, first_q, 0)
                    load_q_half(Q, q_l1, first_row, first_q1, 1)
                for task in T.Pipelined(task_count, num_stages=4,
                                        annotations={"enable_offset": True}):
                    if full_length:
                        row = T.min(start + task // tiles_per_query, sq - 1)
                        tile = task % tiles_per_query
                        last = tile + 1 == tiles_per_query
                    else:
                        base = 3 + task * TASK_STRIDE
                        row = T.max(T.min(Meta[bx, base], sq - 1), 0)
                        tile = Meta[bx, base + 1]
                        last = Meta[bx, base + 4] != 0
                    qslot = T.floormod(row * 2, q_slots)
                    qslot1 = T.floormod(row * 2 + 1, q_slots)
                    slot = T.floormod(task, kv_slots)
                    state = T.floormod(row, 2)
                    if full_length:
                        if width % block == 0:
                            n = block
                            tail64 = False
                        else:
                            tail64 = last
                            n = T.if_then_else(tail64, BLOCK_K, block)
                    else:
                        # Expose the valid prefix bound to address analysis
                        # for the second AIV's vector gather.
                        n = T.max(0, T.min(Meta[bx, base + 2], block))
                        tail64 = n <= 64
                    tail_rows = tail_rows_for(n)
                    if direct_kv_l1:
                        with T.Stage(1):
                            gather(Cache, Indices, KVWorkspace0, KVWorkspace1, bx, row, tile,
                                   slot, sid, nd0, nd1, valid, n, tail64, kv_l1)
                    else:
                        with T.Stage(0):
                            gather(Cache, Indices, KVWorkspace0, KVWorkspace1, bx, row, tile,
                                   slot, sid, nd0, nd1, valid, n, tail64)
                        with T.Stage(1):
                            load_kv(KVWorkspace0, KVWorkspace1, kv_l1, bx, slot, n, tail64)
                    with T.Stage(2):
                        if tail64:
                            qk_tail(q_l1, kv_l1, qa, kb, scores, score_ub, qslot,
                                    qslot1, n)
                        else:
                            qk(q_l1, kv_l1, qa, kb, scores, score_ub, qslot,
                               qslot1, block)
                        # The last QK read has drained both halves of this
                        # query. Issue the next query's two DMA copies after
                        # that read and before this task's PV FIXP.
                        if last:
                            if task + 1 < task_count:
                                if full_length:
                                    next_row = T.min(start + (task + 1) // tiles_per_query, sq - 1)
                                else:
                                    next_base = 3 + (task + 1) * TASK_STRIDE
                                    next_row = T.max(T.min(Meta[bx, next_base], sq - 1), 0)
                                next_q = T.floormod(next_row * 2, q_slots)
                                next_q1 = T.floormod(next_row * 2 + 1, q_slots)
                                load_q_half(Q, q_l1, next_row, next_q, 0)
                                load_q_half(Q, q_l1, next_row, next_q1, 1)
                    with T.Stage(2):
                        softmax(tile, score_ub, valid, prob_ub, p_l1, m, ell, alpha,
                                sink_ub, tail64, tail_rows, state, task_max, task_sum)
                    with T.Stage(3):
                        if tail64:
                            pv_tail(kv_l1, p_l1, pa, vb, product_parts, product,
                                    product_ub, n)
                        else:
                            pv(kv_l1, p_l1, pa, vb, product_parts, product,
                               product_ub, block)
                    with T.Stage(3):
                        update_output(tile, product_ub, accum, alpha)
                        if last:
                            finish(row, O, MaxLogits, LSE, accum, out_ub, task_max, task_sum,
                                   sink_ub, max_ub, lse_ub, inv_ub, sid, 0)
            else:
                # Startup load for the first query, then rotate the half slots
                # (0,1) -> (2,0) -> (1,2) by advancing two slots per query.
                start = T.if_then_else(Meta[bx, 0] >= 0, Meta[bx, 0], fallback_start)
                count = T.if_then_else(Meta[bx, 1] >= 0, Meta[bx, 1], fallback_count)
                start = T.if_then_else(count > 0, start, 0)
                load_q_half(Q, q_l1, start, 0, 0)
                load_q_half(Q, q_l1, start, 1, 1)
                workspace_base = T.alloc_var("int32", init=0)
                for query in T.serial(count):
                    row = start + query
                    q_base = (query * 2) % q_slots
                    slot0 = q_base
                    slot1 = (q_base + 1) % q_slots
                    nslot0 = (q_base + 2) % q_slots
                    nslot1 = q_base
                    # Full lengths use the static tile count; varying lengths
                    # run only the tiles covered by the prefix, with one
                    # masked tile for an empty Query.
                    if full_length:
                        tiles_eff = tiles_per_query
                    else:
                        slice_length = T.max(T.min(Lengths[row // kv_split] -
                                                    (row % kv_split) * width, width), 0)
                        tiles_eff = T.max(T.ceildiv(slice_length, block), 1)
                    for tile in T.Pipelined(tiles_eff, num_stages=stages):
                        slot = (workspace_base + tile) % kv_slots
                        if full_length:
                            if width % block == 0:
                                n = block
                                tail64 = False
                            else:
                                tail64 = tile + 1 == tiles_per_query
                                n = T.if_then_else(tail64, BLOCK_K, block)
                        else:
                            n = T.max(T.min(Lengths[row // kv_split] -
                                            (row % kv_split) * width - tile * block, block), 0)
                            tail64 = n <= 64
                        tail_rows = tail_rows_for(n)
                        gather(Cache, Indices, KVWorkspace0, KVWorkspace1, bx, row, tile,
                               slot, sid, nd0, nd1, valid, n, tail64)
                        load_kv(KVWorkspace0, KVWorkspace1, kv_l1, bx, slot, n, tail64)
                        if tail64:
                            qk_tail(q_l1, kv_l1, qa, kb, scores, score_ub, slot0, slot1, n)
                        else:
                            qk(q_l1, kv_l1, qa, kb, scores, score_ub, slot0, slot1, block)
                        softmax(tile, score_ub, valid, prob_ub, p_l1, m, ell, alpha,
                                sink_ub, tail64, tail_rows, 0, task_max, task_sum)
                        if tail64:
                            pv_tail(kv_l1, p_l1, pa, vb, product_parts, product, product_ub, n)
                        else:
                            pv(kv_l1, p_l1, pa, vb, product_parts, product, product_ub, block)
                        update_output(tile, product_ub, accum, alpha)
                    workspace_base = (workspace_base + tiles_eff) % kv_slots
                    # The last query needs no additional GM loads.
                    if query + 1 < count:
                        load_q_half(Q, q_l1, row + 1, nslot0, 0)
                        load_q_half(Q, q_l1, row + 1, nslot1, 1)
                    finish(row, O, MaxLogits, LSE, accum, out_ub, m, ell,
                           sink_ub, max_ub, lse_ub, inv_ub, sid, 0)
    return main


def build_prefill_kernel(sq, skv, width, sm_scale=SM_SCALE, cores=None, *, pipeline_stages=2, pair_dma=False,
                         prefetch_kv=True, pipeline_queries=False, full_valid_kv=False, fixed_offset=None,
                         full_length=False, workspace_l2_cache="NORMAL_FV", unroll_pairs=False,
                         kv_split=1):
    """Build the prefill PrimFunc with an internal KV workspace.

    Sq and Skv must be positive. Every positive multiple of 64 for T uses the
    same 128-entry tiling and FP32 online accumulation, with a 64-entry tail
    when needed. Variable-length tails shrink the Cube N to a 16-row bucket of
    the real prefix (64 stays the cap). The kernel supports lengths, invalid
    indices and sinks.
    pipeline_stages accepts 1 or 2. Scheduling flags are optimization hints;
    they do not change numerical semantics.
    pipeline_queries=True scans one flat S2 task stream per core (tasks may
    cross query boundaries) with Vec0(i), LoadQK(i-1), BMM1/Vec1(i-2) and
    BMM2/Vec2(i-3) stages; False keeps the per-query schedule for A/B.
    full_length=True requires every length to equal T and lets the kernel
    derive the row, tile and query boundary from each core's range instead of
    a per-tile table. pair_dma merges gathers for adjacent KV rows; False
    issues one DMA per valid row. unroll_pairs additionally unrolls the 8-pair
    loop of statically full groups so the first slot store is not blocked by the
    second slot's gather; it only takes effect with pair_dma=True.
    workspace_l2_cache selects the L2 policy for AIV workspace stores.
    NORMAL_FV keeps them resident for the Cube read, and NOTALLOC_CI reproduces
    the previous invalidate-on-write behavior.
    Vec0 splits each tile by real prefix: AIV0 gathers ceil(N/2) rows, AIV1 the
    rest. The full-block task stream combines the two halves in one GM slot
    for a single Cube import. Other schedules use separate GM half slots.
    Vec0 zeroes padding and the softmax mask follows the L1 row order.
    full_valid_kv additionally requires every index in [0, Skv); together with
    full_length it skips index checks and uses the fixed 128- or 64-row tile
    geometry for gather and softmax. Offline callers must validate
    these conditions before launch. fixed_offset=C uses exp(score-C);
    callers must ensure representable probabilities and FP32 sums.
    """
    if min(sq, skv) <= 0 or width <= 0 or width % BLOCK_K:
        raise ValueError("Positive dimensions/cores and T divisible by 64 are required")
    cores = _core_count() if cores is None else cores
    if cores <= 0:
        raise ValueError("Positive dimensions/cores and T divisible by 64 are required")
    if pipeline_stages not in (1, 2):
        raise ValueError("Attention supports one or two pipeline stages")
    if not isinstance(kv_split, int) or kv_split < 1 or width % kv_split or (width // kv_split) % BLOCK_K:
        raise ValueError("KV split must divide T into positive multiples of 64")
    scale = _check_scale(sm_scale)
    fixed_offset = _check_fixed_offset(fixed_offset)
    prefetch_kv = bool(prefetch_kv and pipeline_stages == 2)
    options = dict(pipeline_stages=pipeline_stages, prefetch_kv=prefetch_kv,
                   pipeline_queries=bool(pipeline_queries and prefetch_kv), full_valid_kv=bool(full_valid_kv),
                   fixed_offset=fixed_offset)
    return _build_multi_tile(sq * kv_split, skv, width // kv_split, scale, cores, pair_dma=pair_dma,
                             full_length=bool(full_length), workspace_l2_cache=workspace_l2_cache,
                             unroll_pairs=bool(unroll_pairs), input_sq=sq, kv_split=kv_split, **options)


class _WorkspaceKernel:
    """Reuse the internal GM workspace across launches on the same stream."""

    def __init__(self, kernel, shape, *, task_pipeline=False, width=None, full_length=False, packed_kv=False,
                 direct_kv_l1=False, kv_split=1):
        self._kernel = kernel
        self._shape = tuple(shape)
        if direct_kv_l1:
            self._workspace_shapes = ((1,), (1,))
        elif packed_kv:
            grid, slots, half_rows, dim = self._shape
            # Retain the argument positions for output tensors.
            self._workspace_shapes = ((grid, slots, 2 * half_rows, dim), (1,))
        else:
            self._workspace_shapes = (self._shape, self._shape)
        self._task_pipeline = bool(task_pipeline)
        self._width = width
        self._full_length = bool(full_length)
        self._kv_split = kv_split
        self._workspace = None
        self._fallback_meta = None

    def __call__(self, q, kv, indices, lengths, unused_kv, unused_indices, zeros, sink, meta=None):
        stream = torch.npu.current_stream(q.device)
        key = (q.device, stream.npu_stream)
        if meta is None:
            if self._task_pipeline and not self._full_length:
                # Variable-length streaming needs the descriptor table.
                host = lengths.detach().cpu() if lengths is not None else None
                meta = self.prepare_metadata(host, q.shape[0] // H).to(q.device)
            else:
                # Full-length streaming and the per-query fallback both
                # derive the equal-count split on device.
                cached = self._fallback_meta
                if cached is None or cached[0] != key:
                    fallback = torch.full((self._shape[0], 2), -1, dtype=torch.int32, device=q.device)
                    self._fallback_meta = (key, fallback)
                else:
                    fallback = cached[1]
                meta = fallback
        cached = self._workspace
        if cached is None or cached[0] != key:
            # Launches on different streams may overlap. Allocate the replacement
            # on the launch stream so the allocator tracks outstanding uses there.
            workspace = tuple(torch.empty(shape, dtype=torch.bfloat16, device=q.device)
                              for shape in self._workspace_shapes)
            self._workspace = (key, workspace)
        else:
            workspace = cached[1]
        return self._kernel(q, kv, indices, lengths, unused_kv, unused_indices,
                            zeros, sink, meta, *workspace)

    def get_kernel_source(self):
        return self._kernel.get_kernel_source()

    def prepare_benchmark_metadata(self, inputs):
        return self.prepare_metadata(inputs.get("topk_length"), inputs["indices"].shape[0])

    def prepare_metadata(self, host_lengths, rows):
        if self._full_length:
            lengths = None
        elif host_lengths is not None:
            lengths = host_lengths.detach().cpu().reshape(rows, 1)
            offsets = torch.arange(self._kv_split, dtype=torch.int32).reshape(1, -1) * self._width
            lengths = (lengths - offsets).clamp(0, self._width).reshape(-1)
        else:
            lengths = None
        return _cost_meta(lengths, rows * self._kv_split, self._shape[0], self._full_length,
                          self._width, task_pipeline=self._task_pipeline)


def _kv_split_factor(sq, width, cores, requested=None, full_length=True, max_topk_length=None):
    """Split the KV dimension across idle cores when Sq cannot fill them.

    Partition the common tile stream only when the active prefix has enough
    work. Variable lengths without a validated host bound stay unsplit.
    """
    import os
    if requested is None:
        env = os.environ.get("MLA_KV_SPLIT", "auto").strip()
        if env.lower() in ("off", "0", "none", "disabled"):
            return 1
        if env.lower() != "auto" and env:
            requested = int(env)
    if not full_length and max_topk_length is None:
        return 1
    active = width if full_length else max(0, min(width, int(max_topk_length)))
    if sq >= cores or active < 8 * KV_TILE_ROWS:
        # Conservative launch/merge amortization threshold; hardware tuning
        # may refine it without changing the arithmetic or caller contract.
        return 1
    limit = min(cores // max(sq, 1), width // KV_TILE_ROWS)
    split = 1
    for cand in (32, 16, 8, 4, 2):
        if cand > limit:
            continue
        if width % cand:
            continue
        if (width // cand) % 64:
            continue
        active_parts = (active + width // cand - 1) // (width // cand)
        if active_parts < 2 or 2 * active_parts < cand:
            continue
        split = cand
        break
    if requested is not None:
        requested = int(requested)
        if (requested > 1 and width % requested == 0
                and (width // requested) % 64 == 0 and sq * requested <= cores):
            split = requested
        elif requested <= 1:
            split = 1
    return split


class _SplitKernel:
    """One address-mapped attention launch followed by one device merge."""

    def __init__(self, inner, merge, sq, split, width):
        self._inner, self._merge = inner, merge
        self._sq, self._split, self._width = sq, split, width

    def get_kernel_source(self):
        return self._inner.get_kernel_source() + "\n" + self._merge.get_kernel_source()

    def prepare_metadata(self, host_lengths, rows):
        return self._inner.prepare_metadata(host_lengths, rows)

    def prepare_benchmark_metadata(self, inputs):
        return self._inner.prepare_benchmark_metadata(inputs)

    def __call__(self, q, kv, indices, lengths, unused_kv, unused_indices, zeros, sink, meta=None):
        partial, maximum, lse = self._inner(q, kv, indices, lengths, unused_kv, unused_indices,
                                          zeros, sink, meta)
        return self._merge(partial, maximum, lse, sink)


def build_kv_merge_kernel(sq, split, cores):
    """Fuse stable split reduction, a single sink term, and BF16 output."""
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    rows = sq * H
    grid = min(rows, 2 * cores)

    @T.prim_func
    def main(Partial: T.Tensor((sq * split * H, D), "float32"),
             PartialMax: T.Tensor((sq * split * H,), "float32"),
             PartialLSE: T.Tensor((sq * split * H,), "float32"),
             Sink: T.Tensor((H,), "float32"),
             O: T.Tensor((rows, D), "bfloat16"),
             MaxLogits: T.Tensor((rows,), "float32"),
             LSE: T.Tensor((rows,), "float32")):
        with T.Kernel(grid) as bx:
            values = T.alloc_shared((split, D), "float32")
            maxima = T.alloc_shared((split, H), "float32")
            logs = T.alloc_shared((split, H), "float32")
            weights = T.alloc_shared((split, 64), "float32")
            sink_ub = T.alloc_shared((64,), "float32")
            out = T.alloc_shared((D,), "bfloat16")
            max_ub = T.alloc_shared((64,), "float32")
            lse_ub = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({values: 1, maxima: 1, logs: 1, weights: 1,
                                       sink_ub: 1, out: 1, max_ub: 1, lse_ub: 1})
            for step in T.serial(T.ceildiv(rows, grid)):
                row = step * grid + bx
                if row < rows:
                    query, head = row // H, row % H
                    T.copy(Sink[head:head + 1], sink_ub[:1])
                    for piece in T.serial(split):
                        base = (query * split + piece) * H
                        T.copy(Partial[base + head, :], values[piece, :])
                        T.copy(PartialMax[base:base + H], maxima[piece, :])
                        T.copy(PartialLSE[base:base + H], logs[piece, :])
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        zero = S.vdup(T.float32(0), "float32", full)
                        one = S.vdup(T.float32(1), "float32", full)
                        neg_inf = S.vdup(-T.infinity("float32"), "float32", full)
                        inf = S.vdup(T.infinity("float32"), "float32", full)
                        best = T.alloc_var("float32x64", init=neg_inf)
                        anchor = T.alloc_var("float32x64", init=neg_inf)
                        for piece in T.serial(split):
                            maximum = S.vld(maxima[piece, head], "BRC_B32")
                            valid = S.vcmps(maximum, -T.infinity("float32"), full, "ne")
                            best = S.vmax(best, maximum, full)
                            logz = S.vsel(S.vld(logs[piece, head], "BRC_B32"), neg_inf, valid)
                            anchor = S.vmax(anchor, logz, full)
                        nonempty = S.vcmps(anchor, -T.infinity("float32"), full, "ne")
                        safe_anchor = S.vsel(anchor, zero, nonempty)
                        total = T.alloc_var("float32x64", init=zero)
                        for piece in T.serial(split):
                            maximum = S.vld(maxima[piece, head], "BRC_B32")
                            valid = S.vcmps(maximum, -T.infinity("float32"), full, "ne")
                            logz = S.vsel(S.vld(logs[piece, head], "BRC_B32"), neg_inf, valid)
                            weight = S.vsel(S.vexp(S.vsub(logz, safe_anchor, full), full), zero, valid)
                            S.vsts(weights[piece, 0], weight, full)
                            total = S.vadd(total, weight, full)
                        safe_total = S.vsel(total, one, nonempty)
                        kv_lse = S.vadd(safe_anchor, S.vln(safe_total, full), full)
                        sink = S.vld(sink_ub[0], "BRC_B32")
                        sink_inf = S.vcmps(sink, T.infinity("float32"), full, "eq")
                        origin = S.vmax(S.vsel(kv_lse, neg_inf, nonempty), sink, full)
                        degenerate = S.por(sink_inf, S.vcmps(origin, -T.infinity("float32"), full, "eq"), full)
                        safe_origin = S.vsel(zero, origin, degenerate)
                        active = S.pand(nonempty, S.vcmps(sink, T.infinity("float32"), full, "ne"), full)
                        kv_scale = S.vsel(S.vexp(S.vsub(safe_anchor, safe_origin, full), full), zero, active)
                        sink_weight = S.vsel(one, S.vexp(S.vsub(sink, safe_origin, full), full), sink_inf)
                        denominator = S.vadd(S.vmul(total, kv_scale, full), sink_weight, full)
                        safe_denominator = S.vsel(denominator, one, S.vcmps(denominator, T.float32(0), full, "gt"))
                        factor = S.vsel(S.vdiv(kv_scale, safe_denominator, full, precision="exact"), zero, active)
                        S.vsts(max_ub[0], best, full)
                        S.vsts(lse_ub[0], S.vsel(kv_lse, inf, nonempty), full)
                        S.mem_bar("VST_VLD")
                        for chunk in T.serial(D // 64):
                            result = T.alloc_var("float32x64", init=zero)
                            for piece in T.serial(split):
                                coefficient = S.vmul(S.vld(weights[piece, 0]), factor, full)
                                value = S.vld(values[piece, chunk * 64])
                                result = S.vadd(result, S.vmul(value, coefficient, full), full)
                            packed = S.vcvt(result, "bfloat16", full, sat=False, part=0)
                            S.vsts(out[chunk * 64], packed, full, "PK_B32")
                    T.copy(out, O[row, :])
                    T.copy(max_ub[:1], MaxLogits[row:row + 1])
                    T.copy(lse_ub[:1], LSE[row:row + 1])
    return main


def _resolve_query_pipeline(sq, width, cores, prefetch_kv, requested):
    if not prefetch_kv:
        return False
    return (width > KV_TILE_ROWS or sq > cores) if requested is None else bool(requested)


def _compile_prefill(sq, skv, width, scale, cores, *, pair_dma=False, prefetch_kv=True,
                     pipeline_queries=None, full_valid_kv=False, fixed_offset=None,
                     full_length=False, workspace_l2_cache="NORMAL_FV", unroll_pairs=False,
                     kv_split=None, max_topk_length=None):
    # Resolve environment overrides before the cache lookup so a stale cached
    # kernel never hides a requested change of partitioning.
    split = _kv_split_factor(sq, width, cores, kv_split, full_length, max_topk_length)
    stream = _resolve_query_pipeline(sq, width, cores, prefetch_kv, pipeline_queries)
    return _compile_prefill_cached(sq, skv, width, scale, cores, bool(pair_dma), bool(prefetch_kv),
                                   stream, bool(full_valid_kv), fixed_offset, bool(full_length),
                                   workspace_l2_cache, bool(unroll_pairs), split)


@lru_cache(maxsize=32)
def _compile_prefill_cached(sq, skv, width, scale, cores, pair_dma, prefetch_kv, pipeline_queries,
                            full_valid_kv, fixed_offset, full_length, workspace_l2_cache,
                            unroll_pairs, split):
    import tilelang
    from tvm.target import Target
    with Target("ascend") as target:
        kernel = tilelang.compile(
            build_prefill_kernel(sq, skv, width, scale, cores, pair_dma=pair_dma,
                                 prefetch_kv=prefetch_kv, pipeline_queries=pipeline_queries,
                                 full_valid_kv=full_valid_kv, fixed_offset=fixed_offset,
                                 full_length=full_length, workspace_l2_cache=workspace_l2_cache,
                                 unroll_pairs=unroll_pairs, kv_split=split),
            target=target, execution_backend="tvm_ffi", out_idx=[11, 12, 13])
        grid = min(sq * split, cores)
        stream = pipeline_queries and prefetch_kv
        inner = _WorkspaceKernel(kernel, (grid, KV_WORKSPACE_SLOTS, KV_TILE_ROWS // 2, D),
                                 task_pipeline=stream, width=width // split, full_length=full_length,
                                 packed_kv=_packed_kv_workspace(width // split, full_length, stream, full_valid_kv),
                                 direct_kv_l1=_use_direct_kv_l1(width // split, full_length, stream,
                                                                full_valid_kv, pair_dma, split),
                                 kv_split=split)
        if split == 1:
            return inner
        merge = tilelang.compile(build_kv_merge_kernel(sq, split, cores), target=target,
                                 execution_backend="tvm_ffi", out_idx=[4, 5, 6])
        return _SplitKernel(inner, merge, sq, split, width)


def flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512,
                         attn_sink=None, topk_length=None, *, validate_values=True, pipeline_queries=None,
                         full_valid_kv=False, fixed_offset=None):
    """[Sq,64,512] BF16 -> (O BF16, max_logits FP32, lse FP32).

    indices has shape [Sq,1,T], where T can be any positive multiple of 64.
    Invalid indices (<0 or >=Skv) are masked. Empty rows return O=0,
    max_logits=-inf, lse=+inf, matching the checked FlashMLA implementation.
    validate_values=False skips host synchronization for length-range checks;
    the caller must then guarantee 0 <= topk_length <= T.
    pipeline_queries=None enables the shared task stream for T>128 or Sq
    above the core count. Explicit lengths require validation for streaming.
    The same policy is used by the benchmark; False keeps the per-Query path.
    full_valid_kv=True requires full lengths and validates all KV indices on
    the host. The kernel uses the fixed tile geometry to skip per-index checks
    and masks.
    Full-length launches, either implicit (no topk_length) or explicit and
    validated equal to T, derive every core's row, tile and query boundary
    arithmetically and upload only the start/count range.
    For varying lengths, launches balance each core's query range by the S2
    tile cost; validate_values=False keeps the equal-count fallback.
    fixed_offset=C explicitly uses probabilities exp(score-C).
    The caller guarantees a safe numerical range, including BF16 probabilities
    and FP32 denominator/PV sums; validate_values checks lengths only. None
    uses stable online softmax. Returned max_logits always contains actual
    scaled score maxima and LSE excludes the sink in both modes.
    """
    for name, value in (("q", q), ("kv", kv), ("indices", indices)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if q.ndim != 3 or tuple(q.shape[1:]) != (H, D) or q.shape[0] <= 0:
        raise ValueError("q must have shape [Sq,64,512], Sq>0")
    _check_tensor("q", q, q.shape, torch.bfloat16, q.device)
    if kv.ndim != 3 or tuple(kv.shape[1:]) != (1, D) or kv.shape[0] <= 0:
        raise ValueError("kv must have shape [Skv,1,512], Skv>0")
    _check_tensor("kv", kv, kv.shape, torch.bfloat16, q.device)
    if indices.ndim != 3:
        raise ValueError("indices must have rank 3")
    sq, skv, width = q.shape[0], kv.shape[0], indices.shape[-1]
    host_lengths = _check_indices_lengths(indices, topk_length, sq, width, q.device, validate_values)
    if d_v != D:
        raise ValueError("d_v must be 512")
    scale = _check_scale(sm_scale)
    fixed_offset = _check_fixed_offset(fixed_offset)
    if full_valid_kv:
        host_indices = indices.detach().cpu()
        if bool(((host_indices < 0) | (host_indices >= skv)).any()):
            raise ValueError("full_valid_kv requires every index in [0, Skv)")
        if topk_length is not None:
            # Reuse the range-check copy when it exists; this validation
            # already needs host lengths, so it adds no extra synchronization.
            checked = host_lengths if host_lengths is not None else topk_length.detach().cpu()
            if not bool((checked == width).all()):
                raise ValueError("full_valid_kv requires every topk_length to equal T")
            host_lengths = checked
    # Implicit lengths are constant T; explicit ones reuse the host copy above,
    # so selecting the static schedule adds no synchronization.
    full_length = topk_length is None or (host_lengths is not None and bool((host_lengths == width).all()))
    if attn_sink is not None:
        _check_tensor("attn_sink", attn_sink, (H,), torch.float32, q.device)
    _require_npu(q)
    lengths = (_constant_on_device((sq,), width, torch.int32, q.device)
               if topk_length is None else topk_length.contiguous())
    sink = (_constant_on_device((H,), -math.inf, torch.float32, q.device)
            if attn_sink is None else attn_sink.contiguous())
    # The reserved ExtraIndices/ExtraLengths arguments are never read by the
    # shared main kernel; reuse a correctly shaped INT32 tensor as their ABI
    # placeholder instead of allocating and transferring zeros per call.
    zeros = lengths
    kv_flat = kv.contiguous().view(skv, D)
    idx = indices.contiguous().view(sq, width)
    with torch.npu.device(q.device):
        cores = _core_count(q)
        stream_queries = _resolve_query_pipeline(sq, width, cores, True, pipeline_queries)
        if stream_queries and host_lengths is None and topk_length is not None:
            # The flat task table needs host-visible lengths; keep the
            # per-query schedule when validate_values=False skipped the read.
            stream_queries = False
        options = {"pipeline_queries": stream_queries}
        if full_length:
            # One start/count record per core replaces the per-tile table.
            options["full_length"] = True
        if full_valid_kv:
            options["full_valid_kv"] = True
        if fixed_offset is not None:
            options["fixed_offset"] = fixed_offset
        if host_lengths is not None:
            options["max_topk_length"] = int(host_lengths.max())
        kernel = _compile_prefill(sq, skv, width, scale, cores, **options)
        flat_q = q.contiguous().view(sq * H, D)
        # Full prefixes use the kernel's cached equal-range header. Varying
        # prefixes use the same virtual-query metadata builder as benchmarks.
        meta = None if full_length else kernel.prepare_metadata(host_lengths, sq).to(q.device)
        o, max_logits, lse = kernel(flat_q, kv_flat, idx, lengths, kv_flat[:1],
                                    zeros.view(sq, 1), zeros, sink, meta)
    return o.view(sq, H, D), max_logits.view(sq, H), lse.view(sq, H)
