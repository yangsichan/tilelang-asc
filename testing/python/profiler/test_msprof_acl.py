"""CPU regression checks for CANN capture ownership and the requested features."""

import struct
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tilelang.profiler import _acl


@pytest.fixture
def library(monkeypatch):
    library = Mock()
    library.aclprofCreateConfig.return_value = 123
    for name in ("Init", "Start", "Stop", "DestroyConfig", "Finalize"):
        getattr(library, f"aclprof{name}").return_value = 0
    monkeypatch.setattr(_acl, "_load_library", lambda: library)
    return library


def test_capture_uses_only_task_timing_and_pipe_counters(library, tmp_path):
    sync = Mock()
    with _acl.profile(tmp_path, 3, sync):
        library.aclprofStart.assert_called_once_with(123)
        library.aclprofStop.assert_not_called()
    devices, count, metric, events, mask = library.aclprofCreateConfig.call_args.args
    assert list(devices) == [3]
    assert (count, metric, events, mask) == (1, 1, None, 0x7)
    assert mask & 0x800000000 == 0  # No AICORE_SHAPE request.
    assert [call[0] for call in library.method_calls] == [
        "aclprofInit",
        "aclprofCreateConfig",
        "aclprofStart",
        "aclprofStop",
        "aclprofDestroyConfig",
        "aclprofFinalize",
    ]
    assert sync.call_count == 2


@pytest.mark.parametrize("failure", ["Init", "CreateConfig", "Start", "body", "synchronize", "Stop", "DestroyConfig"])
def test_failures_release_acquired_resources(library, tmp_path, failure):
    sync = Mock()
    if failure == "CreateConfig":
        library.aclprofCreateConfig.return_value = None
    elif failure == "synchronize":
        sync.side_effect = [None, RuntimeError("device error")]
    elif failure != "body":
        getattr(library, f"aclprof{failure}").return_value = 42
    with pytest.raises((RuntimeError, ValueError)), _acl.profile(tmp_path, 0, sync):
        if failure == "body":
            raise ValueError("kernel failure")
    if failure == "Init":
        library.aclprofFinalize.assert_not_called()
    else:
        library.aclprofFinalize.assert_called_once()
    if failure not in ("Init", "CreateConfig"):
        library.aclprofDestroyConfig.assert_called_once_with(123)
    if failure not in ("Init", "CreateConfig", "Start"):
        library.aclprofStop.assert_called_once_with(123)


def test_ctypes_signatures_keep_config_pointers_and_mask_width(monkeypatch):
    library = Mock()
    loader = Mock(return_value=library)
    monkeypatch.setattr(_acl.ctypes, "CDLL", loader)
    # Call the undecorated loader so this test cannot populate the runtime cache.
    assert _acl._load_library.__wrapped__() is library
    assert library.aclprofCreateConfig.restype is _acl.ctypes.c_void_p
    assert library.aclprofCreateConfig.argtypes[-1] is _acl.ctypes.c_uint64
    assert library.aclprofInit.argtypes[1] is _acl.ctypes.c_size_t


@pytest.mark.parametrize("mixed", [True, False], ids=["mixed-aic-time", "vector-aiv-time"])
def test_msprof_keeps_ffts_units_engine_selection_and_flush_filter(monkeypatch, mixed):
    from tilelang.profiler import msprof

    flush, run = Mock(), Mock()
    npu = SimpleNamespace(current_device=lambda: 2, synchronize=Mock())
    monkeypatch.setattr(msprof.torch, "npu", npu, raising=False)
    monkeypatch.setattr(msprof, "get_msprof_cache_flush_kernel", lambda size: flush)

    @contextmanager
    def capture(directory, device, synchronize):
        assert device == 2
        yield
        root = Path(directory)
        (root / "hash_dic.slice_0").write_text("1:_tilelang_profiler_cache_flush_kernel\n2:operator\n")
        tracks = []
        for sequence in (1, 2):
            track = [0] * 8
            track[3], track[5] = sequence << 32, sequence
            tracks.append(struct.pack("<8q", *track))
        (root / "task_track.slice_0").write_bytes(b"".join(tracks))
        records = []
        for _ in range(3):
            events = [(1, True, 100), (2, True, 1200)]
            if mixed:
                events.append((2, False, 1000))
            for sequence, vector, duration in events:
                event = [0] * 16
                event[0], event[1], event[2] = sequence << 32, 100, (1 << 31) if vector else 0
                event[4], event[5], event[14], event[15] = 30, 40, 10000, 10000 + duration
                records.append(struct.pack("<16q", *event))
        (root / "ffts_profile.data").write_bytes(b"".join(records))

    monkeypatch.setattr(_acl, "profile", capture)
    result = msprof.bench_with_msprof(run, SimpleNamespace(numel=lambda: 1), 3, detailed=True)
    assert result.dur_us == (1.0 if mixed else 1.2)
    assert result.aic_total_cycles == (100 if mixed else 0)
    assert result.aiv_total_cycles == 100
    assert run.call_count == 3
    assert flush.call_count == 4  # One warm launch outside capture.
