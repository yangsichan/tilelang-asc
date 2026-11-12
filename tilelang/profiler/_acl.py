"""Minimal CANN capture for the FFTS reader, without optional feature negotiation."""

import ctypes
import functools
import os
from contextlib import contextmanager


@functools.lru_cache(maxsize=1)
def _load_library():
    try:
        library = ctypes.CDLL("libmsprofiler.so", mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise RuntimeError("Cannot load CANN libmsprofiler.so; source CANN's set_env.sh before profiling") from exc
    signatures = {
        "aclprofInit": ([ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int),
        "aclprofCreateConfig": (
            [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64],
            ctypes.c_void_p,
        ),
        "aclprofStart": ([ctypes.c_void_p], ctypes.c_int),
        "aclprofStop": ([ctypes.c_void_p], ctypes.c_int),
        "aclprofDestroyConfig": ([ctypes.c_void_p], ctypes.c_int),
        "aclprofFinalize": ([], ctypes.c_int),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result
    return library


def _check(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} failed with CANN status {status}")


@contextmanager
def profile(directory, device_index, synchronize):
    """Capture task names, durations and pipe counters; release resources on failure.

    The reader needs ACL API tracking, TASK_TIME and AICORE_METRICS (mask 0x7),
    with PIPE_UTILIZATION (1). It does not need AICORE_SHAPE, OP_ATTR or the
    torch_npu feature negotiation that warns about newer CANN feature names.
    Captures use CANN's process-global profiling session and must run serially.
    """
    library = _load_library()
    initialized = started = False
    config = None
    try:
        synchronize()
        encoded = os.fsencode(directory)
        _check(library.aclprofInit(encoded, len(encoded)), "aclprofInit")
        initialized = True
        devices = (ctypes.c_uint32 * 1)(device_index)
        config = library.aclprofCreateConfig(devices, 1, 1, None, 0x0001 | 0x0002 | 0x0004)
        if not config:
            raise RuntimeError("aclprofCreateConfig returned null")
        _check(library.aclprofStart(config), "aclprofStart")
        started = True
        yield
    finally:
        operations = []
        if started:
            operations += [("synchronize", synchronize), ("aclprofStop", lambda: library.aclprofStop(config))]
        if config:
            operations.append(("aclprofDestroyConfig", lambda: library.aclprofDestroyConfig(config)))
        if initialized:
            operations.append(("aclprofFinalize", library.aclprofFinalize))
        errors = []
        for name, operation in operations:
            try:
                result = operation()
                if name != "synchronize":
                    _check(result, name)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError(f"CANN profiler cleanup failed: {'; '.join(errors)}")
