# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Q4Kernels contributors.
"""Keep benchmark results on the console and full native/Python output in a log."""

import ctypes
import os
import sys
import tempfile
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout


def open_log(destination, stage):
    if destination is None:
        return tempfile.NamedTemporaryFile(mode="w", buffering=1, prefix=f"tilelang_mhc_{stage}_", suffix=".log", delete=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.open("x", buffering=1)


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()
    # CANN and TVM also write through libc, independently of Python streams.
    fflush = ctypes.CDLL(None).fflush
    fflush.argtypes = [ctypes.c_void_p]
    fflush.restype = ctypes.c_int
    fflush(None)


@contextmanager
def capture_output(log):
    """Redirect both Python streams and OS descriptors, including child compilers."""
    _flush()
    saved = [os.dup(1), os.dup(2)]
    try:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        with redirect_stdout(log), redirect_stderr(log):
            try:
                yield
            except BaseException:
                traceback.print_exc(file=log)
                raise
            finally:
                _flush()
    finally:
        for destination, original in enumerate(saved, 1):
            os.dup2(original, destination)
            os.close(original)
