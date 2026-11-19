"""Kernels shared by the P1 analysis-report tests."""

import tilelang.ascend.language as T


def no_vf_kernel():
    """A PrimFunc with no ``SIMD_VF`` block, shared by the ``no SIMD_VF`` report tests."""

    @T.prim_func
    def main(A: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((16,), "float32")
            T.copy(A, a_ub)

    return main
