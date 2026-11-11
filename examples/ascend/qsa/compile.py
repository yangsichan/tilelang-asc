"""Compilation entry point for the static and dynamic-shape QSA kernels."""

import tilelang

from examples.ascend.qsa.kernel import qsa_fwd


def compile_qsa(num_tokens, kv_length, top_k=2176, *, dynamic_shape=False):
    return tilelang.compile(
        qsa_fwd(num_tokens, kv_length, top_k, dynamic_shape=dynamic_shape),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
