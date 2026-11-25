"""Compilation entry point for the final QSA schedule."""

import tilelang

from examples.ascend.qsa.kernel import qsa_fwd, SUPPORTED_AIC_COUNTS


def compile_qsa(num_tokens, kv_length, *, dynamic_shape=False, aic_count=32):
    """Compile one static or dynamic QSA kernel for the fixed Qwen contract."""
    assert aic_count in SUPPORTED_AIC_COUNTS, "aic_count must be 28, 32, or 36"
    program = qsa_fwd(
        num_tokens,
        kv_length,
        aic_count=aic_count,
        dynamic_shape=dynamic_shape,
    )
    return tilelang.compile(
        program,
        target="ascend",
        out_idx=-1,
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
        },
    )
