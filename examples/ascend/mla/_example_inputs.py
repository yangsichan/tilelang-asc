"""Build optional input tensors for the two MLA examples."""

import torch


def make_sink(seed=88888888, *, enabled=True, weights=None):
    """Use supplied FP32 model weights or generate a synthetic per-head sink."""
    if weights is not None:
        if not isinstance(weights, torch.Tensor) or weights.shape != (64,) or weights.dtype != torch.float32:
            raise ValueError("attn_sink must be a float32 tensor of shape [64]")
        return weights.detach().cpu()
    if not enabled:
        return None
    generator = torch.Generator().manual_seed(seed + 1)
    return torch.randn(64, generator=generator, dtype=torch.float32, device="cpu") * 0.5


def load_sink(path):
    if path is None:
        return None
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(weights, torch.Tensor):
        raise ValueError("--sink-file must contain a tensor saved with torch.save")
    return make_sink(weights=weights)


def make_topk_length(values, rows, width, name="topk_length"):
    """Accept one length for all rows, one per row, or a contract-shaped tensor."""
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.dtype != torch.int32 or values.shape != (rows,):
            raise ValueError(f"{name} must be an int32 tensor of shape [{rows}]")
        cpu_lengths = values.detach().cpu()
    else:
        values = (values,) if isinstance(values, int) else tuple(values)
        if len(values) not in (1, rows) or any(type(value) is not int for value in values):
            raise ValueError(f"{name} requires one integer or {rows} integers")
        if any(not 0 <= value <= width for value in values):
            raise ValueError(f"{name} must be in [0, {width}]")
        return torch.tensor(values * rows if len(values) == 1 else values, dtype=torch.int32, device="cpu")
    if bool(((cpu_lengths < 0) | (cpu_lengths > width)).any()):
        raise ValueError(f"{name} must be in [0, {width}]")
    return cpu_lengths


def add_sink_options(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sink", action="store_true", help="Use synthetic FP32 [64] sink weights")
    group.add_argument("--sink-file", help="Load a .pt file containing the model's FP32 [64] sink tensor")
