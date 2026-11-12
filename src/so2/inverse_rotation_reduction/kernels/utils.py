import warnings

import torch

from ...utils.layout import get_angular_layout


def logical_rows(spec, order):
    width = spec.sector_widths[order]
    return width if order == 0 else 2 * width


def validate_messages(messages, edges, lmax, mmax):
    if not 0 <= mmax <= lmax or len(messages) != mmax + 1:
        raise ValueError("one message tensor is required for every retained order")
    if messages[0].ndim != 3:
        raise ValueError("sum aggregation requires rank-three message sectors")
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=messages[0].shape[-1])
    for order, (message, width) in enumerate(zip(messages, spec.sector_widths, strict=True)):
        if order == 0:
            expected = (edges, width, spec.channels)
        else:
            expected = (edges, 2, 2 * width * spec.channels)
        if tuple(message.shape) != expected:
            raise ValueError(f"unexpected message shape for order {order}")
    return spec


def require_cuda_fp32(
    *tensors: torch.Tensor,
    operation: str,
    contiguous: bool = False,
) -> None:
    if not tensors or not isinstance(tensors[0], torch.Tensor):
        raise ValueError(f"{operation} requires tensor inputs")
    device = tensors[0].device
    for value in tensors:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{operation} requires tensor inputs")
        if value.dtype != torch.float32:
            raise ValueError(f"{operation} requires CUDA FP32 tensors")
        if contiguous and not value.is_contiguous():
            raise ValueError(f"{operation} requires contiguous tensors")
        if value.device != device:
            raise ValueError(f"{operation} inputs must share one device")
    if device.type != "cuda":
        raise ValueError(f"{operation} requires CUDA FP32 tensors")


def allow_atomic_reduction(operation):
    if not torch.are_deterministic_algorithms_enabled():
        return
    message = (
        f"{operation} uses floating-point atomic reductions and is not deterministic"
    )
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    else:
        raise RuntimeError(message)
