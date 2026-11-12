import torch
from torch.autograd.function import once_differentiable

from ..utils.layout import retained_rotation_size
from .kernels.backward import sum_backward
from .kernels.double_backward import sum_double_backward
from .kernels.forward import receiver_sector_tail_forward
from .kernels.utils import require_cuda_fp32, validate_messages


class SumAggregationBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, envelope, wigner, receivers, row_ptr, lmax, mmax, *messages):
        gradients = sum_backward(
            grad_output, messages, envelope, wigner, receivers, lmax=lmax, mmax=mmax
        )
        ctx.save_for_backward(grad_output, envelope, wigner, receivers, row_ptr, *messages)
        ctx.orders = (lmax, mmax)
        return gradients

    @staticmethod
    @once_differentiable
    def backward(ctx, envelope_direction, wigner_direction, *message_directions):
        grad_output, envelope, wigner, receivers, row_ptr, *messages = ctx.saved_tensors
        if envelope_direction is None:
            envelope_direction = torch.zeros_like(envelope)
        if wigner_direction is None:
            wigner_direction = torch.zeros_like(wigner)
        message_directions = tuple(
            torch.zeros_like(message) if direction is None else direction.contiguous()
            for message, direction in zip(messages, message_directions, strict=True)
        )
        lmax, mmax = ctx.orders
        gradients = sum_double_backward(
            grad_output,
            messages, envelope, wigner,
            receivers, row_ptr,
            envelope_direction.contiguous(), wigner_direction.contiguous(), message_directions,
            lmax=lmax, mmax=mmax,
        )
        doutput, denvelope, dwigner, *dmessages = gradients
        return doutput, denvelope, dwigner, None, None, None, None, *dmessages


class SumAggregation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, envelope, wigner, receivers, row_ptr, lmax, mmax, *messages):
        nodes = row_ptr.numel() - 1
        output = messages[0].new_empty((nodes, (lmax + 1) ** 2, messages[0].shape[-1]))
        receiver_sector_tail_forward(list(messages), envelope, wigner, row_ptr, output, lmax, mmax)
        ctx.save_for_backward(envelope, wigner, receivers, row_ptr, *messages)
        ctx.orders = (lmax, mmax)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        envelope, wigner, receivers, row_ptr, *messages = ctx.saved_tensors
        lmax, mmax = ctx.orders
        grad_output = grad_output.contiguous()
        if torch.is_grad_enabled():
            gradients = SumAggregationBackward.apply(
                grad_output, envelope, wigner, receivers, row_ptr, lmax, mmax, *messages
            )
        else:
            gradients = sum_backward(
                grad_output, messages, envelope, wigner, receivers, lmax=lmax, mmax=mmax
            )
        denvelope, dwigner, *dmessages = gradients
        return denvelope, dwigner, None, None, None, None, *dmessages


def sum_aggregation(messages, envelope, wigner, receivers, nodes, *, lmax, mmax, row_ptr):
    if envelope.ndim != 1 and not (envelope.ndim == 2 and envelope.shape[1] == 1):
        raise ValueError("envelope must contain one scalar per edge")
    edges = envelope.shape[0]
    validate_messages(messages, edges, lmax, mmax)
    require_cuda_fp32(*messages, envelope, wigner, operation="SO(2) sum aggregation")
    if wigner.shape != (edges, retained_rotation_size(lmax, mmax)) or not wigner.is_contiguous():
        raise ValueError("expected contiguous retained rotations of shape (E, K)")
    if (
        receivers.dtype not in (torch.int32, torch.int64)
        or receivers.device != envelope.device
        or tuple(receivers.shape) != (edges,)
    ):
        raise ValueError("receivers must be integer edge indices on the message device")
    if (
        nodes <= 0
        or row_ptr.dtype != torch.int32
        or row_ptr.device != envelope.device
        or tuple(row_ptr.shape) != (nodes + 1,)
    ):
        raise ValueError("row_ptr must be int32 with nodes + 1 entries on the message device")
    return SumAggregation.apply(
        envelope.contiguous(), wigner,
        receivers.contiguous(), row_ptr.contiguous(),
        lmax, mmax, *messages,
    )
