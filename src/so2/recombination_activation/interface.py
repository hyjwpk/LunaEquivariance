from __future__ import annotations

from torch.autograd.function import once_differentiable
import torch
import triton

from .kernels.backward import gate_backward
from .kernels.double_backward import double_backward_kernel
from .kernels.forward import forward_kernel


class GroupedGateBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gating, lmax, mmax, *args):
        count = mmax + 1
        grads, sectors = args[:count], args[count:]
        values = gate_backward(grads, gating, sectors, lmax=lmax, mmax=mmax)
        ctx.save_for_backward(*grads, gating, *sectors)
        ctx.dimensions = (lmax, mmax)
        return values

    @staticmethod
    @once_differentiable
    def backward(ctx, gating_direction, *sector_directions):
        lmax, mmax = ctx.dimensions
        count = mmax + 1
        grads = ctx.saved_tensors[:count]
        gating = ctx.saved_tensors[count]
        sectors = ctx.saved_tensors[count + 1 :]
        if gating_direction is None:
            gating_direction = torch.zeros_like(gating)
        gating_direction = gating_direction.contiguous()
        edges, channels = gating.shape[0], sectors[0].shape[-1]
        directions = []
        for order, direction in enumerate(sector_directions):
            if direction is None:
                shape = (
                    tuple(sectors[0].shape)
                    if order == 0
                    else (edges, 2, 2 * (lmax - order + 1) * channels)
                )
                direction = sectors[0].new_zeros(shape)
            directions.append(direction)
        directions = (directions[0].contiguous(), *directions[1:])
        dgrads = tuple(torch.empty_like(value) for value in grads)
        dgating = gating.new_empty(gating.shape)
        dsectors = tuple(value.new_empty(value.shape) for value in sectors)
        block = triton.next_power_of_2(channels)
        double_backward_kernel[(edges,)](
            grads, gating, sectors,
            gating_direction, directions,
            dgrads, dgating, dsectors,
            gating.stride(0), sectors[0].stride(0), gating_direction.stride(0),
            directions[0].stride(0), directions[0].stride(1),
            tuple(value.stride() for value in directions),
            channels=channels,
            lmax=lmax,
            mmax=mmax,
            block_channels=block,
        )
        return dgating, None, None, *dgrads, *dsectors


class GroupedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gating, lmax, mmax, *sectors):
        edges, channels = gating.shape[0], sectors[0].shape[-1]
        outputs = (sectors[0].new_empty(sectors[0].shape),) + tuple(
            sectors[0].new_empty((edges, 2 * (lmax - order + 1), channels))
            for order in range(1, mmax + 1)
        )
        block = triton.next_power_of_2(channels)
        forward_kernel[(edges,)](
            gating, sectors, outputs,
            gating.stride(0), sectors[0].stride(0),
            channels=channels,
            lmax=lmax,
            mmax=mmax,
            block_channels=block,
        )
        ctx.save_for_backward(gating, sectors[0], *outputs[1:])
        ctx.dimensions = (lmax, mmax)
        return outputs

    @staticmethod
    def backward(ctx, *grads):
        gating, *sectors = ctx.saved_tensors
        lmax, mmax = ctx.dimensions
        grads = tuple(
            sector.new_zeros(sector.shape) if grad is None else grad.contiguous()
            for sector, grad in zip(sectors, grads, strict=True)
        )
        if not torch.is_grad_enabled():
            values = gate_backward(grads, gating, sectors, lmax=lmax, mmax=mmax)
            return values[0], None, None, *values[1:]
        values = GroupedGateBackward.apply(gating, lmax, mmax, *grads, *sectors)
        return values[0], None, None, *values[1:]


def gated_activation(gating, input0, inputs, *, lmax, mmax):
    if not 0 <= mmax <= lmax or len(inputs) != mmax:
        raise ValueError(
            "grouped raw gate requires one sector for each 0 <= m <= M <= L"
        )
    if input0.ndim != 3 or input0.shape[-1] <= 0:
        raise ValueError(
            "grouped raw gate requires rank-three sectors with positive channels"
        )
    edges, channels = input0.shape[0], input0.shape[-1]
    expected = [(edges, lmax + 1, channels)]
    expected.extend(
        (edges, 2, 2 * (lmax - order + 1) * channels) for order in range(1, mmax + 1)
    )
    actual = [tuple(input0.shape), *(tuple(value.shape) for value in inputs)]
    if actual != expected or tuple(gating.shape) != (edges, lmax * channels):
        raise ValueError("grouped raw gate received inconsistent sector shapes")
    if not all(
        value.is_cuda and value.dtype == torch.float32 and value.device == input0.device
        for value in (gating, input0, *inputs)
    ):
        raise ValueError("grouped raw gate requires CUDA FP32 tensors")
    if gating.stride(1) != 1:
        gating = gating.contiguous()
    if input0.stride(1) != channels or input0.stride(2) != 1:
        input0 = input0.contiguous()
    inputs = tuple(value.contiguous() for value in inputs)
    sectors = (input0, *inputs)
    return GroupedGate.apply(gating, lmax, mmax, *sectors)
