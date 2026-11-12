from __future__ import annotations

from torch.autograd.function import once_differentiable
import torch

from .kernels.backward import gather_backward_source_csr
from .kernels.double_backward import gather_double_backward_source_csr
from .kernels.forward import prefix_sector_forward


class CSRPrefixBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x, edge_index, radial, wigner,
        source_order,
        source_row_ptr,
        receiver_row_ptr,
        lmax,
        mmax,
        *grad_outputs,
    ):
        gradients = gather_backward_source_csr(
            grad_outputs,
            x, edge_index, radial, wigner,
            (source_order, source_row_ptr, receiver_row_ptr),
            lmax=lmax, mmax=mmax,
        )
        ctx.save_for_backward(
            x, edge_index, radial, wigner,
            source_order,
            source_row_ptr,
            receiver_row_ptr,
            *grad_outputs,
        )
        ctx.dimensions = (lmax, mmax, len(grad_outputs))
        return gradients

    @staticmethod
    @once_differentiable
    def backward(ctx, input_direction, radial_direction, wigner_direction):
        (
            x, edge_index, radial, wigner,
            source_order,
            source_row_ptr,
            receiver_row_ptr,
            *grad_outputs,
        ) = ctx.saved_tensors
        if input_direction is None:
            input_direction = torch.zeros_like(x)
        if radial_direction is None:
            radial_direction = torch.zeros_like(radial)
        if wigner_direction is None:
            wigner_direction = torch.zeros_like(wigner)
        lmax, mmax, grad_count = ctx.dimensions
        if len(grad_outputs) != grad_count:
            raise RuntimeError("saved prefix sector count changed")
        gradients = gather_double_backward_source_csr(
            grad_outputs,
            x, edge_index, radial, wigner,
            (source_order, source_row_ptr, receiver_row_ptr),
            input_direction.contiguous(),
            radial_direction.contiguous(),
            wigner_direction.contiguous(),
            lmax=lmax, mmax=mmax,
        )
        return (
            gradients[grad_count],
            None,
            gradients[grad_count + 1],
            gradients[grad_count + 2],
            None,
            None,
            None,
            None,
            None,
            *gradients[:grad_count],
        )


class CSRPrefix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x, edge_index, radial, wigner,
        source_order,
        source_row_ptr,
        receiver_row_ptr,
        lmax,
        mmax,
    ):
        sectors = prefix_sector_forward(
            x, edge_index, radial, wigner, lmax=lmax, mmax=mmax,
        )
        ctx.save_for_backward(
            x, edge_index, radial, wigner,
            source_order,
            source_row_ptr,
            receiver_row_ptr,
        )
        ctx.dimensions = (lmax, mmax)
        return tuple(sectors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            x, edge_index, radial, wigner,
            source_order,
            source_row_ptr,
            receiver_row_ptr,
        ) = ctx.saved_tensors
        lmax, mmax = ctx.dimensions
        contiguous_grads = tuple(grad.contiguous() for grad in grad_outputs)
        if not torch.is_grad_enabled() and not torch.are_deterministic_algorithms_enabled():
            gradients = gather_backward_source_csr(
                contiguous_grads,
                x, edge_index, radial, wigner,
                (source_order, source_row_ptr, receiver_row_ptr),
                lmax=lmax, mmax=mmax,
            )
        else:
            gradients = CSRPrefixBackward.apply(
                x, edge_index, radial, wigner,
                source_order, source_row_ptr, receiver_row_ptr,
                lmax, mmax,
                *contiguous_grads,
            )
        return (
            gradients[0],
            None,
            gradients[1],
            gradients[2],
            None,
            None,
            None,
            None,
            None,
        )


def rotate_gather_order_degree(
    x, edge_index, radial, wigner,
    *,
    lmax,
    mmax,
    metadata,
):
    if not 0 <= mmax <= lmax:
        raise ValueError("CSR prefix supports 0 <= mmax <= lmax")
    if len(metadata) != 3:
        raise ValueError(
            "prefix metadata must contain source order and two CSR row pointers"
        )
    expected = ((edge_index.shape[1],), (x.shape[0] + 1,), (x.shape[0] + 1,))
    if any(
        tuple(value.shape) != shape
        or value.dtype != torch.int32
        or value.device != edge_index.device
        or not value.is_contiguous()
        for value, shape in zip(metadata, expected)
    ):
        raise ValueError(
            "prefix metadata must be contiguous int32 tensors matching this graph"
        )
    return CSRPrefix.apply(
        x, edge_index, radial, wigner,
        *metadata,
        lmax,
        mmax,
    )
