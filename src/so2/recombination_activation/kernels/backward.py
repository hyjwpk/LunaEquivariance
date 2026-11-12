from __future__ import annotations


import triton
import triton.language as tl

from .utils import silu_terms, store_raw


@triton.jit
def backward_kernel(
    grad_ptrs,
    gating_ptr,
    x_ptrs,
    gating_gradient_ptr,
    x_gradient_ptrs,
    gating_stride_edge,
    x0_stride_edge,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    mmax: tl.constexpr,
    block_channels: tl.constexpr,
):
    edge = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    mask = channel < channels
    x0_scalar_address = edge * x0_stride_edge + channel
    grad0_scalar_address = edge * (lmax + 1) * channels + channel
    scalar = tl.load(x_ptrs[0] + x0_scalar_address, mask=mask, other=0.0).to(tl.float32)
    scalar_grad = tl.load(grad_ptrs[0] + grad0_scalar_address, mask=mask, other=0.0).to(
        tl.float32
    )
    _, silu_first, _ = silu_terms(scalar)
    tl.store(
        x_gradient_ptrs[0] + grad0_scalar_address, scalar_grad * silu_first, mask=mask
    )

    for degree in tl.static_range(1, lmax + 1):
        gate_input_address = (
            edge * gating_stride_edge + (degree - 1) * channels + channel
        )
        gate_output_address = edge * lmax * channels + (degree - 1) * channels + channel
        gate = tl.sigmoid(
            tl.load(gating_ptr + gate_input_address, mask=mask, other=0.0).to(
                tl.float32
            )
        )
        grad0_address = edge * (lmax + 1) * channels + degree * channels + channel
        x0_address = edge * x0_stride_edge + degree * channels + channel
        grad0 = tl.load(grad_ptrs[0] + grad0_address, mask=mask, other=0.0).to(
            tl.float32
        )
        value0 = tl.load(x_ptrs[0] + x0_address, mask=mask, other=0.0).to(tl.float32)
        tl.store(x_gradient_ptrs[0] + grad0_address, grad0 * gate, mask=mask)
        contraction = grad0 * (gate * value0)
        for order in tl.static_range(1, mmax + 1):
            if order <= degree:
                width = lmax - order + 1
                local = degree - order
                grad_ptr = grad_ptrs[order]
                x_ptr = x_ptrs[order]
                x_gradient_ptr = x_gradient_ptrs[order]
                for sign in tl.static_range(0, 2):
                    row = local + sign * width
                    grad_address = (
                        edge * 2 * width * channels + row * channels + channel
                    )
                    gradient = tl.load(
                        grad_ptr + grad_address, mask=mask, other=0.0
                    ).to(tl.float32)
                    value = tl.load(x_ptr + grad_address, mask=mask, other=0.0).to(
                        tl.float32
                    )
                    store_raw(
                        x_gradient_ptr,
                        gradient * gate,
                        edge,
                        row,
                        channel,
                        width,
                        channels,
                        mask,
                    )
                    contraction += gradient * value
        tl.store(
            gating_gradient_ptr + gate_output_address,
            (1.0 - gate) * contraction,
            mask=mask,
        )


def gate_backward(grads, gating, sectors, *, lmax, mmax):
    edges, channels = gating.shape[0], sectors[0].shape[-1]
    dgating = gating.new_empty(gating.shape)
    dsectors = (sectors[0].new_empty(sectors[0].shape),) + tuple(
        sectors[0].new_empty((edges, 2, 2 * (lmax - order + 1) * channels))
        for order in range(1, mmax + 1)
    )
    block = triton.next_power_of_2(channels)
    backward_kernel[(edges,)](
        grads, gating, tuple(sectors),
        dgating, dsectors,
        gating.stride(0), sectors[0].stride(0),
        channels=channels,
        lmax=lmax,
        mmax=mmax,
        block_channels=block,
    )
    return dgating, *dsectors
