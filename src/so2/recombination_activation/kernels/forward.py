from __future__ import annotations


import triton
import triton.language as tl

from .utils import load_raw, silu_terms


@triton.jit
def forward_kernel(
    gating_ptr, x_ptrs, y_ptrs,
    gating_stride_edge, x0_stride_edge,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    mmax: tl.constexpr,
    block_channels: tl.constexpr,
):
    edge = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    mask = channel < channels
    scalar = tl.load(
        x_ptrs[0] + edge * x0_stride_edge + channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    silu, _, _ = silu_terms(scalar)
    tl.store(y_ptrs[0] + edge * (lmax + 1) * channels + channel, silu, mask=mask)

    for degree in tl.static_range(1, lmax + 1):
        gate = tl.sigmoid(
            tl.load(
                gating_ptr
                + edge * gating_stride_edge
                + (degree - 1) * channels
                + channel,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        )
        x0_address = edge * x0_stride_edge + degree * channels + channel
        y0_address = edge * (lmax + 1) * channels + degree * channels + channel
        tl.store(
            y_ptrs[0] + y0_address,
            tl.load(x_ptrs[0] + x0_address, mask=mask, other=0.0) * gate,
            mask=mask,
        )
        for order in tl.static_range(1, mmax + 1):
            if order <= degree:
                width = lmax - order + 1
                local = degree - order
                x_ptr = x_ptrs[order]
                y_ptr = y_ptrs[order]
                for sign in tl.static_range(0, 2):
                    row = local + sign * width
                    value = load_raw(
                        x_ptr,
                        edge,
                        row,
                        channel,
                        width,
                        channels,
                        mask,
                    )
                    tl.store(
                        y_ptr + edge * 2 * width * channels + row * channels + channel,
                        value * gate,
                        mask=mask,
                    )
