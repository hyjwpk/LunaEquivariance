from __future__ import annotations

import triton
import triton.language as tl

from ...utils.sector_layout import load_sector_value, store_sector_gradient


@triton.jit
def silu_terms(value):
    sigmoid = tl.sigmoid(value)
    first = sigmoid * (1.0 + value * (1.0 - sigmoid))
    second = sigmoid * (1.0 - sigmoid) * (2.0 + value * (1.0 - 2.0 * sigmoid))
    return value * sigmoid, first, second


@triton.jit
def load_raw(
    pointer, edge, row, channel,
    width: tl.constexpr,
    channels: tl.constexpr,
    mask,
):
    stride_edge = 4 * width * channels
    stride_row = 2 * width * channels
    return load_sector_value(
        pointer, edge, row, channel,
        stride_edge, stride_row, 1,
        channels, width, 1, mask,
    )


@triton.jit
def store_raw(
    pointer, value, edge, row, channel,
    width: tl.constexpr,
    channels: tl.constexpr,
    mask,
):
    stride_edge = 4 * width * channels
    stride_row = 2 * width * channels
    store_sector_gradient(
        pointer, value, edge, row, channel,
        stride_edge, stride_row, 1,
        channels, width, 1, mask,
    )
