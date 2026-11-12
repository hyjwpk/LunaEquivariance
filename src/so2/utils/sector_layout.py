from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def retained_wigner_offset(
    edge,
    dense_row,
    full,
    compact_elements: tl.constexpr,
    mmax: tl.constexpr,
):
    degree = tl.sqrt(full.to(tl.float32)).to(tl.int32)
    block_start = degree * (4 * degree * degree - 1) // 3
    degree_start = degree * degree
    width = 2 * degree + 1
    complete_degrees: tl.constexpr = mmax + 1
    complete_elements: tl.constexpr = (
        complete_degrees * (4 * complete_degrees * complete_degrees - 1) // 3
    )
    block_start = tl.where(
        degree <= complete_degrees,
        block_start,
        complete_elements
        + (2 * mmax + 1)
        * (degree * degree - complete_degrees * complete_degrees),
    )
    order = dense_row - degree_start - degree
    selected_row = tl.where(order == 0, 0, 2 * tl.abs(order) - (order > 0).to(tl.int32))
    return (
        edge * compact_elements
        + block_start
        + selected_row * width
        + full
        - degree_start
    )


@triton.jit
def load_sector_value(
    pointer, edge, row, channel,
    stride_edge, stride_row, stride_channel,
    channels: tl.constexpr,
    width: tl.constexpr,
    order: tl.constexpr,
    mask,
):
    if order > 0:
        positive = row < width
        local_row = tl.where(positive, row, row - width)
        flat = local_row * channels + channel
        half = width * channels
        first_address = (
            edge * stride_edge
            + tl.where(positive, 0, stride_row)
            + flat * stride_channel
        )
        second_address = (
            edge * stride_edge
            + tl.where(positive, stride_row, 0)
            + (half + flat) * stride_channel
        )
        first = tl.load(pointer + first_address, mask=mask, other=0.0).to(tl.float32)
        second = tl.load(pointer + second_address, mask=mask, other=0.0).to(tl.float32)
        return tl.where(positive, first - second, first + second)
    return tl.load(
        pointer + edge * stride_edge + row * stride_row + channel * stride_channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)


@triton.jit
def store_sector_gradient(
    pointer, value, edge, row, channel,
    stride_edge, stride_row, stride_channel,
    channels: tl.constexpr,
    width: tl.constexpr,
    order: tl.constexpr,
    mask,
):
    if order > 0:
        positive = row < width
        local_row = tl.where(positive, row, row - width)
        flat = local_row * channels + channel
        half = width * channels
        first_address = (
            edge * stride_edge
            + tl.where(positive, 0, stride_row)
            + flat * stride_channel
        )
        second_address = (
            edge * stride_edge
            + tl.where(positive, stride_row, 0)
            + (half + flat) * stride_channel
        )
        tl.store(pointer + first_address, value, mask=mask)
        tl.store(
            pointer + second_address,
            tl.where(positive, -value, value),
            mask=mask,
        )
    else:
        tl.store(
            pointer + edge * stride_edge + row * stride_row + channel * stride_channel,
            value,
            mask=mask,
        )
