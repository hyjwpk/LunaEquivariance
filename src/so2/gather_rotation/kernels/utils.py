from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.layout import get_angular_layout, retained_rotation_size


@triton.jit
def sector_pointer(
    pointers,
    edge,
    row,
    mmax: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    channels: tl.constexpr,
):
    if mmax == 0:
        pointer = pointers[0] + edge * list(elements)[0] + row * 2 * channels
    elif mmax <= 4:
        offset1: tl.constexpr = list(offsets)[0]
        offset2: tl.constexpr = list(offsets)[1] if mmax > 1 else 0
        offset3: tl.constexpr = list(offsets)[2] if mmax > 2 else 0
        offset4: tl.constexpr = list(offsets)[3] if mmax > 3 else 0
        if row < offset1:
            pointer = pointers[0] + edge * list(elements)[0] + row * 2 * channels
        elif mmax == 1 or row < offset2:
            pointer = (
                pointers[1] + edge * list(elements)[1] + (row - offset1) * 2 * channels
            )
        elif mmax == 2 or row < offset3:
            pointer = (
                pointers[2 if mmax >= 2 else 0]
                + edge * list(elements)[2 if mmax >= 2 else 0]
                + (row - offset2) * 2 * channels
            )
        elif mmax == 3 or row < offset4:
            pointer = (
                pointers[3 if mmax >= 3 else 0]
                + edge * list(elements)[3 if mmax >= 3 else 0]
                + (row - offset3) * 2 * channels
            )
        else:
            pointer = (
                pointers[4 if mmax >= 4 else 0]
                + edge * list(elements)[4 if mmax >= 4 else 0]
                + (row - offset4) * 2 * channels
            )
    else:
        pointer = pointers[0] + edge * list(elements)[0] + row * 2 * channels
        for order in tl.static_range(1, mmax + 1):
            if row >= list(offsets)[order - 1]:
                pointer = (
                    pointers[order]
                    + edge * list(elements)[order]
                    + (row - list(offsets)[order - 1]) * 2 * channels
                )
    return pointer


@triton.jit
def load_sector_cotangent(
    grads,
    edge,
    row,
    output_channel,
    mask,
    mmax: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    channels: tl.constexpr,
):
    pointer = sector_pointer(grads, edge, row, mmax, offsets, elements, channels)
    return tl.load(pointer + output_channel, mask=mask, other=0.0).to(tl.float32)


@triton.jit
def store_sector_direction(
    grads,
    edge,
    row,
    output_channel,
    value,
    mask,
    mmax: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    channels: tl.constexpr,
):
    pointer = sector_pointer(grads, edge, row, mmax, offsets, elements, channels)
    tl.store(pointer + output_channel, value, mask=mask)


def validate_inputs(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    radial: torch.Tensor,
    wigner: torch.Tensor,
    *,
    lmax: int,
    mmax: int,
):
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=x.shape[2])
    edges = edge_index.shape[1]
    expected = {
        "x": (x.shape[0], spec.full_components, spec.channels),
        "edge_index": (2, edges),
        "radial": (edges, spec.radial_elements),
        "wigner": (edges, retained_rotation_size(lmax, mmax)),
    }
    actual = {
        "x": tuple(x.shape),
        "edge_index": tuple(edge_index.shape),
        "radial": tuple(radial.shape),
        "wigner": tuple(wigner.shape),
    }
    if expected != actual:
        raise ValueError(f"unexpected sector prefix shapes: {actual=}, {expected=}")
    if not all(value.is_cuda for value in (x, edge_index, radial, wigner)):
        raise ValueError("sector prefix inputs must be CUDA tensors")
    if not all(value.is_contiguous() for value in (x, edge_index, radial, wigner)):
        raise ValueError("sector prefix inputs must be contiguous")
    if x.dtype != torch.float32 or radial.dtype != x.dtype or wigner.dtype != x.dtype:
        raise ValueError("the sector prefix runtime is FP32 only")
    if any(value.device != x.device for value in (edge_index, radial, wigner)):
        raise ValueError("prefix inputs must share a device")
    if edge_index.dtype != torch.int64:
        raise ValueError("edge_index must be int64")
    return spec


def sector_metadata(spec):
    starts = tuple(spec.sector_row_offsets[1:])
    row_counts = tuple(
        width if order == 0 else 2 * width
        for order, width in enumerate(spec.sector_widths)
    )
    elements = tuple(rows * 2 * spec.channels for rows in row_counts)
    return starts, elements
