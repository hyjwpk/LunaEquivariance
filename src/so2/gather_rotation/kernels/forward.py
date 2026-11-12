from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.sector_layout import retained_wigner_offset
from .utils import (
    sector_metadata,
    validate_inputs,
)


@triton.jit
def prefix_sector_forward_kernel(
    x_ptr, edge_index_ptr, radial_ptr, wigner_ptr,
    outputs,
    edges,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    reduced_components: tl.constexpr,
    reduced_degrees: tl.constexpr,
    dense_rows: tl.constexpr,
    radial_row_indices: tl.constexpr,
    radial_rows: tl.constexpr,
    mmax: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    block_channels: tl.constexpr,
    compact_elements: tl.constexpr,
):
    edge = tl.program_id(0)
    branch = tl.program_id(1)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    node = tl.load(edge_index_ptr + branch * edges + edge)
    output_channel = branch * channels + channel

    for row in tl.static_range(0, reduced_components):
        degree = list(reduced_degrees)[row]
        accumulator = tl.zeros(channel.shape, dtype=tl.float32)
        for local_component in tl.static_range(0, 2 * list(reduced_degrees)[row] + 1):
            component = degree * degree + local_component
            feature = tl.load(
                x_ptr
                + node * full_components * channels
                + component * channels
                + channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient = tl.load(
                wigner_ptr
                + retained_wigner_offset(
                    edge,
                    tl.constexpr(list(dense_rows)[row]),
                    component,
                    compact_elements,
                    mmax,
                )
            ).to(tl.float32)
            accumulator += coefficient * feature
        radial_value = tl.load(
            radial_ptr
            + edge * radial_rows * 2 * channels
            + tl.constexpr(list(radial_row_indices)[row]) * 2 * channels
            + output_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        value = accumulator * radial_value
        pointer = outputs[0] + edge * list(elements)[0] + row * 2 * channels
        for order in tl.static_range(1, mmax + 1):
            if row >= list(offsets)[order - 1]:
                pointer = (
                    outputs[order]
                    + edge * list(elements)[order]
                    + (row - list(offsets)[order - 1]) * 2 * channels
                )
        pointer += output_channel
        tl.store(pointer, value, mask=channel_mask)


def prefix_sector_forward(x, edge_index, radial, wigner, *, lmax, mmax):
    spec = validate_inputs(x, edge_index, radial, wigner, lmax=lmax, mmax=mmax)
    block_channels = triton.next_power_of_2(spec.channels)
    edges = edge_index.shape[1]
    outputs = []
    for order, width in enumerate(spec.sector_widths):
        shape = (
            (edges, width * 2 * spec.channels)
            if order == 0
            else (edges, 2, width * 2 * spec.channels)
        )
        outputs.append(torch.empty(shape, device=x.device, dtype=x.dtype))
    offsets, sector_elements = sector_metadata(spec)
    radial_row_indices = tuple(
        offset // (2 * spec.channels) for offset in spec.radial_row_offsets
    )
    radial_rows = spec.radial_elements // (2 * spec.channels)
    prefix_sector_forward_kernel[(edges, 2)](
        x, edge_index, radial, wigner,
        tuple(outputs),
        edges,
        channels=spec.channels,
        full_components=spec.full_components,
        reduced_components=spec.reduced_components,
        reduced_degrees=spec.reduced_degrees,
        dense_rows=spec.dense_rows,
        radial_row_indices=radial_row_indices,
        radial_rows=radial_rows,
        mmax=spec.mmax,
        offsets=offsets,
        elements=sector_elements,
        block_channels=block_channels,
        compact_elements=wigner.stride(0),
    )
    return tuple(outputs)
