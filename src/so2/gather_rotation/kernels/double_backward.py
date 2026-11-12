from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.sector_layout import retained_wigner_offset
from .utils import (
    load_sector_cotangent,
    sector_metadata,
    store_sector_direction,
    validate_inputs,
)


@triton.jit
def prefix_wigner_degree_double_backward_kernel(
    grads,
    x_ptr,
    edge_index_ptr,
    radial_ptr,
    input_direction_ptr,
    radial_direction_ptr,
    wigner_gradient_ptr,
    edges,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    radial_elements: tl.constexpr,
    lmax: tl.constexpr,
    mmax: tl.constexpr,
    degree: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    component_block: tl.constexpr,
    channel_block: tl.constexpr,
    compact_elements: tl.constexpr,
):
    edge = tl.program_id(0)
    slot = tl.program_id(1)
    is_scalar = slot == 0
    order = tl.where(is_scalar, 0, (slot + 1) // 2)
    sign = (slot - 1) & 1
    width = lmax - order + 1
    sector_start = lmax + 1 + (order - 1) * (2 * (lmax + 1) - order)
    sector_row = tl.where(is_scalar, degree, sector_start + degree - order + sign * width)
    dense_center = degree * degree + degree
    dense_row = tl.where(is_scalar, dense_center, dense_center + tl.where(sign == 0, order, -order))
    radial_start = (
        ((lmax + 1) + (order - 1) * (lmax + 1) - ((order - 1) * order) // 2)
        * 2
        * channels
    )
    radial_row_offset = tl.where(
        is_scalar,
        degree * 2 * channels,
        radial_start + (degree - order) * 2 * channels,
    )

    local_component_1d = tl.arange(0, component_block)
    local_component = local_component_1d[:, None]
    component_mask_1d = local_component_1d < 2 * degree + 1
    component_mask = component_mask_1d[:, None]
    channel = tl.arange(0, channel_block)[None, :]
    channel_mask = channel < channels
    value_mask = component_mask & channel_mask
    component = degree * degree + local_component
    source_node = tl.load(edge_index_ptr + edge)
    target_node = tl.load(edge_index_ptr + edges + edge)
    source_address = (
        source_node * full_components * channels + component * channels + channel
    )
    target_address = (
        target_node * full_components * channels + component * channels + channel
    )
    source = tl.load(x_ptr + source_address, mask=value_mask, other=0.0).to(tl.float32)
    target = tl.load(x_ptr + target_address, mask=value_mask, other=0.0).to(tl.float32)
    source_direction = tl.load(input_direction_ptr + source_address, mask=value_mask, other=0.0).to(tl.float32)
    target_direction = tl.load(input_direction_ptr + target_address, mask=value_mask, other=0.0).to(tl.float32)
    radial_base = edge * radial_elements + radial_row_offset
    source_radial = tl.load(radial_ptr + radial_base + channel, mask=channel_mask, other=0.0).to(tl.float32)
    target_radial = tl.load(radial_ptr + radial_base + channels + channel, mask=channel_mask, other=0.0).to(tl.float32)
    source_radial_direction = tl.load(
        radial_direction_ptr + radial_base + channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    target_radial_direction = tl.load(
        radial_direction_ptr + radial_base + channels + channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    source_cotangent = load_sector_cotangent(
        grads,
        edge,
        sector_row,
        channel,
        channel_mask,
        mmax,
        offsets,
        elements,
        channels,
    )
    target_cotangent = load_sector_cotangent(
        grads,
        edge,
        sector_row,
        channels + channel,
        channel_mask,
        mmax,
        offsets,
        elements,
        channels,
    )
    gradient = tl.sum(
        source_cotangent
        * (source_radial * source_direction + source_radial_direction * source)
        + target_cotangent
        * (target_radial * target_direction + target_radial_direction * target),
        axis=1,
    )
    tl.store(
        wigner_gradient_ptr
        + retained_wigner_offset(
            edge,
            dense_row,
            degree * degree + local_component_1d,
            compact_elements,
            mmax,
        ),
        gradient,
        mask=component_mask_1d,
    )


@triton.jit
def prefix_edge_double_backward_kernel(
    grads,
    x_ptr, edge_index_ptr, radial_ptr, wigner_ptr,
    input_direction_ptr,
    radial_direction_ptr,
    wigner_direction_ptr,
    grad_gradients,
    radial_gradient_ptr,
    edges,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    radial_group_rows0: tl.constexpr,
    radial_group_rows1: tl.constexpr,
    radial_group_dense_rows0: tl.constexpr,
    radial_group_dense_rows1: tl.constexpr,
    radial_group_degrees: tl.constexpr,
    radial_group_offsets: tl.constexpr,
    radial_group_count: tl.constexpr,
    radial_elements: tl.constexpr,
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
    for group in tl.static_range(0, radial_group_count):
        row0 = list(radial_group_rows0)[group]
        row1 = list(radial_group_rows1)[group]
        degree = list(radial_group_degrees)[group]
        radial_address = (
            edge * radial_elements
            + tl.constexpr(list(radial_group_offsets)[group])
            + output_channel
        )
        radial = tl.load(radial_ptr + radial_address, mask=channel_mask, other=0.0).to(tl.float32)
        radial_direction = tl.load(radial_direction_ptr + radial_address, mask=channel_mask, other=0.0).to(tl.float32)
        cotangent0 = load_sector_cotangent(
            grads,
            edge,
            row0,
            output_channel,
            channel_mask,
            mmax,
            offsets,
            elements,
            channels,
        )
        cotangent1 = tl.zeros(channel.shape, dtype=tl.float32)
        if row1 >= 0:
            cotangent1 = load_sector_cotangent(
                grads,
                edge,
                row1,
                output_channel,
                channel_mask,
                mmax,
                offsets,
                elements,
                channels,
            )
        projected0 = tl.zeros(channel.shape, dtype=tl.float32)
        projected1 = tl.zeros(channel.shape, dtype=tl.float32)
        projected_direction0 = tl.zeros(channel.shape, dtype=tl.float32)
        projected_direction1 = tl.zeros(channel.shape, dtype=tl.float32)
        for local_component in tl.static_range(
            0, 2 * list(radial_group_degrees)[group] + 1
        ):
            component = degree * degree + local_component
            feature = tl.load(
                x_ptr
                + node * full_components * channels
                + component * channels
                + channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            feature_direction = tl.load(
                input_direction_ptr
                + node * full_components * channels
                + component * channels
                + channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient0_address = retained_wigner_offset(
                edge,
                tl.constexpr(list(radial_group_dense_rows0)[group]),
                component,
                compact_elements,
                mmax,
            )
            coefficient0 = tl.load(wigner_ptr + coefficient0_address).to(tl.float32)
            coefficient_direction0 = tl.load(wigner_direction_ptr + coefficient0_address).to(tl.float32)
            projected0 += coefficient0 * feature
            projected_direction0 += (
                coefficient0 * feature_direction + coefficient_direction0 * feature
            )
            if row1 >= 0:
                coefficient1_address = retained_wigner_offset(
                    edge,
                    tl.constexpr(list(radial_group_dense_rows1)[group]),
                    component,
                    compact_elements,
                    mmax,
                )
                coefficient1 = tl.load(wigner_ptr + coefficient1_address).to(tl.float32)
                coefficient_direction1 = tl.load(wigner_direction_ptr + coefficient1_address).to(tl.float32)
                projected1 += coefficient1 * feature
                projected_direction1 += (
                    coefficient1 * feature_direction + coefficient_direction1 * feature
                )
        store_sector_direction(
            grad_gradients,
            edge,
            row0,
            output_channel,
            radial * projected_direction0 + radial_direction * projected0,
            channel_mask,
            mmax,
            offsets,
            elements,
            channels,
        )
        if row1 >= 0:
            store_sector_direction(
                grad_gradients,
                edge,
                row1,
                output_channel,
                radial * projected_direction1 + radial_direction * projected1,
                channel_mask,
                mmax,
                offsets,
                elements,
                channels,
            )
        tl.store(
            radial_gradient_ptr + radial_address,
            cotangent0 * projected_direction0 + cotangent1 * projected_direction1,
            mask=channel_mask,
        )


@triton.jit
def prefix_input_csr_double_backward_kernel(
    grads,
    radial_ptr,
    wigner_ptr,
    radial_direction_ptr,
    wigner_direction_ptr,
    source_order_ptr,
    source_row_ptr,
    receiver_row_ptr,
    input_gradient_ptr,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr,
    mmax: tl.constexpr,
    radial_elements: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    block_channels: tl.constexpr,
    compact_elements: tl.constexpr,
):
    node = tl.program_id(0)
    full = tl.program_id(1)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    degree = tl.sqrt(full.to(tl.float32)).to(tl.int32)
    dense_center = degree * degree + degree
    accumulator = tl.zeros(channel.shape, dtype=tl.float32)
    for branch in tl.static_range(0, 2):
        if branch == 0:
            start = tl.load(source_row_ptr + node).to(tl.int32)
            end = tl.load(source_row_ptr + node + 1).to(tl.int32)
        else:
            start = tl.load(receiver_row_ptr + node).to(tl.int32)
            end = tl.load(receiver_row_ptr + node + 1).to(tl.int32)
        output_channel = branch * channels + channel
        for position in range(start, end):
            edge = (
                tl.load(source_order_ptr + position).to(tl.int32)
                if branch == 0
                else position
            )
            radial0_address = (
                edge * radial_elements + degree * 2 * channels + output_channel
            )
            radial0 = tl.load(radial_ptr + radial0_address, mask=channel_mask, other=0.0).to(tl.float32)
            radial0_direction = tl.load(
                radial_direction_ptr + radial0_address,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient0_address = retained_wigner_offset(
                edge,
                dense_center,
                full,
                compact_elements,
                mmax,
            )
            coefficient0 = tl.load(wigner_ptr + coefficient0_address).to(tl.float32)
            coefficient0_direction = tl.load(wigner_direction_ptr + coefficient0_address).to(tl.float32)
            cotangent0 = load_sector_cotangent(
                grads,
                edge,
                degree,
                output_channel,
                channel_mask,
                mmax,
                offsets,
                elements,
                channels,
            )
            local = cotangent0 * (
                radial0_direction * coefficient0 + radial0 * coefficient0_direction
            )

            for order in tl.static_range(1, mmax + 1):
                order_mask = channel_mask & (degree >= order)
                width = lmax - order + 1
                local_row = degree - order
                radial_start = (
                    ((lmax + 1) + (order - 1) * (lmax + 1) - ((order - 1) * order) // 2)
                    * 2
                    * channels
                )
                radial_address = (
                    edge * radial_elements
                    + radial_start
                    + local_row * 2 * channels
                    + output_channel
                )
                radial_value = tl.load(radial_ptr + radial_address, mask=order_mask, other=0.0).to(tl.float32)
                radial_value_direction = tl.load(
                    radial_direction_ptr + radial_address,
                    mask=order_mask,
                    other=0.0,
                ).to(tl.float32)
                for sign in tl.static_range(0, 2):
                    sector_row = local_row + sign * width
                    dense_row = dense_center + (order if sign == 0 else -order)
                    coefficient_address = retained_wigner_offset(
                        edge,
                        dense_row,
                        full,
                        compact_elements,
                        mmax,
                    )
                    coefficient = tl.load(wigner_ptr + coefficient_address, mask=degree >= order, other=0.0).to(tl.float32)
                    coefficient_direction = tl.load(
                        wigner_direction_ptr + coefficient_address,
                        mask=degree >= order,
                        other=0.0,
                    ).to(tl.float32)
                    cotangent = tl.load(
                        grads[order]
                        + edge * list(elements)[order]
                        + sector_row * 2 * channels
                        + output_channel,
                        mask=order_mask,
                        other=0.0,
                    ).to(tl.float32)
                    local += cotangent * (
                        radial_value_direction * coefficient
                        + radial_value * coefficient_direction
                    )
            accumulator += local
    tl.store(
        input_gradient_ptr
        + node * full_components * channels
        + full * channels
        + channel,
        accumulator,
        mask=channel_mask,
    )


def gather_double_backward_source_csr(
    grad_outputs,
    x, edge_index, radial, wigner,
    metadata,
    input_direction,
    radial_direction,
    wigner_direction,
    *,
    lmax,
    mmax,
):
    spec = validate_inputs(x, edge_index, radial, wigner, lmax=lmax, mmax=mmax)
    block_channels = triton.next_power_of_2(spec.channels)
    edges = edge_index.shape[1]
    offsets, sector_elements = sector_metadata(spec)
    sector_grads = tuple(grad_outputs)
    source_order, source_row_ptr, receiver_row_ptr = metadata
    output_directions = tuple(torch.empty_like(value) for value in grad_outputs)
    input_gradient = torch.empty_like(x)
    radial_gradient = torch.empty_like(radial)
    wigner_gradient = torch.zeros_like(wigner)
    prefix_edge_double_backward_kernel[(edges, 2)](
        tuple(sector_grads),
        x, edge_index, radial, wigner,
        input_direction,
        radial_direction,
        wigner_direction,
        tuple(output_directions),
        radial_gradient,
        edges,
        channels=spec.channels,
        full_components=spec.full_components,
        radial_group_rows0=spec.radial_group_rows0,
        radial_group_rows1=spec.radial_group_rows1,
        radial_group_dense_rows0=spec.radial_group_dense_rows0,
        radial_group_dense_rows1=spec.radial_group_dense_rows1,
        radial_group_degrees=spec.radial_group_degrees,
        radial_group_offsets=spec.radial_group_offsets,
        radial_group_count=len(spec.radial_group_rows0),
        radial_elements=spec.radial_elements,
        mmax=spec.mmax,
        offsets=offsets,
        elements=sector_elements,
        block_channels=block_channels,
        compact_elements=wigner.stride(0),
    )
    prefix_input_csr_double_backward_kernel[(x.shape[0], spec.full_components)](
        tuple(sector_grads),
        radial,
        wigner,
        radial_direction,
        wigner_direction,
        source_order,
        source_row_ptr,
        receiver_row_ptr,
        input_gradient,
        channels=spec.channels,
        full_components=spec.full_components,
        lmax=spec.lmax,
        mmax=spec.mmax,
        radial_elements=spec.radial_elements,
        offsets=offsets,
        elements=sector_elements,
        block_channels=block_channels,
        compact_elements=wigner.stride(0),
    )
    for degree in range(spec.lmax + 1):
        retained_rows = 1 + 2 * min(spec.mmax, degree)
        prefix_wigner_degree_double_backward_kernel[(edges, retained_rows)](
            tuple(sector_grads),
            x,
            edge_index,
            radial,
            input_direction,
            radial_direction,
            wigner_gradient,
            edges,
            channels=spec.channels,
            full_components=spec.full_components,
            radial_elements=spec.radial_elements,
            lmax=spec.lmax,
            mmax=spec.mmax,
            degree=degree,
            offsets=offsets,
            elements=sector_elements,
            component_block=triton.next_power_of_2(2 * degree + 1),
            channel_block=triton.next_power_of_2(spec.channels),
            compact_elements=wigner.stride(0),
        )
    return (
        *output_directions,
        input_gradient,
        radial_gradient,
        wigner_gradient,
    )
