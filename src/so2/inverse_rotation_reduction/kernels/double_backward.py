from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.sector_layout import load_sector_value, retained_wigner_offset, store_sector_gradient
from .utils import allow_atomic_reduction, logical_rows, validate_messages


@triton.jit
def sector_message_double_backward_kernel(
    grad_output_ptr,
    envelope_ptr, wigner_ptr,
    envelope_direction_ptr, wigner_direction_ptr,
    receivers_ptr,
    message_gradient_ptr,
    message_gradient_stride_edge,
    message_gradient_stride_row,
    message_gradient_stride_channel,
    compact_elements: tl.constexpr,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr,
    order: tl.constexpr,
    mmax: tl.constexpr,
    rows: tl.constexpr,
    block_channels: tl.constexpr,
):
    edge = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    receiver = tl.load(receivers_ptr + edge)
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)
    envelope_direction = tl.load(envelope_direction_ptr + edge).to(tl.float32)
    width = lmax - order + 1
    for row in tl.static_range(0, rows):
        if order == 0:
            degree = row
            dense_row = degree * degree + degree
        elif row < width:
            degree = order + row
            dense_row = degree * degree + degree + order
        else:
            degree = order + row - width
            dense_row = degree * degree + degree - order
        message_gradient = tl.zeros(channel.shape, dtype=tl.float32)
        for local_full in tl.static_range(0, 2 * lmax + 1):
            full_mask = local_full < 2 * degree + 1
            full = degree * degree + local_full
            cotangent = tl.load(
                grad_output_ptr + receiver * full_components * channels + full * channels + channel,
                mask=channel_mask & full_mask,
                other=0.0,
            ).to(tl.float32)
            address = retained_wigner_offset(edge, dense_row, full, compact_elements, mmax)
            coefficient = tl.load(wigner_ptr + address, mask=full_mask, other=0.0).to(tl.float32)
            coefficient_direction = tl.load(
                wigner_direction_ptr + address,
                mask=full_mask,
                other=0.0,
            ).to(tl.float32)
            message_gradient += (
                envelope_direction * coefficient * cotangent
                + envelope * coefficient_direction * cotangent
            )
        store_sector_gradient(
            message_gradient_ptr,
            message_gradient,
            edge,
            row,
            channel,
            message_gradient_stride_edge,
            message_gradient_stride_row,
            message_gradient_stride_channel,
            channels,
            width,
            order,
            channel_mask,
        )


@triton.jit
def receiver_node_double_backward_kernel(
    messages,
    directions,
    envelope_ptr,
    envelope_direction_ptr,
    wigner_ptr,
    wigner_direction_ptr,
    row_ptr,
    output_ptr,
    message_strides: tl.constexpr,
    direction_strides: tl.constexpr,
    compact_elements: tl.constexpr,
    channels: tl.constexpr,
    lmax: tl.constexpr, mmax: tl.constexpr,
    block_channels: tl.constexpr,
):
    receiver = tl.program_id(0)
    full = tl.program_id(1)
    degree = tl.sqrt(full.to(tl.float32)).to(tl.int32)
    center = degree * degree + degree
    channel = tl.arange(0, block_channels)
    mask = channel < channels
    start = tl.load(row_ptr + receiver)
    end = tl.load(row_ptr + receiver + 1)
    total = tl.zeros((block_channels,), tl.float32)
    for edge in range(start, end):
        envelope = tl.load(envelope_ptr + edge)
        envelope_direction = tl.load(envelope_direction_ptr + edge)
        for order in tl.static_range(mmax + 1):
            for sign in tl.static_range(1 if order == 0 else 2):
                width = lmax - order + 1
                row = degree - order + sign * width
                dense_row = center + (order if sign == 0 else -order)
                valid = mask & (degree >= order)
                message = load_sector_value(
                    messages[order],
                    edge,
                    row,
                    channel,
                    list(message_strides)[order][0],
                    list(message_strides)[order][1],
                    list(message_strides)[order][2],
                    channels,
                    width,
                    order,
                    valid,
                )
                direction = load_sector_value(
                    directions[order],
                    edge,
                    row,
                    channel,
                    list(direction_strides)[order][0],
                    list(direction_strides)[order][1],
                    list(direction_strides)[order][2],
                    channels,
                    width,
                    order,
                    valid,
                )
                address = retained_wigner_offset(edge, dense_row, full, compact_elements, mmax)
                coefficient = tl.load(
                    wigner_ptr + address, mask=degree >= order, other=0.0
                )
                coefficient_direction = tl.load(
                    wigner_direction_ptr + address,
                    mask=degree >= order,
                    other=0.0,
                )
                total += (
                    envelope_direction * coefficient * message
                    + envelope * coefficient_direction * message
                    + envelope * coefficient * direction
                )
    tl.store(
        output_ptr + (receiver * (lmax + 1) ** 2 + full) * channels + channel,
        total,
        mask=mask,
    )


@triton.jit
def sector_parameter_double_backward_kernel(
    grad_output_ptr, message_ptr,
    envelope_ptr, wigner_ptr,
    message_direction_ptr, envelope_direction_ptr, wigner_direction_ptr,
    receivers_ptr,
    envelope_gradient_ptr, wigner_gradient_ptr,
    message_stride_edge, message_stride_row, message_stride_channel,
    message_direction_stride_edge,
    message_direction_stride_row,
    message_direction_stride_channel,
    compact_elements: tl.constexpr,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr,
    order: tl.constexpr,
    mmax: tl.constexpr,
    full_block: tl.constexpr,
    channel_block: tl.constexpr,
):
    edge = tl.program_id(0)
    row = tl.program_id(1)
    width = lmax - order + 1
    if order == 0:
        degree = row
        dense_row = degree * degree + degree
    else:
        positive = row < width
        degree = order + tl.where(positive, row, row - width)
        dense_row = degree * degree + degree + tl.where(positive, order, -order)
    local_full_1d = tl.arange(0, full_block)
    local_full = local_full_1d[:, None]
    full_mask_1d = local_full_1d < 2 * degree + 1
    full_mask = full_mask_1d[:, None]
    channel = tl.arange(0, channel_block)[None, :]
    channel_mask = channel < channels
    value_mask = full_mask & channel_mask
    receiver = tl.load(receivers_ptr + edge)
    full = degree * degree + local_full
    message = load_sector_value(
        message_ptr, edge, row, channel,
        message_stride_edge, message_stride_row, message_stride_channel,
        channels, width, order, channel_mask,
    )
    message_direction = load_sector_value(
        message_direction_ptr,
        edge,
        row,
        channel,
        message_direction_stride_edge,
        message_direction_stride_row,
        message_direction_stride_channel,
        channels,
        width,
        order,
        channel_mask,
    )
    cotangent = tl.load(
        grad_output_ptr + receiver * full_components * channels + full * channels + channel,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)
    envelope_direction = tl.load(envelope_direction_ptr + edge).to(tl.float32)
    channel_dot = tl.sum(cotangent * message, axis=1)
    channel_dot_direction = tl.sum(cotangent * message_direction, axis=1)
    wigner_address = retained_wigner_offset(
        edge, dense_row, degree * degree + local_full_1d, compact_elements, mmax,
    )
    tl.store(
        wigner_gradient_ptr + wigner_address,
        envelope_direction * channel_dot + envelope * channel_dot_direction,
        mask=full_mask_1d,
    )
    coefficient = tl.load(
        wigner_ptr + wigner_address,
        mask=full_mask_1d,
        other=0.0,
    ).to(tl.float32)
    coefficient_direction = tl.load(
        wigner_direction_ptr + wigner_address,
        mask=full_mask_1d,
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        envelope_gradient_ptr + edge,
        tl.sum(
            coefficient_direction * channel_dot + coefficient * channel_dot_direction,
            axis=0,
        ),
    )


def sum_double_backward(
    grad_output,
    messages, envelope, wigner,
    receivers, row_ptr,
    envelope_direction, wigner_direction, message_directions,
    *,
    lmax, mmax,
):
    allow_atomic_reduction("dataflow/edge_to_node/sum_aggregation/derivatives.double_backward")
    edges = envelope.shape[0]
    spec = validate_messages(messages, edges, lmax, mmax)
    nodes = row_ptr.numel() - 1
    grad_output_gradient = torch.empty_like(grad_output)
    message_gradients = [torch.empty_like(message) for message in messages]
    envelope_gradient = torch.zeros_like(envelope)
    wigner_gradient = torch.zeros_like(wigner)
    for order, (message, message_direction, message_gradient) in enumerate(
        zip(messages, message_directions, message_gradients, strict=True)
    ):
        rows = logical_rows(spec, order)
        sector_message_double_backward_kernel[(edges,)](
            grad_output, envelope, wigner,
            envelope_direction, wigner_direction,
            receivers, message_gradient,
            *message_gradient.stride(), wigner.shape[1],
            channels=spec.channels,
            full_components=spec.full_components,
            lmax=lmax, mmax=mmax, order=order, rows=rows,
            block_channels=triton.next_power_of_2(spec.channels),
        )
        sector_parameter_double_backward_kernel[(edges, rows)](
            grad_output, message,
            envelope, wigner,
            message_direction, envelope_direction, wigner_direction,
            receivers,
            envelope_gradient, wigner_gradient,
            *message.stride(), *message_direction.stride(), wigner.shape[1],
            channels=spec.channels,
            full_components=spec.full_components,
            lmax=lmax, mmax=mmax, order=order,
            full_block=triton.next_power_of_2(2 * lmax + 1),
            channel_block=triton.next_power_of_2(spec.channels),
        )
    receiver_node_double_backward_kernel[(nodes, spec.full_components)](
        tuple(messages), tuple(message_directions),
        envelope, envelope_direction, wigner, wigner_direction,
        row_ptr, grad_output_gradient,
        message_strides=tuple(x.stride() for x in messages),
        direction_strides=tuple(x.stride() for x in message_directions),
        compact_elements=wigner.shape[1],
        channels=spec.channels,
        lmax=lmax, mmax=mmax,
        block_channels=triton.next_power_of_2(spec.channels),
    )
    return grad_output_gradient, envelope_gradient, wigner_gradient, *message_gradients
