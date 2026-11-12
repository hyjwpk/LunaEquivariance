from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.sector_layout import load_sector_value, retained_wigner_offset, store_sector_gradient
from .utils import allow_atomic_reduction, logical_rows, validate_messages


@triton.jit
def sector_message_backward_kernel(
    grad_output_ptr,
    envelope_ptr, wigner_ptr,
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
        accumulator = tl.zeros(channel.shape, dtype=tl.float32)
        for local_full in tl.static_range(0, 2 * lmax + 1):
            full_mask = local_full < 2 * degree + 1
            full = degree * degree + local_full
            cotangent = tl.load(
                grad_output_ptr + receiver * full_components * channels + full * channels + channel,
                mask=channel_mask & full_mask,
                other=0.0,
            ).to(tl.float32)
            address = retained_wigner_offset(edge, dense_row, full, compact_elements, mmax)
            coefficient = tl.load(
                wigner_ptr + address,
                mask=full_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += coefficient * cotangent
        store_sector_gradient(
            message_gradient_ptr,
            envelope * accumulator,
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
def sector_parameter_backward_kernel(
    grad_output_ptr, message_ptr,
    envelope_ptr, wigner_ptr,
    receivers_ptr,
    envelope_gradient_ptr, wigner_gradient_ptr,
    message_stride_edge, message_stride_row, message_stride_channel,
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
    cotangent = tl.load(
        grad_output_ptr + receiver * full_components * channels + full * channels + channel,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    channel_dot = tl.sum(cotangent * message, axis=1)
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)
    wigner_address = retained_wigner_offset(
        edge, dense_row, degree * degree + local_full_1d, compact_elements, mmax,
    )
    tl.store(
        wigner_gradient_ptr + wigner_address,
        envelope * channel_dot,
        mask=full_mask_1d,
    )
    coefficient = tl.load(
        wigner_ptr + wigner_address,
        mask=full_mask_1d,
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        envelope_gradient_ptr + edge,
        tl.sum(coefficient * channel_dot, axis=0),
    )


def sum_backward(
    grad_output,
    messages, envelope, wigner,
    receivers,
    *,
    lmax, mmax,
):
    allow_atomic_reduction("dataflow/edge_to_node/sum_aggregation/derivatives.backward")
    spec = validate_messages(messages, envelope.shape[0], lmax, mmax)
    edges = envelope.shape[0]
    message_gradients = [torch.empty_like(message) for message in messages]
    envelope_gradient = torch.zeros_like(envelope)
    wigner_gradient = torch.zeros_like(wigner)
    for order, (message, message_gradient) in enumerate(
        zip(messages, message_gradients, strict=True)
    ):
        rows = logical_rows(spec, order)
        sector_message_backward_kernel[(edges,)](
            grad_output,
            envelope,
            wigner,
            receivers,
            message_gradient,
            *message_gradient.stride(), wigner.shape[1],
            channels=spec.channels,
            full_components=spec.full_components,
            lmax=spec.lmax,
            order=order, mmax=mmax, rows=rows,
            block_channels=triton.next_power_of_2(spec.channels),
        )
        sector_parameter_backward_kernel[(edges, rows)](
            grad_output, message,
            envelope, wigner,
            receivers,
            envelope_gradient, wigner_gradient,
            *message.stride(), wigner.shape[1],
            channels=spec.channels,
            full_components=spec.full_components,
            lmax=spec.lmax,
            order=order, mmax=mmax,
            full_block=triton.next_power_of_2(2 * lmax + 1),
            channel_block=triton.next_power_of_2(spec.channels),
        )
    return envelope_gradient, wigner_gradient, *message_gradients
