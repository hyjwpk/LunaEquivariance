from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ...utils.sector_layout import load_sector_value, retained_wigner_offset


@triton.jit
def receiver_forward_kernel(
    messages,
    envelope_ptr, wigner_ptr,
    row_ptr, output_ptr,
    message_strides,
    compact_elements: tl.constexpr,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr, mmax: tl.constexpr,
    block_channels: tl.constexpr,
):
    receiver = tl.program_id(0)
    full = tl.program_id(1)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    degree = tl.sqrt(full.to(tl.float32)).to(tl.int32)
    dense_center = degree * degree + degree

    start = tl.load(row_ptr + receiver).to(tl.int32)
    end = tl.load(row_ptr + receiver + 1).to(tl.int32)
    accumulator = tl.zeros(channel.shape, dtype=tl.float32)
    for edge in range(start, end):
        envelope = tl.load(envelope_ptr + edge).to(tl.float32)
        message0 = load_sector_value(
            messages[0], edge, degree, channel,
            message_strides[0][0], message_strides[0][1], message_strides[0][2],
            channels, lmax + 1, 0, channel_mask,
        )
        coefficient0 = tl.load(
            wigner_ptr
            + retained_wigner_offset(edge, dense_center, full, compact_elements, mmax)
        ).to(tl.float32)
        local = coefficient0 * message0

        for order in tl.static_range(1, mmax + 1):
            m_mask = channel_mask & (degree >= order)
            m_row = degree - order
            m_width = tl.constexpr(lmax - order + 1)
            message_positive = load_sector_value(
                messages[order], edge, m_row, channel,
                message_strides[order][0], message_strides[order][1], message_strides[order][2],
                channels, m_width, order, m_mask,
            )
            message_negative = load_sector_value(
                messages[order], edge, m_width + m_row, channel,
                message_strides[order][0], message_strides[order][1], message_strides[order][2],
                channels, m_width, order, m_mask,
            )
            coefficient_positive = tl.load(
                wigner_ptr
                + retained_wigner_offset(
                    edge, dense_center + order, full, compact_elements, mmax,
                ),
                mask=degree >= order,
                other=0.0,
            ).to(tl.float32)
            coefficient_negative = tl.load(
                wigner_ptr
                + retained_wigner_offset(
                    edge, dense_center - order, full, compact_elements, mmax,
                ),
                mask=degree >= order,
                other=0.0,
            ).to(tl.float32)
            local += (
                coefficient_positive * message_positive
                + coefficient_negative * message_negative
            )

        accumulator += envelope * local
    tl.store(
        output_ptr + (receiver * full_components + full) * channels + channel,
        accumulator,
        mask=channel_mask,
    )


@triton_op("so2::compact_receiver_tail_forward", mutates_args=("output",))
def receiver_sector_tail_forward(
    messages: list[torch.Tensor],
    envelope: torch.Tensor,
    wigner: torch.Tensor,
    row_ptr: torch.Tensor,
    output: torch.Tensor,
    lmax: int,
    mmax: int,
) -> None:
    nodes, components, channels = output.shape
    wrap_triton(receiver_forward_kernel)[(nodes, components)](
        tuple(messages), envelope, wigner,
        row_ptr, output,
        tuple(message.stride() for message in messages),
        wigner.shape[1],
        channels=channels,
        full_components=components,
        lmax=lmax,
        mmax=mmax,
        block_channels=triton.next_power_of_2(channels),
    )
