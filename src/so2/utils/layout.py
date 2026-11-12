from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class AngularLayout:

    lmax: int
    mmax: int
    channels: int
    full_components: int
    reduced_components: int
    reduced_degrees: tuple[int, ...]
    dense_rows: tuple[int, ...]
    radial_row_offsets: tuple[int, ...]
    radial_group_rows0: tuple[int, ...]
    radial_group_rows1: tuple[int, ...]
    radial_group_dense_rows0: tuple[int, ...]
    radial_group_dense_rows1: tuple[int, ...]
    radial_group_degrees: tuple[int, ...]
    radial_group_offsets: tuple[int, ...]
    sector_widths: tuple[int, ...]
    sector_row_offsets: tuple[int, ...]
    radial_elements: int

    @classmethod
    def create(cls, *, lmax: int, mmax: int, channels: int) -> "AngularLayout":
        if lmax < 0 or not 0 <= mmax <= lmax or channels <= 0:
            raise ValueError(f"invalid prefix dimensions: {lmax=} {mmax=} {channels=}")

        degrees: list[int] = list(range(lmax + 1))
        dense_rows: list[int] = [degree * degree + degree for degree in degrees]
        radial_offsets: list[int] = [degree * 2 * channels for degree in degrees]
        sector_widths = tuple(lmax - order + 1 for order in range(mmax + 1))
        sector_row_offsets = [0]
        radial_sector_offset = (lmax + 1) * 2 * channels
        row_offset = lmax + 1

        for order in range(1, mmax + 1):
            width = lmax - order + 1
            sector_row_offsets.append(row_offset)
            local_offsets = [
                radial_sector_offset + local * 2 * channels for local in range(width)
            ]
            for sign in (1, -1):
                for degree in range(order, lmax + 1):
                    degrees.append(degree)
                    dense_rows.append(degree * degree + degree + sign * order)
                radial_offsets.extend(local_offsets)
            row_offset += 2 * width
            radial_sector_offset += width * 2 * channels

        group_rows0 = list(range(lmax + 1))
        group_rows1 = [-1] * (lmax + 1)
        group_degrees = list(range(lmax + 1))
        group_offsets = [degree * 2 * channels for degree in range(lmax + 1)]
        positive_start = lmax + 1
        radial_sector_offset = (lmax + 1) * 2 * channels
        for order in range(1, mmax + 1):
            width = lmax - order + 1
            negative_start = positive_start + width
            for local, degree in enumerate(range(order, lmax + 1)):
                group_rows0.append(positive_start + local)
                group_rows1.append(negative_start + local)
                group_degrees.append(degree)
                group_offsets.append(radial_sector_offset + local * 2 * channels)
            positive_start += 2 * width
            radial_sector_offset += width * 2 * channels

        return cls(
            lmax=lmax,
            mmax=mmax,
            channels=channels,
            full_components=(lmax + 1) ** 2,
            reduced_components=len(degrees),
            reduced_degrees=tuple(degrees),
            dense_rows=tuple(dense_rows),
            radial_row_offsets=tuple(radial_offsets),
            radial_group_rows0=tuple(group_rows0),
            radial_group_rows1=tuple(group_rows1),
            radial_group_dense_rows0=tuple(dense_rows[row] for row in group_rows0),
            radial_group_dense_rows1=tuple(
                dense_rows[row] if row >= 0 else -1 for row in group_rows1
            ),
            radial_group_degrees=tuple(group_degrees),
            radial_group_offsets=tuple(group_offsets),
            sector_widths=sector_widths,
            sector_row_offsets=tuple(sector_row_offsets),
            radial_elements=radial_sector_offset,
        )


@lru_cache(maxsize=None)
def get_angular_layout(*, lmax: int, mmax: int, channels: int) -> AngularLayout:
    return AngularLayout.create(lmax=lmax, mmax=mmax, channels=channels)


def retained_rotation_size(lmax: int, mmax: int) -> int:
    return sum((2 * min(degree, mmax) + 1) * (2 * degree + 1) for degree in range(lmax + 1))
