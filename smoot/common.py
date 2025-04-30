from __future__ import annotations

from enum import Flag


class UnitFormat(Flag):
    default = 0
    compact = 1
    without_spaces = 1 << 1
    with_scaling_factor = 1 << 2

    @classmethod
    def from_format_spec(cls, format_spec: str) -> UnitFormat:
        fmt = UnitFormat.default
        for c in format_spec:
            if c == "~":
                fmt |= UnitFormat.compact
            elif c == "C":
                fmt |= UnitFormat.without_spaces
            else:
                msg = f"Unknown symbol '{c}' in format spec '{format_spec}'"
                raise ValueError(msg)
        return fmt
