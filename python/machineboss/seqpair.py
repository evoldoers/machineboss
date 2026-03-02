"""SeqPair and Envelope types for sequence pair I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SeqPair:
    """A pair of input/output sequences for DP algorithms."""
    input: list[str] = field(default_factory=list)
    output: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, j: dict) -> SeqPair:
        return cls(
            input=j.get("input", []),
            output=j.get("output", []),
        )

    def to_json(self) -> dict:
        d: dict[str, Any] = {}
        if self.input:
            d["input"] = self.input
        if self.output:
            d["output"] = self.output
        return d

    @classmethod
    def from_strings(cls, input_str: str = "", output_str: str = "") -> SeqPair:
        return cls(
            input=list(input_str) if input_str else [],
            output=list(output_str) if output_str else [],
        )


@dataclass
class Envelope:
    """DP envelope constraining the (input_pos, output_pos) pairs to compute."""
    input_start: int = 0
    input_end: int = 0
    output_start: int = 0
    output_end: int = 0
