"""EvaluatedMachine: tokenize alphabets and evaluate transition weights.

Bridge between JSON machines and numerical DP algorithms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .machine import Machine
from .weight import evaluate as eval_weight


@dataclass
class EvaluatedTransition:
    """A transition with evaluated log-weight and token indices."""
    src: int
    dst: int
    in_tok: int  # 0 = empty
    out_tok: int  # 0 = empty
    log_weight: float


@dataclass
class EvaluatedMachine:
    """Machine with tokenized alphabets and numerically evaluated weights."""
    n_states: int
    input_tokens: list[str]   # index 0 = empty token
    output_tokens: list[str]  # index 0 = empty token
    transitions: list[EvaluatedTransition] = field(default_factory=list)

    # Indexed lookups
    _by_src: dict[int, list[int]] | None = field(default=None, repr=False)

    @classmethod
    def from_machine(cls, machine: Machine, params: dict[str, float] | None = None) -> EvaluatedMachine:
        """Create an EvaluatedMachine from a Machine with optional parameters."""
        if params is None:
            params = {}

        in_alpha = machine.input_alphabet()
        out_alpha = machine.output_alphabet()

        # Token maps: 0 = empty, 1..N = alphabet symbols
        in_tok_map: dict[str, int] = {sym: i + 1 for i, sym in enumerate(in_alpha)}
        out_tok_map: dict[str, int] = {sym: i + 1 for i, sym in enumerate(out_alpha)}

        transitions = []
        for src, state in enumerate(machine.state):
            for t in state.trans:
                w = eval_weight(t.weight, params)
                if w <= 0:
                    continue  # skip zero-weight transitions
                transitions.append(EvaluatedTransition(
                    src=src,
                    dst=t.dest,
                    in_tok=in_tok_map.get(t.input, 0) if t.input else 0,
                    out_tok=out_tok_map.get(t.output, 0) if t.output else 0,
                    log_weight=math.log(w),
                ))

        em = cls(
            n_states=machine.n_states,
            input_tokens=[""] + in_alpha,
            output_tokens=[""] + out_alpha,
            transitions=transitions,
        )
        em._build_index()
        return em

    def _build_index(self) -> None:
        self._by_src = {}
        for i, t in enumerate(self.transitions):
            self._by_src.setdefault(t.src, []).append(i)

    def transitions_from(self, src: int) -> list[EvaluatedTransition]:
        if self._by_src is None:
            self._build_index()
        return [self.transitions[i] for i in self._by_src.get(src, [])]

    def tokenize_input(self, seq: list[str]) -> list[int]:
        tok_map = {sym: i for i, sym in enumerate(self.input_tokens)}
        return [tok_map[s] for s in seq]

    def tokenize_output(self, seq: list[str]) -> list[int]:
        tok_map = {sym: i for i, sym in enumerate(self.output_tokens)}
        return [tok_map[s] for s in seq]
