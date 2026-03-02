"""Machine, MachineState, MachineTransition dataclasses for WFST representation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MachineTransition:
    """A single transition in a WFST."""
    dest: int
    weight: Any = 1  # JSON weight expression (dict, number, or string)
    input: str | None = None
    output: str | None = None

    @classmethod
    def from_json(cls, j: dict) -> MachineTransition:
        return cls(
            dest=j["to"],
            weight=j.get("weight", 1),
            input=j.get("in"),
            output=j.get("out"),
        )

    def to_json(self) -> dict:
        d: dict[str, Any] = {"to": self.dest}
        if self.input:
            d["in"] = self.input
        if self.output:
            d["out"] = self.output
        if self.weight != 1:
            d["weight"] = self.weight
        return d

    @property
    def is_silent(self) -> bool:
        return not self.input and not self.output


@dataclass
class MachineState:
    """A single state in a WFST."""
    trans: list[MachineTransition] = field(default_factory=list)
    name: Any = None  # JSON StateName

    @classmethod
    def from_json(cls, j: dict) -> MachineState:
        return cls(
            trans=[MachineTransition.from_json(t) for t in j.get("trans", [])],
            name=j.get("id"),
        )

    def to_json(self) -> dict:
        d: dict[str, Any] = {}
        if self.name is not None:
            d["id"] = self.name
        d["trans"] = [t.to_json() for t in self.trans]
        return d


@dataclass
class Machine:
    """A weighted finite-state transducer."""
    state: list[MachineState] = field(default_factory=list)

    @classmethod
    def from_json(cls, j: dict | str) -> Machine:
        if isinstance(j, str):
            j = json.loads(j)
        m = cls(state=[MachineState.from_json(s) for s in j["state"]])
        m._resolve_state_names()
        return m

    def _resolve_state_names(self) -> None:
        """Resolve string state name references in 'dest' to integer indices."""
        # Build name-to-index map (only for hashable names)
        name_to_idx: dict = {}
        for i, s in enumerate(self.state):
            if s.name is not None:
                try:
                    key = tuple(s.name) if isinstance(s.name, list) else s.name
                    name_to_idx[key] = i
                except TypeError:
                    pass  # unhashable name, skip
            name_to_idx[i] = i

        for s in self.state:
            for t in s.trans:
                if not isinstance(t.dest, int):
                    key = tuple(t.dest) if isinstance(t.dest, list) else t.dest
                    if key in name_to_idx:
                        t.dest = name_to_idx[key]
                    else:
                        raise ValueError(f"Unknown state reference: {t.dest}")

    @classmethod
    def from_file(cls, path: str) -> Machine:
        with open(path) as f:
            return cls.from_json(json.load(f))

    def to_json(self) -> dict:
        return {"state": [s.to_json() for s in self.state]}

    def to_json_string(self, indent: int | None = None) -> str:
        return json.dumps(self.to_json(), indent=indent)

    @property
    def n_states(self) -> int:
        return len(self.state)

    @property
    def start_state(self) -> int:
        return 0

    @property
    def end_state(self) -> int:
        return len(self.state) - 1

    def input_alphabet(self) -> list[str]:
        syms = set()
        for s in self.state:
            for t in s.trans:
                if t.input:
                    syms.add(t.input)
        return sorted(syms)

    def output_alphabet(self) -> list[str]:
        syms = set()
        for s in self.state:
            for t in s.trans:
                if t.output:
                    syms.add(t.output)
        return sorted(syms)

    @property
    def n_transitions(self) -> int:
        return sum(len(s.trans) for s in self.state)
