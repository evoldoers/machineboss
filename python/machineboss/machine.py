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
    defs: dict[str, Any] = field(default_factory=dict)  # parameter/function definitions

    @classmethod
    def from_json(cls, j: dict | str) -> Machine:
        if isinstance(j, str):
            j = json.loads(j)
        m = cls(
            state=[MachineState.from_json(s) for s in j["state"]],
            defs=j.get("defs", {}),
        )
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
        d: dict[str, Any] = {"state": [s.to_json() for s in self.state]}
        if self.defs:
            d["defs"] = self.defs
        return d

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

    def merge_equivalent_states(self) -> Machine:
        """Merge states with identical outgoing transitions (collapse bubbles)."""
        current = self
        while True:
            n_old = current.n_states
            # Step 1: Merge parallel transitions
            for s in current.state:
                merged: dict[tuple, Any] = {}
                for t in s.trans:
                    key = (t.dest, t.input or "", t.output or "")
                    if key in merged:
                        merged[key] = {"+" : [merged[key], t.weight]}
                    else:
                        merged[key] = t.weight
                s.trans = [
                    MachineTransition(
                        dest=k[0],
                        input=k[1] or None,
                        output=k[2] or None,
                        weight=w,
                    )
                    for k, w in merged.items()
                ]
            # Step 2: Compute signatures and group states
            import json as _json
            sig_groups: dict[str, list[int]] = {}
            for i, s in enumerate(current.state):
                sig = sorted(
                    (_json.dumps(t.dest), t.input or "", t.output or "", _json.dumps(t.weight))
                    for t in s.trans
                )
                key = _json.dumps(sig)
                sig_groups.setdefault(key, []).append(i)
            redirect: dict[int, int] = {}
            for states in sig_groups.values():
                if len(states) > 1:
                    rep = states[0]
                    for si in states:
                        if si == current.start_state or si == current.end_state:
                            rep = si
                            break
                    for si in states:
                        if si != rep:
                            redirect[si] = rep
            if not redirect:
                break
            # Step 3: Redirect transitions
            for s in current.state:
                for t in s.trans:
                    if t.dest in redirect:
                        t.dest = redirect[t.dest]
            # Step 4: Remove unreachable states
            reachable = set()
            queue = [current.start_state]
            reachable.add(current.start_state)
            while queue:
                si = queue.pop()
                for t in current.state[si].trans:
                    if t.dest not in reachable:
                        reachable.add(t.dest)
                        queue.append(t.dest)
            if current.end_state not in reachable:
                break
            old2new: dict[int, int] = {}
            new_states: list[MachineState] = []
            for i in range(current.n_states):
                if i in reachable:
                    old2new[i] = len(new_states)
                    new_states.append(current.state[i])
            for s in new_states:
                for t in s.trans:
                    t.dest = old2new[t.dest]
            current = Machine(state=new_states, defs=current.defs)
            if current.n_states == n_old:
                break
        # Final parallel-transition merge
        for s in current.state:
            merged_f: dict[tuple, Any] = {}
            for t in s.trans:
                key = (t.dest, t.input or "", t.output or "")
                if key in merged_f:
                    merged_f[key] = {"+" : [merged_f[key], t.weight]}
                else:
                    merged_f[key] = t.weight
            s.trans = [
                MachineTransition(
                    dest=k[0],
                    input=k[1] or None,
                    output=k[2] or None,
                    weight=w,
                )
                for k, w in merged_f.items()
            ]
        return current
