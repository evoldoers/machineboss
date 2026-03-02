"""JSON I/O utilities for machines, params, and seq pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .machine import Machine
from .seqpair import SeqPair


def load_machine(path: str | Path) -> Machine:
    """Load a Machine from a JSON file."""
    with open(path) as f:
        return Machine.from_json(json.load(f))


def save_machine(machine: Machine, path: str | Path, indent: int | None = None) -> None:
    """Save a Machine to a JSON file."""
    with open(path, "w") as f:
        json.dump(machine.to_json(), f, indent=indent)


def load_params(path: str | Path) -> dict[str, float]:
    """Load parameters from a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_params(params: dict[str, float], path: str | Path) -> None:
    """Save parameters to a JSON file."""
    with open(path, "w") as f:
        json.dump(params, f)


def load_seqpair(path: str | Path) -> SeqPair:
    """Load a SeqPair from a JSON file."""
    with open(path) as f:
        return SeqPair.from_json(json.load(f))


def load_seqpair_list(path: str | Path) -> list[SeqPair]:
    """Load a list of SeqPairs from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [SeqPair.from_json(sp) for sp in data]
    return [SeqPair.from_json(data)]
