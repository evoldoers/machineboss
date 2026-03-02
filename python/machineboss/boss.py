"""Subprocess wrapper for the bin/boss CLI."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .machine import Machine


def _find_boss() -> str:
    """Find the boss binary, checking common locations."""
    # Check relative to this file (repo layout)
    repo_root = Path(__file__).parent.parent.parent
    candidate = repo_root / "bin" / "boss"
    if candidate.is_file():
        return str(candidate)
    # Check PATH
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = Path(d) / "boss"
        if p.is_file():
            return str(p)
    raise FileNotFoundError("Cannot find bin/boss executable")


class Boss:
    """Wrapper around the bin/boss CLI."""

    def __init__(self, executable: str | None = None):
        self.executable = executable or _find_boss()

    def run(self, *args: str, input_json: Any = None, timeout: float = 60) -> str:
        """Run boss with given args, return stdout as string.

        If input_json is provided, it is written to a temp file and passed as an argument.
        """
        cmd = [self.executable] + list(args)
        stdin_data = None
        if input_json is not None:
            stdin_data = json.dumps(input_json) if not isinstance(input_json, str) else input_json

        result = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        return result.stdout

    def run_json(self, *args: str, input_json: Any = None, timeout: float = 60) -> Any:
        """Run boss and parse output as JSON."""
        out = self.run(*args, input_json=input_json, timeout=timeout)
        return json.loads(out)

    def load_machine(self, *args: str) -> Machine:
        """Run boss and parse output as a Machine."""
        return Machine.from_json(self.run_json(*args))

    def compose(self, m1: Machine, m2: Machine) -> Machine:
        """Compose two machines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            try:
                json.dump(m1.to_json(), f1)
                f1.flush()
                json.dump(m2.to_json(), f2)
                f2.flush()
                return self.load_machine(f1.name, f2.name)
            finally:
                os.unlink(f1.name)
                os.unlink(f2.name)

    def forward(self, machine: Machine, input_seq: list[str] | None = None,
                output_seq: list[str] | None = None,
                params: dict[str, float] | None = None) -> float:
        """Run forward algorithm, return log-likelihood."""
        args: list[str] = []
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as mf:
            try:
                json.dump(machine.to_json(), mf)
                mf.flush()
                args.append(mf.name)
                if input_seq is not None:
                    args.extend(["--input-chars", "".join(input_seq)])
                if output_seq is not None:
                    args.extend(["--output-chars", "".join(output_seq)])
                if params is not None:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
                        json.dump(params, pf)
                        pf.flush()
                        args.extend(["-P", pf.name])
                        result = self.run_json(*args, "-L")
                        os.unlink(pf.name)
                else:
                    result = self.run_json(*args, "-L")
                return float(result)
            finally:
                os.unlink(mf.name)
