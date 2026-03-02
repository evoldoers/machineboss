"""Tests for JAX Forward algorithm against C++ boss output."""

import json
import subprocess
import tempfile
import os

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.forward import log_forward


def _boss_loglike(boss_path, machine_file, input_str=None, output_str=None, params_file=None):
    """Get log-likelihood from bin/boss."""
    args = [boss_path, machine_file]
    if input_str:
        args += ["--input-chars", input_str]
    if output_str:
        args += ["--output-chars", output_str]
    if params_file:
        args += ["-P", params_file]
    args.append("-L")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"boss failed: {result.stderr}")
    data = json.loads(result.stdout)
    # boss -L returns [[input, output, loglike]] or just a number
    if isinstance(data, list):
        return float(data[0][-1]) if isinstance(data[0], list) else float(data[0])
    return float(data)


class TestForwardBitecho:
    """Test Forward on bitecho machine (identity transducer for {0,1})."""

    def test_matching_sequences(self, repo_root, boss_path):
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        # Input=output=101 should have loglike = 0 (prob 1)
        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("101")))
        ll = float(log_forward(jm, in_seq, out_seq))

        boss_ll = _boss_loglike(boss_path, machine_path,
                                input_str="101", output_str="101")
        assert ll == pytest.approx(boss_ll, abs=0.01)

    def test_mismatching_sequences(self, repo_root, boss_path):
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        # Input=101, output=001 should have very low probability
        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("001")))
        ll = float(log_forward(jm, in_seq, out_seq))
        assert ll < -30  # essentially zero probability


class TestForwardBitnoise:
    """Test Forward on bitnoise machine (noisy channel for {0,1})."""

    def test_loglike_matches_boss(self, repo_root, boss_path):
        machine_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        m = Machine.from_file(machine_path)
        with open(params_path) as f:
            params = json.load(f)
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("001")))
        ll = float(log_forward(jm, in_seq, out_seq))

        boss_ll = _boss_loglike(boss_path, machine_path,
                                input_str="101", output_str="001",
                                params_file=params_path)
        assert ll == pytest.approx(boss_ll, abs=0.01)


class TestForwardGenerator:
    """Test Forward on a simple generator (no input)."""

    def test_generator_loglike(self, repo_root, boss_path):
        # Generate "101" generator machine
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        # No input, output=101
        in_seq = jnp.array([], dtype=jnp.int32)
        out_seq = jnp.array(em.tokenize_output(list("101")))
        ll = float(log_forward(jm, in_seq, out_seq))

        # Should be 0 (prob 1) for exact match
        assert ll == pytest.approx(0.0, abs=0.01)
