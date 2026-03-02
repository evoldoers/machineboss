"""Tests for JAX Viterbi algorithm."""

import json
import subprocess

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.viterbi import log_viterbi


class TestViterbiBitecho:
    """Test Viterbi on bitecho (identity transducer)."""

    def test_matching_sequences(self, repo_root, boss_path):
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("101")))
        ll = float(log_viterbi(jm, in_seq, out_seq))

        # For bitecho with matching input/output, Viterbi = Forward = 0
        assert ll == pytest.approx(0.0, abs=0.01)


class TestViterbiBitnoise:
    """Test Viterbi on bitnoise."""

    def test_viterbi_le_forward(self, repo_root, boss_path):
        """Viterbi log-prob should be <= Forward log-prob."""
        machine_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        m = Machine.from_file(machine_path)
        with open(params_path) as f:
            params = json.load(f)
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("001")))

        from machineboss.jax.forward import log_forward
        fwd_ll = float(log_forward(jm, in_seq, out_seq))
        vit_ll = float(log_viterbi(jm, in_seq, out_seq))

        # Viterbi (max) <= Forward (sum)
        assert vit_ll <= fwd_ll + 0.01
