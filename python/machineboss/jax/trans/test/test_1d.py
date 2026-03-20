"""Tests for 1D DP algorithms on TransMachine."""

import json
import subprocess

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.semiring import LOGSUMEXP, MAXPLUS
from machineboss.jax.forward import log_forward
from machineboss.jax.backward import log_backward_matrix
from machineboss.jax.trans.machine import TransMachine
from machineboss.jax.trans.dp_1d import forward_1d, backward_1d, viterbi_1d


def _load_generator(repo_root):
    """Load a generator machine (output-only)."""
    path = repo_root / "t" / "machine" / "merge-chain.json"
    if not path.exists():
        return None, None, None
    m = Machine.from_file(str(path))
    em = EvaluatedMachine.from_machine(m)
    tm = TransMachine.from_evaluated(em)
    jm = JAXMachine.from_evaluated(em)
    return tm, jm, em


class TestForward1DSimple:
    """Test 1D forward (simple strategy) cross-validates with existing."""

    def test_generator_matches_existing(self, repo_root):
        tm, jm, em = _load_generator(repo_root)
        if tm is None:
            pytest.skip("bitcompose.json not found")

        out_seq = jnp.array(em.tokenize_output(list("qqq")))

        result_trans = float(forward_1d(tm, output_seq=out_seq, strategy='simple'))
        result_existing = float(log_forward(jm, output_seq=out_seq))

        assert result_trans == pytest.approx(result_existing, abs=1e-4)


class TestForward1DOptimal:
    """Test 1D forward (optimal strategy) cross-validates."""

    def test_generator_matches_simple(self, repo_root):
        tm, jm, em = _load_generator(repo_root)
        if tm is None:
            pytest.skip("bitcompose.json not found")

        out_seq = jnp.array(em.tokenize_output(list("qqq")))

        result_simple = float(forward_1d(tm, output_seq=out_seq, strategy='simple'))
        result_optimal = float(forward_1d(tm, output_seq=out_seq, strategy='optimal'))

        assert result_optimal == pytest.approx(result_simple, abs=1e-4)


class TestBackward1D:
    """Test 1D backward cross-validates."""

    def test_backward_matches_existing(self, repo_root):
        tm, jm, em = _load_generator(repo_root)
        if tm is None:
            pytest.skip("bitcompose.json not found")

        out_seq = jnp.array(em.tokenize_output(list("qq")))

        bp_trans = backward_1d(tm, output_seq=out_seq, strategy='simple')
        bp_existing = log_backward_matrix(jm, output_seq=out_seq)

        assert jnp.allclose(bp_trans, bp_existing, atol=1e-4)


class TestViterbi1D:
    """Test 1D Viterbi."""

    def test_viterbi_le_forward(self, repo_root):
        tm, jm, em = _load_generator(repo_root)
        if tm is None:
            pytest.skip("bitcompose.json not found")

        out_seq = jnp.array(em.tokenize_output(list("qqq")))

        fwd = float(forward_1d(tm, output_seq=out_seq))
        vit = float(viterbi_1d(tm, output_seq=out_seq))

        # Viterbi <= Forward (max path <= sum of all paths)
        assert vit <= fwd + 1e-5
