"""Tests for sparse kernel: sparse results must match dense."""

import json
import subprocess

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine, NEG_INF
from machineboss.jax.semiring import LOGSUMEXP, MAXPLUS
from machineboss.jax.seq import TokenSeq, PSWMSeq
from machineboss.jax.dp_2d_simple import (
    forward_2d_dense, backward_2d_dense,
    forward_2d_sparse, backward_2d_sparse,
)
from machineboss.jax.dp_1d_simple import forward_1d_simple, backward_1d_simple


class TestSparse2DForward:
    """Sparse 2D Forward should match dense."""

    def test_bitecho_forward(self, repo_root, boss_path):
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("101")))

        ll_dense = float(forward_2d_dense(jm, in_seq, out_seq, LOGSUMEXP))
        ll_sparse = float(forward_2d_sparse(jm, in_seq, out_seq, LOGSUMEXP))
        assert ll_sparse == pytest.approx(ll_dense, abs=0.01)

    def test_bitnoise_forward(self, repo_root, boss_path):
        machine_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        m = Machine.from_file(machine_path)
        with open(params_path) as f:
            params = json.load(f)
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("001")))

        ll_dense = float(forward_2d_dense(jm, in_seq, out_seq, LOGSUMEXP))
        ll_sparse = float(forward_2d_sparse(jm, in_seq, out_seq, LOGSUMEXP))
        assert ll_sparse == pytest.approx(ll_dense, abs=0.01)


class TestSparse2DViterbi:
    """Sparse 2D Viterbi should match dense."""

    def test_bitecho_viterbi(self, repo_root, boss_path):
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("101")))

        vit_dense = float(forward_2d_dense(jm, in_seq, out_seq, MAXPLUS))
        vit_sparse = float(forward_2d_sparse(jm, in_seq, out_seq, MAXPLUS))
        assert vit_sparse == pytest.approx(vit_dense, abs=0.01)


class TestSparse2DBackward:
    """Sparse 2D Backward should match dense."""

    def test_bitecho_backward(self, repo_root, boss_path):
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("101")))

        bp_dense = backward_2d_dense(jm, in_seq, out_seq, LOGSUMEXP)
        bp_sparse = backward_2d_sparse(jm, in_seq, out_seq, LOGSUMEXP)

        # Compare at start state position (0, 0, 0) = log-likelihood
        assert float(bp_sparse[0, 0, 0]) == pytest.approx(float(bp_dense[0, 0, 0]), abs=0.01)


class TestSparse1D:
    """Sparse 1D should match dense."""

    def test_generator_forward(self, repo_root, boss_path):
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("101")))

        ll_dense = float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP, kernel='dense'))
        ll_sparse = float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP, kernel='sparse'))
        assert ll_sparse == pytest.approx(ll_dense, abs=0.01)

    def test_generator_backward(self, repo_root, boss_path):
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("101")))

        bp_dense = backward_1d_simple(jm, None, out_seq, LOGSUMEXP, kernel='dense')
        bp_sparse = backward_1d_simple(jm, None, out_seq, LOGSUMEXP, kernel='sparse')
        assert float(bp_sparse[0, 0]) == pytest.approx(float(bp_dense[0, 0]), abs=0.01)

    def test_generator_viterbi(self, repo_root, boss_path):
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("101")))

        vit_dense = float(forward_1d_simple(jm, None, out_seq, MAXPLUS, kernel='dense'))
        vit_sparse = float(forward_1d_simple(jm, None, out_seq, MAXPLUS, kernel='sparse'))
        assert vit_sparse == pytest.approx(vit_dense, abs=0.01)


from .conftest import *
