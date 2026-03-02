"""Tests for 2D OPTIMAL DP (diagonal wavefront): must match SIMPLE."""

import json
import subprocess

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.semiring import LOGSUMEXP, MAXPLUS
from machineboss.jax.seq import TokenSeq, PSWMSeq
from machineboss.jax.dp_2d_simple import forward_2d_dense, backward_2d_dense
from machineboss.jax.dp_2d_optimal import forward_2d_optimal, backward_2d_optimal


class TestOptimal2DForward:
    """OPTIMAL 2D Forward must match SIMPLE."""

    def test_bitecho_matching(self, repo_root, boss_path):
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("101")))

        ll_simple = float(forward_2d_dense(jm, in_seq, out_seq, LOGSUMEXP))
        ll_optimal = float(forward_2d_optimal(jm, in_seq, out_seq, LOGSUMEXP))
        assert ll_optimal == pytest.approx(ll_simple, abs=0.01)

    def test_bitnoise(self, repo_root, boss_path):
        machine_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        m = Machine.from_file(machine_path)
        with open(params_path) as f:
            params = json.load(f)
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("001")))

        ll_simple = float(forward_2d_dense(jm, in_seq, out_seq, LOGSUMEXP))
        ll_optimal = float(forward_2d_optimal(jm, in_seq, out_seq, LOGSUMEXP))
        assert ll_optimal == pytest.approx(ll_simple, abs=0.01)


class TestOptimal2DViterbi:
    """OPTIMAL 2D Viterbi must match SIMPLE."""

    def test_bitecho_viterbi(self, repo_root, boss_path):
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("101")))

        vit_simple = float(forward_2d_dense(jm, in_seq, out_seq, MAXPLUS))
        vit_optimal = float(forward_2d_optimal(jm, in_seq, out_seq, MAXPLUS))
        assert vit_optimal == pytest.approx(vit_simple, abs=0.01)


class TestOptimal2DBackward:
    """OPTIMAL 2D Backward must match SIMPLE."""

    def test_bitecho_backward(self, repo_root, boss_path):
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("101")))

        bp_simple = backward_2d_dense(jm, in_seq, out_seq, LOGSUMEXP)
        bp_optimal = backward_2d_optimal(jm, in_seq, out_seq, LOGSUMEXP)

        assert float(bp_optimal[0, 0, 0]) == pytest.approx(float(bp_simple[0, 0, 0]), abs=0.01)

    def test_backward_eq_forward(self, repo_root, boss_path):
        """Backward[0,0,start] should equal Forward log-likelihood."""
        machine_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        m = Machine.from_file(machine_path)
        with open(params_path) as f:
            params = json.load(f)
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("001")))

        ll_fwd = float(forward_2d_optimal(jm, in_seq, out_seq, LOGSUMEXP))
        bp = backward_2d_optimal(jm, in_seq, out_seq, LOGSUMEXP)
        assert float(bp[0, 0, 0]) == pytest.approx(ll_fwd, abs=0.01)


from .conftest import *
