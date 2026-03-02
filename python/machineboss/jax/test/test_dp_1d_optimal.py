"""Tests for 1D OPTIMAL DP (associative_scan): must match SIMPLE."""

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
from machineboss.jax.dp_1d_simple import forward_1d_simple, backward_1d_simple
from machineboss.jax.dp_1d_optimal import forward_1d_optimal, backward_1d_optimal


class TestOptimal1DForward:
    """OPTIMAL Forward must match SIMPLE Forward."""

    def test_generator_101(self, repo_root, boss_path):
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("101")))
        ll_simple = float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP))
        ll_optimal = float(forward_1d_optimal(jm, None, out_seq, LOGSUMEXP))
        assert ll_optimal == pytest.approx(ll_simple, abs=0.01)

    def test_generator_wrong_output(self, repo_root, boss_path):
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("110")))
        ll_simple = float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP))
        ll_optimal = float(forward_1d_optimal(jm, None, out_seq, LOGSUMEXP))
        # Both should be effectively -inf (impossible path)
        assert ll_optimal < -30
        assert ll_simple < -30

    def test_recognizer_101(self, repo_root, boss_path):
        result = subprocess.run(
            [boss_path, "--recognize-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        ll_simple = float(forward_1d_simple(jm, in_seq, None, LOGSUMEXP))
        ll_optimal = float(forward_1d_optimal(jm, in_seq, None, LOGSUMEXP))
        assert ll_optimal == pytest.approx(ll_simple, abs=0.01)


class TestOptimal1DViterbi:
    """OPTIMAL Viterbi must match SIMPLE Viterbi."""

    def test_generator_viterbi(self, repo_root, boss_path):
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("101")))
        vit_simple = float(forward_1d_simple(jm, None, out_seq, MAXPLUS))
        vit_optimal = float(forward_1d_optimal(jm, None, out_seq, MAXPLUS))
        assert vit_optimal == pytest.approx(vit_simple, abs=0.01)


class TestOptimal1DBackward:
    """OPTIMAL Backward must match SIMPLE Backward."""

    def test_backward_generator(self, repo_root, boss_path):
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("101")))
        bp_simple = backward_1d_simple(jm, None, out_seq, LOGSUMEXP)
        bp_optimal = backward_1d_optimal(jm, None, out_seq, LOGSUMEXP)

        assert float(bp_optimal[0, 0]) == pytest.approx(float(bp_simple[0, 0]), abs=0.01)

    def test_backward_eq_forward(self, repo_root, boss_path):
        """Backward[0, start_state] should equal Forward log-likelihood."""
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("101")))
        ll_fwd = float(forward_1d_optimal(jm, None, out_seq, LOGSUMEXP))
        bp = backward_1d_optimal(jm, None, out_seq, LOGSUMEXP)
        assert float(bp[0, 0]) == pytest.approx(ll_fwd, abs=0.01)


class TestOptimal1DPSWM:
    """OPTIMAL with PSWM should match SIMPLE with PSWM."""

    def test_one_hot_pswm(self, repo_root, boss_path):
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        tokens = jnp.array(em.tokenize_output(list("101")))
        tok_seq = TokenSeq(tokens=tokens)
        n_out = jm.n_output_tokens
        pswm_data = tok_seq.emission_weights(n_out)
        pswm_seq = PSWMSeq(log_probs=pswm_data)

        ll_tok = float(forward_1d_optimal(jm, None, tok_seq, LOGSUMEXP))
        ll_pswm = float(forward_1d_optimal(jm, None, pswm_seq, LOGSUMEXP))
        assert ll_pswm == pytest.approx(ll_tok, abs=0.01)


from .conftest import *
