"""Tests for PSWM sequence support in 2D DP."""

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
from machineboss.jax.dp_2d_simple import forward_2d_dense, backward_2d_dense
from machineboss.jax.dp_2d_optimal import forward_2d_optimal


class TestPSWM2D:
    """Test PSWM in 2D DP."""

    def test_one_hot_pswm_eq_tok_2d(self, repo_root, boss_path):
        """One-hot PSWM should give same result as TokenSeq in 2D."""
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_tokens = jnp.array(em.tokenize_input(list("101")))
        out_tokens = jnp.array(em.tokenize_output(list("101")))

        in_tok_seq = TokenSeq(tokens=in_tokens)
        out_tok_seq = TokenSeq(tokens=out_tokens)

        in_pswm = PSWMSeq(log_probs=in_tok_seq.emission_weights(jm.n_input_tokens))
        out_pswm = PSWMSeq(log_probs=out_tok_seq.emission_weights(jm.n_output_tokens))

        ll_tok = float(forward_2d_dense(jm, in_tok_seq, out_tok_seq, LOGSUMEXP))
        ll_pswm = float(forward_2d_dense(jm, in_pswm, out_pswm, LOGSUMEXP))
        assert ll_pswm == pytest.approx(ll_tok, abs=0.01)

    def test_pswm_input_tok_output(self, repo_root, boss_path):
        """PSWM input with token output should work."""
        machine_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        m = Machine.from_file(machine_path)
        with open(params_path) as f:
            params = json.load(f)
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_tokens = jnp.array(em.tokenize_input(list("101")))
        out_tokens = jnp.array(em.tokenize_output(list("001")))

        # Use PSWM for input, token for output
        in_pswm = PSWMSeq(log_probs=TokenSeq(tokens=in_tokens).emission_weights(jm.n_input_tokens))
        ll_pswm = float(forward_2d_dense(jm, in_pswm, out_tokens, LOGSUMEXP))
        ll_tok = float(forward_2d_dense(jm, in_tokens, out_tokens, LOGSUMEXP))
        assert ll_pswm == pytest.approx(ll_tok, abs=0.01)

    def test_pswm_backward_2d(self, repo_root, boss_path):
        """PSWM should work with backward too."""
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_tokens = jnp.array(em.tokenize_input(list("10")))
        out_tokens = jnp.array(em.tokenize_output(list("10")))

        in_pswm = PSWMSeq(log_probs=TokenSeq(tokens=in_tokens).emission_weights(jm.n_input_tokens))
        out_pswm = PSWMSeq(log_probs=TokenSeq(tokens=out_tokens).emission_weights(jm.n_output_tokens))

        bp_tok = backward_2d_dense(jm, in_tokens, out_tokens, LOGSUMEXP)
        bp_pswm = backward_2d_dense(jm, in_pswm, out_pswm, LOGSUMEXP)

        assert float(bp_pswm[0, 0, 0]) == pytest.approx(float(bp_tok[0, 0, 0]), abs=0.01)

    def test_pswm_optimal_2d(self, repo_root, boss_path):
        """PSWM should work with 2D optimal."""
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_tokens = jnp.array(em.tokenize_input(list("10")))
        out_tokens = jnp.array(em.tokenize_output(list("10")))

        in_pswm = PSWMSeq(log_probs=TokenSeq(tokens=in_tokens).emission_weights(jm.n_input_tokens))
        out_pswm = PSWMSeq(log_probs=TokenSeq(tokens=out_tokens).emission_weights(jm.n_output_tokens))

        ll_simple = float(forward_2d_dense(jm, in_pswm, out_pswm, LOGSUMEXP))
        ll_optimal = float(forward_2d_optimal(jm, in_pswm, out_pswm, LOGSUMEXP))
        assert ll_optimal == pytest.approx(ll_simple, abs=0.01)


from .conftest import *
