"""Tests for forward-backward with counts on TransMachine."""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.fwdback import log_likelihood_with_counts
from machineboss.jax.trans.machine import TransMachine
from machineboss.jax.trans.fwdback import forward_backward, forward_backward_1d


class TestForwardBackward2D:
    """Test 2D forward-backward matches existing implementation."""

    def test_loglike_matches(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        ll_trans, _ = forward_backward(bitecho_tm, in_seq, out_seq)
        ll_existing, _ = log_likelihood_with_counts(bitecho_jm, in_seq, out_seq)

        assert float(ll_trans) == pytest.approx(float(ll_existing), abs=0.01)

    def test_counts_nonnegative(self, bitecho_tm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        _, counts = forward_backward(bitecho_tm, in_seq, out_seq)
        assert jnp.all(counts >= -1e-5)

    def test_counts_match_existing(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        _, counts_trans = forward_backward(bitecho_tm, in_seq, out_seq)
        _, counts_existing = log_likelihood_with_counts(
            bitecho_jm, in_seq, out_seq)

        assert jnp.allclose(counts_trans, counts_existing, atol=0.01)

    def test_simple_strategy(self, bitecho_tm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        ll_auto, counts_auto = forward_backward(bitecho_tm, in_seq, out_seq)
        ll_simple, counts_simple = forward_backward(
            bitecho_tm, in_seq, out_seq, strategy='simple')

        assert float(ll_auto) == pytest.approx(float(ll_simple), abs=0.01)
        assert jnp.allclose(counts_auto, counts_simple, atol=0.01)

    def test_optimal_strategy(self, bitecho_tm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        ll_simple, counts_simple = forward_backward(
            bitecho_tm, in_seq, out_seq, strategy='simple')
        ll_optimal, counts_optimal = forward_backward(
            bitecho_tm, in_seq, out_seq, strategy='optimal')

        assert float(ll_optimal) == pytest.approx(float(ll_simple), abs=0.01)
        assert jnp.allclose(counts_optimal, counts_simple, atol=0.01)

    def test_bitnoise_counts(self, bitnoise_tm, bitnoise_jm):
        """Test on bitnoise machine (more interesting topology)."""
        tok_map_in = {s: i for i, s in enumerate(bitnoise_jm.input_token_list)}
        tok_map_out = {s: i for i, s in enumerate(bitnoise_jm.output_token_list)}
        in_seq = jnp.array([tok_map_in['1'], tok_map_in['0']])
        out_seq = jnp.array([tok_map_out['0'], tok_map_out['1']])

        ll_trans, counts_trans = forward_backward(bitnoise_tm, in_seq, out_seq)
        ll_existing, counts_existing = log_likelihood_with_counts(
            bitnoise_jm, in_seq, out_seq)

        assert float(ll_trans) == pytest.approx(float(ll_existing), abs=0.1)
        assert jnp.allclose(counts_trans, counts_existing, atol=0.1)


class TestForwardBackward1D:
    """Test 1D forward-backward for generators/recognizers."""

    def test_generator(self, repo_root):
        from machineboss.machine import Machine
        from machineboss.eval import EvaluatedMachine

        path = repo_root / "t" / "machine" / "merge-chain.json"
        m = Machine.from_file(str(path))
        em = EvaluatedMachine.from_machine(m)
        tm = TransMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("qq")))

        ll, counts = forward_backward_1d(tm, output_seq=out_seq)

        assert jnp.isfinite(jnp.array(ll))
        assert jnp.all(counts >= -1e-5)
        # Counts should sum to something reasonable
        assert float(jnp.sum(counts)) > 0
