"""Tests for TransMachine kernel operations."""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.semiring import LOGSUMEXP, MAXPLUS
from machineboss.jax.trans.machine import TransMachine
from machineboss.jax.trans import kernel as tk
from machineboss.jax import kernel_sparse as ks
from machineboss.jax.types import NEG_INF


class TestPropagateSilent:
    """Test silent closure matches existing sparse kernel."""

    def test_forward_matches_sparse(self, bitecho_tm, bitecho_jm):
        S = bitecho_tm.n_states
        cell = jnp.full(S, NEG_INF).at[0].set(0.0)

        result_trans = tk.propagate_silent(cell, bitecho_tm, LOGSUMEXP)
        result_sparse = ks.propagate_silent_sparse(cell, bitecho_jm, LOGSUMEXP)

        assert jnp.allclose(result_trans, result_sparse, atol=1e-5)

    def test_backward_matches_sparse(self, bitecho_tm, bitecho_jm):
        S = bitecho_tm.n_states
        cell = jnp.full(S, NEG_INF).at[S - 1].set(0.0)

        result_trans = tk.propagate_silent_backward(cell, bitecho_tm, LOGSUMEXP)
        result_sparse = ks.propagate_silent_backward_sparse(cell, bitecho_jm, LOGSUMEXP)

        assert jnp.allclose(result_trans, result_sparse, atol=1e-5)

    def test_maxplus_forward(self, bitecho_tm, bitecho_jm):
        S = bitecho_tm.n_states
        cell = jnp.full(S, NEG_INF).at[0].set(0.0)

        result_trans = tk.propagate_silent(cell, bitecho_tm, MAXPLUS)
        result_sparse = ks.propagate_silent_sparse(cell, bitecho_jm, MAXPLUS)

        assert jnp.allclose(result_trans, result_sparse, atol=1e-5)


class TestEmitStepForward:
    """Test emit step forward matches existing sparse kernel."""

    def test_emit_in_matches(self, bitecho_tm, bitecho_jm):
        S = bitecho_tm.n_states
        prev = jnp.full(S, NEG_INF).at[0].set(0.0)
        prev = tk.propagate_silent(prev, bitecho_tm, LOGSUMEXP)

        # Use first input token
        in_e = jnp.full(bitecho_tm.n_in, NEG_INF).at[1].set(0.0)
        ew = tk.emission_weights_1d(bitecho_tm, in_e, True)

        result_trans = tk.emit_step_forward(
            jnp.full(S, NEG_INF), prev, bitecho_tm,
            bitecho_tm.emit_in_mask, ew, LOGSUMEXP)

        result_sparse = ks.emit_step_forward_sparse_pswm(
            jnp.full(S, NEG_INF), prev, bitecho_jm,
            in_e, None, emit_in=True, emit_out=False, semiring=LOGSUMEXP)

        assert jnp.allclose(result_trans, result_sparse, atol=1e-5)

    def test_emit_both_matches(self, bitecho_tm, bitecho_jm):
        S = bitecho_tm.n_states
        prev = jnp.full(S, NEG_INF).at[0].set(0.0)
        prev = tk.propagate_silent(prev, bitecho_tm, LOGSUMEXP)

        in_e = jnp.full(bitecho_tm.n_in, NEG_INF).at[1].set(0.0)
        out_e = jnp.full(bitecho_tm.n_out, NEG_INF).at[1].set(0.0)
        ew = tk.emission_weights_2d(bitecho_tm, in_e, out_e)

        result_trans = tk.emit_step_forward(
            jnp.full(S, NEG_INF), prev, bitecho_tm,
            bitecho_tm.emit_both_mask, ew, LOGSUMEXP)

        result_sparse = ks.emit_step_forward_sparse_pswm(
            jnp.full(S, NEG_INF), prev, bitecho_jm,
            in_e, out_e, emit_in=True, emit_out=True, semiring=LOGSUMEXP)

        assert jnp.allclose(result_trans, result_sparse, atol=1e-5)


class TestToMatrix:
    """Test to_matrix produces correct (S,S) matrices."""

    def test_silent_matrix_matches_dense(self, bitecho_tm, bitecho_jm):
        S = bitecho_tm.n_states
        ew = jnp.zeros(bitecho_tm.n_transitions)
        mat = tk.to_matrix(bitecho_tm, bitecho_tm.silent_mask, ew, LOGSUMEXP)
        dense_silent = bitecho_jm.log_trans[0, 0]  # (S, S)
        assert jnp.allclose(mat, dense_silent, atol=1e-5)

    def test_emit_matrix_matches_dense(self, bitecho_tm, bitecho_jm):
        S = bitecho_tm.n_states
        # Match transitions for in_tok=1, out_tok=1
        in_e = jnp.full(bitecho_tm.n_in, NEG_INF).at[1].set(0.0)
        out_e = jnp.full(bitecho_tm.n_out, NEG_INF).at[1].set(0.0)
        ew = tk.emission_weights_2d(bitecho_tm, in_e, out_e)
        mat = tk.to_matrix(bitecho_tm, bitecho_tm.emit_both_mask, ew, LOGSUMEXP)

        # Compare with dense: log_trans[1, 1]
        dense_mat = bitecho_jm.log_trans[1, 1]
        assert jnp.allclose(mat, dense_mat, atol=1e-5)
