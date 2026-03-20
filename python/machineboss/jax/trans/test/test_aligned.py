"""Tests for alignment-constrained DP on TransMachine."""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.dp_aligned import (
    aligned_log_forward as existing_aligned_forward,
    aligned_log_viterbi as existing_aligned_viterbi,
)
from machineboss.jax.trans.machine import TransMachine
from machineboss.jax.trans.dp_aligned import (
    aligned_forward, aligned_viterbi, validate_alignment, MAT, INS, DEL,
)
from machineboss.jax.semiring import LOGSUMEXP


class TestAlignedForward:
    """Test aligned forward matches existing implementation."""

    def test_all_match(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))
        alignment = jnp.array([MAT, MAT])

        result_trans = float(aligned_forward(
            bitecho_tm, in_seq, out_seq, alignment, LOGSUMEXP))
        result_existing = float(existing_aligned_forward(
            bitecho_jm, in_seq, out_seq, alignment))

        assert result_trans == pytest.approx(result_existing, abs=0.01)

    def test_with_indels(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("1")))
        alignment = jnp.array([MAT, INS])

        result_trans = float(aligned_forward(
            bitecho_tm, in_seq, out_seq, alignment, LOGSUMEXP))
        result_existing = float(existing_aligned_forward(
            bitecho_jm, in_seq, out_seq, alignment))

        assert result_trans == pytest.approx(result_existing, abs=0.01)


class TestValidateAlignment:
    """Test alignment validation."""

    def test_valid(self):
        validate_alignment(jnp.array([MAT, MAT, INS, DEL]), 3, 3)

    def test_invalid_input_count(self):
        with pytest.raises(ValueError):
            validate_alignment(jnp.array([MAT, MAT]), 3, 2)

    def test_invalid_output_count(self):
        with pytest.raises(ValueError):
            validate_alignment(jnp.array([MAT, MAT]), 2, 3)
