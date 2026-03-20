"""Tests for beam-Viterbi alignment on TransMachine."""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import numpy as np

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.beam_align import beam_align as existing_beam_align
from machineboss.jax.trans.machine import TransMachine
from machineboss.jax.trans.dp_beam import beam_align


class TestBeamAlign:
    """Test beam align matches existing implementation."""

    def test_bitecho_matches(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = np.array(bitecho_em.tokenize_input(list("10")))
        out_seq = np.array(bitecho_em.tokenize_output(list("10")))

        result_trans = beam_align(bitecho_tm, in_seq, out_seq, beam_width=100)
        result_existing = existing_beam_align(bitecho_jm, in_seq, out_seq, beam_width=100)

        assert result_trans.score == pytest.approx(result_existing.score, abs=1e-5)
