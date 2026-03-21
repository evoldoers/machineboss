"""Tests for 2D DP algorithms on TransMachine."""

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
from machineboss.jax.trans.dp_2d import forward_2d, backward_2d, viterbi_2d


def _boss_loglike(boss_path, machine_file, input_str=None, output_str=None,
                  params_file=None):
    args = [boss_path, machine_file]
    if input_str:
        args += ["--input-chars", input_str]
    if output_str:
        args += ["--output-chars", output_str]
    if params_file:
        args += ["-P", params_file]
    args.append("-L")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"boss failed: {result.stderr}")
    data = json.loads(result.stdout)
    if isinstance(data, list):
        return float(data[0][-1]) if isinstance(data[0], list) else float(data[0])
    return float(data)


class TestForward2DSimple:
    """Test 2D forward (simple) against C++ boss and existing JAX."""

    def test_bitecho_matching(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("101")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("101")))

        result_trans = float(forward_2d(bitecho_tm, in_seq, out_seq, strategy='simple'))
        result_existing = float(log_forward(bitecho_jm, in_seq, out_seq))

        assert result_trans == pytest.approx(result_existing, abs=0.01)

    def test_bitecho_mismatching(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("101")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("001")))

        result_trans = float(forward_2d(bitecho_tm, in_seq, out_seq, strategy='simple'))
        assert result_trans < -30

    def test_matches_boss(self, repo_root, boss_path, bitecho_tm, bitecho_em):
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        in_seq = jnp.array(bitecho_em.tokenize_input(list("101")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("101")))

        result_trans = float(forward_2d(bitecho_tm, in_seq, out_seq, strategy='simple'))
        boss_ll = _boss_loglike(boss_path, machine_path,
                                input_str="101", output_str="101")
        assert result_trans == pytest.approx(boss_ll, abs=0.01)

    def test_bitnoise_matches_existing(self, bitnoise_tm, bitnoise_jm):
        em = EvaluatedMachine(
            n_states=bitnoise_jm.n_states,
            input_tokens=list(bitnoise_jm.input_token_list),
            output_tokens=list(bitnoise_jm.output_token_list),
            transitions=[],
        )
        tok_map_in = {s: i for i, s in enumerate(bitnoise_jm.input_token_list)}
        tok_map_out = {s: i for i, s in enumerate(bitnoise_jm.output_token_list)}
        in_seq = jnp.array([tok_map_in['1'], tok_map_in['0'], tok_map_in['1']])
        out_seq = jnp.array([tok_map_out['0'], tok_map_out['1'], tok_map_out['1']])

        result_trans = float(forward_2d(bitnoise_tm, in_seq, out_seq, strategy='simple'))
        result_existing = float(log_forward(bitnoise_jm, in_seq, out_seq))

        assert result_trans == pytest.approx(result_existing, abs=0.01)


class TestForward2DOptimal:
    """Test 2D forward (optimal) matches simple."""

    def test_matches_simple(self, bitecho_tm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        result_simple = float(forward_2d(bitecho_tm, in_seq, out_seq, strategy='simple'))
        result_optimal = float(forward_2d(bitecho_tm, in_seq, out_seq, strategy='optimal'))

        assert result_optimal == pytest.approx(result_simple, abs=0.01)


class TestBackward2D:
    """Test 2D backward cross-validates."""

    def test_backward_matches_existing(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        bp_trans = backward_2d(bitecho_tm, in_seq, out_seq, strategy='simple')
        bp_existing = log_backward_matrix(bitecho_jm, in_seq, out_seq)

        # Compare only finite values (both -inf representations are equivalent)
        finite_mask = (bp_trans > -1e30) | (bp_existing > -1e30)
        if jnp.any(finite_mask):
            assert jnp.allclose(
                bp_trans[finite_mask], bp_existing[finite_mask], atol=0.1)

    def test_forward_backward_consistent(self, bitecho_tm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        fwd = float(forward_2d(bitecho_tm, in_seq, out_seq, strategy='simple'))
        bp = backward_2d(bitecho_tm, in_seq, out_seq, strategy='simple')
        # bp[0, 0, 0] should equal the total log-likelihood
        bwd_ll = float(bp[0, 0, 0])

        assert fwd == pytest.approx(bwd_ll, abs=0.1)


class TestViterbi2D:
    """Test 2D Viterbi."""

    def test_viterbi_le_forward(self, bitecho_tm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        fwd = float(forward_2d(bitecho_tm, in_seq, out_seq))
        vit = float(viterbi_2d(bitecho_tm, in_seq, out_seq))

        assert vit <= fwd + 1e-5


class TestAutoPad2D:
    """Test 2D auto-padding produces same results."""

    def test_auto_pad_matches_no_pad(self, bitecho_tm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("101")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("101")))

        result_padded = float(forward_2d(bitecho_tm, in_seq, out_seq, auto_pad=True))
        result_unpadded = float(forward_2d(bitecho_tm, in_seq, out_seq, auto_pad=False))

        assert result_padded == pytest.approx(result_unpadded, abs=1e-4)

    def test_auto_pad_longer_seq(self, bitnoise_tm):
        """Longer sequences that trigger actual padding (>4 tokens)."""
        # Use bitnoise which has {0, 1} alphabet
        tok_in = {s: i for i, s in enumerate(bitnoise_tm.input_tokens)}
        tok_out = {s: i for i, s in enumerate(bitnoise_tm.output_tokens)}
        in_seq = jnp.array([tok_in['1'], tok_in['0'], tok_in['1'],
                            tok_in['0'], tok_in['1'], tok_in['0'], tok_in['1']])
        out_seq = jnp.array([tok_out['0'], tok_out['1'], tok_out['0'],
                             tok_out['1'], tok_out['0']])

        result_padded = float(forward_2d(bitnoise_tm, in_seq, out_seq, auto_pad=True))
        result_unpadded = float(forward_2d(bitnoise_tm, in_seq, out_seq, auto_pad=False))

        assert result_padded == pytest.approx(result_unpadded, abs=1e-4)

    def test_viterbi_auto_pad(self, bitecho_tm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10110")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10110")))

        result_padded = float(viterbi_2d(bitecho_tm, in_seq, out_seq, auto_pad=True))
        result_unpadded = float(viterbi_2d(bitecho_tm, in_seq, out_seq, auto_pad=False))

        assert result_padded == pytest.approx(result_unpadded, abs=1e-4)
