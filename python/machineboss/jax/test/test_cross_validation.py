"""Cross-validation tests: all DP variants must agree on the same (machine, sequences)."""

import json
import subprocess

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine, NEG_INF
from machineboss.jax.semiring import LOGSUMEXP, MAXPLUS
from machineboss.jax.seq import TokenSeq, PSWMSeq, pad_length, pad_token_seq
from machineboss.jax.dp_2d_simple import (
    forward_2d_dense, forward_2d_sparse,
    backward_2d_dense, backward_2d_sparse,
)
from machineboss.jax.dp_2d_optimal import forward_2d_optimal, backward_2d_optimal
from machineboss.jax.dp_1d_simple import forward_1d_simple, backward_1d_simple
from machineboss.jax.dp_1d_optimal import forward_1d_optimal, backward_1d_optimal
from machineboss.jax.forward import log_forward
from machineboss.jax.viterbi import log_viterbi


class TestCrossValidation2D:
    """All 2D variants must agree."""

    def _get_bitnoise(self, repo_root):
        machine_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")
        m = Machine.from_file(machine_path)
        with open(params_path) as f:
            params = json.load(f)
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)
        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("001")))
        return jm, in_seq, out_seq

    def test_forward_all_variants(self, repo_root, boss_path):
        """All Forward variants agree on bitnoise."""
        jm, in_seq, out_seq = self._get_bitnoise(repo_root)

        results = {
            '2d_dense_simple': float(forward_2d_dense(jm, in_seq, out_seq, LOGSUMEXP)),
            '2d_sparse_simple': float(forward_2d_sparse(jm, in_seq, out_seq, LOGSUMEXP)),
            '2d_dense_optimal': float(forward_2d_optimal(jm, in_seq, out_seq, LOGSUMEXP)),
        }

        # PSWM variants
        in_pswm = PSWMSeq(log_probs=TokenSeq(tokens=in_seq).emission_weights(jm.n_input_tokens))
        out_pswm = PSWMSeq(log_probs=TokenSeq(tokens=out_seq).emission_weights(jm.n_output_tokens))
        results['2d_dense_simple_pswm'] = float(forward_2d_dense(jm, in_pswm, out_pswm, LOGSUMEXP))
        results['2d_sparse_simple_pswm'] = float(forward_2d_sparse(jm, in_pswm, out_pswm, LOGSUMEXP))
        results['2d_dense_optimal_pswm'] = float(forward_2d_optimal(jm, in_pswm, out_pswm, LOGSUMEXP))

        ref = results['2d_dense_simple']
        for name, val in results.items():
            assert val == pytest.approx(ref, abs=0.01), f"{name} = {val} != ref {ref}"

    def test_viterbi_all_variants(self, repo_root, boss_path):
        """All Viterbi variants agree on bitnoise."""
        jm, in_seq, out_seq = self._get_bitnoise(repo_root)

        results = {
            '2d_dense_simple': float(forward_2d_dense(jm, in_seq, out_seq, MAXPLUS)),
            '2d_sparse_simple': float(forward_2d_sparse(jm, in_seq, out_seq, MAXPLUS)),
            '2d_dense_optimal': float(forward_2d_optimal(jm, in_seq, out_seq, MAXPLUS)),
        }

        ref = results['2d_dense_simple']
        for name, val in results.items():
            assert val == pytest.approx(ref, abs=0.01), f"{name} = {val} != ref {ref}"

    def test_viterbi_le_forward(self, repo_root, boss_path):
        """Viterbi <= Forward for all variants."""
        jm, in_seq, out_seq = self._get_bitnoise(repo_root)
        fwd = float(forward_2d_dense(jm, in_seq, out_seq, LOGSUMEXP))
        vit = float(forward_2d_dense(jm, in_seq, out_seq, MAXPLUS))
        assert vit <= fwd + 0.01

    def test_backward_eq_forward(self, repo_root, boss_path):
        """Backward[0,0,0] == Forward log-likelihood for all variants."""
        jm, in_seq, out_seq = self._get_bitnoise(repo_root)

        fwd = float(forward_2d_dense(jm, in_seq, out_seq, LOGSUMEXP))

        bp_simple = backward_2d_dense(jm, in_seq, out_seq, LOGSUMEXP)
        bp_sparse = backward_2d_sparse(jm, in_seq, out_seq, LOGSUMEXP)
        bp_optimal = backward_2d_optimal(jm, in_seq, out_seq, LOGSUMEXP)

        assert float(bp_simple[0, 0, 0]) == pytest.approx(fwd, abs=0.01)
        assert float(bp_sparse[0, 0, 0]) == pytest.approx(fwd, abs=0.01)
        assert float(bp_optimal[0, 0, 0]) == pytest.approx(fwd, abs=0.01)


class TestCrossValidation1D:
    """All 1D variants must agree."""

    def _get_generator(self, boss_path):
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)
        out_seq = jnp.array(em.tokenize_output(list("101")))
        return jm, out_seq

    def test_forward_all_variants(self, repo_root, boss_path):
        """All 1D Forward variants agree."""
        jm, out_seq = self._get_generator(boss_path)

        results = {
            '1d_dense_simple': float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP, kernel='dense')),
            '1d_sparse_simple': float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP, kernel='sparse')),
            '1d_dense_optimal': float(forward_1d_optimal(jm, None, out_seq, LOGSUMEXP)),
        }

        # PSWM
        tok_seq = TokenSeq(tokens=out_seq)
        pswm_seq = PSWMSeq(log_probs=tok_seq.emission_weights(jm.n_output_tokens))
        results['1d_dense_simple_pswm'] = float(forward_1d_simple(jm, None, pswm_seq, LOGSUMEXP))
        results['1d_sparse_simple_pswm'] = float(forward_1d_simple(jm, None, pswm_seq, LOGSUMEXP, kernel='sparse'))
        results['1d_dense_optimal_pswm'] = float(forward_1d_optimal(jm, None, pswm_seq, LOGSUMEXP))

        ref = results['1d_dense_simple']
        for name, val in results.items():
            assert val == pytest.approx(ref, abs=0.01), f"{name} = {val} != ref {ref}"

    def test_1d_eq_2d_empty_input(self, repo_root, boss_path):
        """1D generator should match 2D with empty input."""
        jm, out_seq = self._get_generator(boss_path)
        in_empty = jnp.array([], dtype=jnp.int32)

        ll_1d = float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP))
        ll_2d = float(forward_2d_dense(jm, in_empty, out_seq, LOGSUMEXP))
        assert ll_1d == pytest.approx(ll_2d, abs=0.01)

    def test_dispatch_auto(self, repo_root, boss_path):
        """log_forward with auto strategy should give correct results."""
        jm, out_seq = self._get_generator(boss_path)

        # Generator: None input should work
        ll_auto = float(log_forward(jm, None, out_seq))
        ll_ref = float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP))
        assert ll_auto == pytest.approx(ll_ref, abs=0.01)


class TestMachineTypeValidation:
    """Test that dispatch wrappers validate machine type."""

    def test_generator_rejects_input(self, repo_root, boss_path):
        """Generator should reject input_seq."""
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array([1, 2, 1], dtype=jnp.int32)
        out_seq = jnp.array(em.tokenize_output(list("101")))

        with pytest.raises(ValueError, match="generator"):
            log_forward(jm, in_seq, out_seq)

    def test_generator_requires_output(self, repo_root, boss_path):
        """Generator should require output_seq."""
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        with pytest.raises(ValueError, match="generator"):
            log_forward(jm, None, None)

    def test_recognizer_rejects_output(self, repo_root, boss_path):
        """Recognizer should reject output_seq."""
        result = subprocess.run(
            [boss_path, "--recognize-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array([1, 2, 1], dtype=jnp.int32)

        with pytest.raises(ValueError, match="recognizer"):
            log_forward(jm, in_seq, out_seq)

    def test_machine_type_detection(self, repo_root, boss_path):
        """Machine type detection should work."""
        # Generator
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm_gen = JAXMachine.from_evaluated(em)
        assert jm_gen.is_generator()
        assert jm_gen.machine_type() == 'generator'

        # Transducer
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm_trans = JAXMachine.from_evaluated(em)
        assert jm_trans.is_transducer()
        assert jm_trans.machine_type() == 'transducer'


class TestPadding:
    """Test padding utilities."""

    def test_pad_length_small(self):
        assert pad_length(1) == 1
        assert pad_length(4) == 4

    def test_pad_length_geometric(self):
        p = pad_length(5)
        assert p >= 5
        # Same padded length for nearby values (avoids recompilation)
        assert pad_length(5) == pad_length(6)

    def test_pad_length_large(self):
        p = pad_length(100)
        assert p >= 100
        assert p <= 200  # should not over-pad

    def test_padded_tok_seq_gives_same_result(self, repo_root, boss_path):
        """Padding a TokenSeq and passing length gives same result as unpadded."""
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_tokens = jnp.array(em.tokenize_output(list("101")))
        tok_seq = TokenSeq(tokens=out_tokens)

        padded_len = pad_length(len(tok_seq))
        padded_seq, orig_len = pad_token_seq(tok_seq, padded_len)

        ll_orig = float(forward_1d_simple(jm, None, tok_seq, LOGSUMEXP))

        # With length parameter, the engine only processes orig_len positions
        ll_padded_simple = float(forward_1d_simple(
            jm, None, padded_seq, LOGSUMEXP, length=orig_len))
        assert ll_padded_simple == pytest.approx(ll_orig, abs=0.01)

        # Optimal engine with padding
        ll_padded_optimal = float(forward_1d_optimal(
            jm, None, padded_seq, LOGSUMEXP, length=orig_len))
        assert ll_padded_optimal == pytest.approx(ll_orig, abs=0.01)

        # Dispatch wrapper with length
        ll_dispatch = float(log_forward(jm, None, padded_seq, length=orig_len))
        assert ll_dispatch == pytest.approx(ll_orig, abs=0.01)

    def test_padded_backward_gives_same_result(self, repo_root, boss_path):
        """Padded backward with length gives same log-likelihood as unpadded."""
        result = subprocess.run(
            [boss_path, "--generate-chars", "10"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_tokens = jnp.array(em.tokenize_output(list("10")))
        tok_seq = TokenSeq(tokens=out_tokens)

        padded_len = pad_length(len(tok_seq))
        padded_seq, orig_len = pad_token_seq(tok_seq, padded_len)

        fwd_orig = float(forward_1d_simple(jm, None, tok_seq, LOGSUMEXP))

        # Backward[0, start_state] should equal Forward log-likelihood
        bp_simple = backward_1d_simple(jm, None, padded_seq, LOGSUMEXP,
                                       length=orig_len)
        assert float(bp_simple[0, 0]) == pytest.approx(fwd_orig, abs=0.01)

        bp_optimal = backward_1d_optimal(jm, None, padded_seq, LOGSUMEXP,
                                         length=orig_len)
        assert float(bp_optimal[0, 0]) == pytest.approx(fwd_orig, abs=0.01)


from .conftest import *
