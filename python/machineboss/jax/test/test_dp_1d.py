"""Tests for 1D DP algorithms (generators/recognizers)."""

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
from machineboss.jax.dp_1d_simple import forward_1d_simple, backward_1d_simple
from machineboss.jax.dp_2d_simple import forward_2d_dense, backward_2d_dense


class TestForward1DGenerator:
    """Test 1D Forward on generators (no input)."""

    def test_generator_101(self, repo_root, boss_path):
        """Generator for "101" should give log-prob 0."""
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("101")))
        ll = float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP))
        assert ll == pytest.approx(0.0, abs=0.01)

    def test_generator_wrong_output(self, repo_root, boss_path):
        """Generator for "101" with output "110" should give very low prob."""
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("110")))
        ll = float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP))
        assert ll < -30

    def test_1d_matches_2d_generator(self, repo_root, boss_path):
        """1D forward on generator should match 2D forward with empty input."""
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("101")))
        in_seq_empty = jnp.array([], dtype=jnp.int32)

        ll_1d = float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP))
        ll_2d = float(forward_2d_dense(jm, in_seq_empty, out_seq, LOGSUMEXP))
        assert ll_1d == pytest.approx(ll_2d, abs=0.01)


class TestForward1DRecognizer:
    """Test 1D Forward on recognizers (no output)."""

    def test_recognizer_101(self, repo_root, boss_path):
        """Recognizer for "101" should give log-prob 0."""
        result = subprocess.run(
            [boss_path, "--recognize-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        ll = float(forward_1d_simple(jm, in_seq, None, LOGSUMEXP))
        assert ll == pytest.approx(0.0, abs=0.01)

    def test_1d_matches_2d_recognizer(self, repo_root, boss_path):
        """1D forward on recognizer should match 2D forward with empty output."""
        result = subprocess.run(
            [boss_path, "--recognize-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq_empty = jnp.array([], dtype=jnp.int32)

        ll_1d = float(forward_1d_simple(jm, in_seq, None, LOGSUMEXP))
        ll_2d = float(forward_2d_dense(jm, in_seq, out_seq_empty, LOGSUMEXP))
        assert ll_1d == pytest.approx(ll_2d, abs=0.01)


class TestViterbi1D:
    """Test 1D Viterbi."""

    def test_viterbi_generator(self, repo_root, boss_path):
        """Viterbi on generator should equal Forward for deterministic machine."""
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("101")))
        fwd = float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP))
        vit = float(forward_1d_simple(jm, None, out_seq, MAXPLUS))
        assert vit == pytest.approx(fwd, abs=0.01)

    def test_viterbi_le_forward(self, repo_root, boss_path):
        """Viterbi <= Forward for non-deterministic machines."""
        machine_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        m = Machine.from_file(machine_path)
        with open(params_path) as f:
            params = json.load(f)
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        # bitnoise is a transducer, not a generator; use 2D for the Viterbi test
        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("001")))
        fwd = float(forward_2d_dense(jm, in_seq, out_seq, LOGSUMEXP))
        vit = float(forward_2d_dense(jm, in_seq, out_seq, MAXPLUS))
        assert vit <= fwd + 0.01


class TestBackward1D:
    """Test 1D Backward."""

    def test_backward_start_eq_forward(self, repo_root, boss_path):
        """Backward[0, start_state] should equal Forward log-likelihood."""
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        out_seq = jnp.array(em.tokenize_output(list("101")))
        ll_fwd = float(forward_1d_simple(jm, None, out_seq, LOGSUMEXP))
        bp = backward_1d_simple(jm, None, out_seq, LOGSUMEXP)
        ll_bwd = float(bp[0, 0])
        assert ll_bwd == pytest.approx(ll_fwd, abs=0.01)


class TestPSWM1D:
    """Test 1D DP with PSWMSeq."""

    def test_one_hot_pswm_eq_tok(self, repo_root, boss_path):
        """One-hot PSWM should give same result as TokenSeq."""
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        tokens = jnp.array(em.tokenize_output(list("101")))
        tok_seq = TokenSeq(tokens=tokens)
        n_out = machine_n_out = jm.n_output_tokens

        # Build one-hot PSWM
        pswm_data = tok_seq.emission_weights(n_out)
        pswm_seq = PSWMSeq(log_probs=pswm_data)

        ll_tok = float(forward_1d_simple(jm, None, tok_seq, LOGSUMEXP))
        ll_pswm = float(forward_1d_simple(jm, None, pswm_seq, LOGSUMEXP))
        assert ll_pswm == pytest.approx(ll_tok, abs=0.01)

    def test_uniform_pswm(self, repo_root, boss_path):
        """Uniform PSWM should give higher prob than a specific mismatch."""
        result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True,
        )
        m = Machine.from_json(result.stdout)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        n_out = jm.n_output_tokens
        L = 3

        # Uniform PSWM: all non-empty tokens equally likely
        log_probs = jnp.full((L, n_out), NEG_INF)
        n_real = n_out - 1  # exclude empty token
        log_probs = log_probs.at[:, 1:].set(-jnp.log(n_real))
        pswm_seq = PSWMSeq(log_probs=log_probs)

        ll_uniform = float(forward_1d_simple(jm, None, pswm_seq, LOGSUMEXP))

        # Wrong tokens
        wrong_tokens = jnp.array(em.tokenize_output(list("110")))
        ll_wrong = float(forward_1d_simple(jm, None, wrong_tokens, LOGSUMEXP))

        # Uniform should be > wrong specific tokens
        assert ll_uniform > ll_wrong


from .conftest import *
