"""Tests for alignment-constrained DP algorithms."""

import json
import math

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import numpy.testing as npt

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine, NEG_INF
from machineboss.jax.forward import log_forward
from machineboss.jax.viterbi import log_viterbi
from machineboss.jax.jax_weight import ParameterizedMachine
from machineboss.jax.dp_neural import neural_log_forward_tok, neural_log_viterbi_tok
from machineboss.jax.dp_aligned import (
    aligned_log_forward, aligned_log_viterbi,
    neural_aligned_log_forward, neural_aligned_log_viterbi,
    validate_alignment, MAT, INS, DEL,
)


@pytest.fixture
def repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[4]


@pytest.fixture
def boss_path(repo_root):
    import shutil
    p = repo_root / "bin" / "boss"
    if not p.exists():
        pytest.skip("boss binary not found")
    return str(p)


class TestValidateAlignment:
    """Test alignment validation."""

    def test_valid_all_match(self):
        """All-match alignment: Li=Lo=3."""
        aln = jnp.array([MAT, MAT, MAT], dtype=jnp.int32)
        validate_alignment(aln, Li=3, Lo=3)  # should not raise

    def test_valid_with_indels(self):
        """MAT+INS=2, MAT+DEL=2."""
        aln = jnp.array([MAT, INS, DEL], dtype=jnp.int32)
        validate_alignment(aln, Li=2, Lo=2)

    def test_invalid_input_count(self):
        """MAT+INS != Li should raise."""
        aln = jnp.array([MAT, MAT], dtype=jnp.int32)
        with pytest.raises(ValueError, match="input length"):
            validate_alignment(aln, Li=3, Lo=2)

    def test_invalid_output_count(self):
        """MAT+DEL != Lo should raise."""
        aln = jnp.array([MAT, MAT], dtype=jnp.int32)
        with pytest.raises(ValueError, match="output length"):
            validate_alignment(aln, Li=2, Lo=3)


class TestAlignedStandard:
    """Test standard (fixed-weight) alignment-constrained DP."""

    @pytest.fixture
    def bitnoise_jm(self, repo_root):
        """bitnoise machine evaluated with p=0.9, q=0.1."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        machine = Machine.from_file(td_path)
        em = EvaluatedMachine.from_machine(machine, {"p": 0.9, "q": 0.1})
        return JAXMachine.from_evaluated(em)

    def test_all_match_le_unconstrained(self, bitnoise_jm):
        """Aligned forward (all-match) should be <= unconstrained forward."""
        jm = bitnoise_jm
        in_toks = jnp.array([1, 2], dtype=jnp.int32)  # "10"
        out_toks = jnp.array([1, 2], dtype=jnp.int32)  # "10"
        aln = jnp.array([MAT, MAT], dtype=jnp.int32)

        ll_aligned = float(aligned_log_forward(jm, in_toks, out_toks, aln))
        ll_full = float(log_forward(jm, in_toks, out_toks))

        assert math.isfinite(ll_aligned)
        # Aligned restricts the path space, so ll_aligned <= ll_full
        assert ll_aligned <= ll_full + 0.01

    def test_all_match_viterbi_le_forward(self, bitnoise_jm):
        """Aligned Viterbi <= aligned Forward."""
        jm = bitnoise_jm
        in_toks = jnp.array([1, 2], dtype=jnp.int32)
        out_toks = jnp.array([1, 2], dtype=jnp.int32)
        aln = jnp.array([MAT, MAT], dtype=jnp.int32)

        ll_fwd = float(aligned_log_forward(jm, in_toks, out_toks, aln))
        ll_vit = float(aligned_log_viterbi(jm, in_toks, out_toks, aln))
        assert ll_vit <= ll_fwd + 1e-5

    def test_indel_alignment(self, bitnoise_jm):
        """Alignment with insertions and deletions."""
        jm = bitnoise_jm
        in_toks = jnp.array([1, 2], dtype=jnp.int32)   # "10"
        out_toks = jnp.array([2, 1], dtype=jnp.int32)   # "01"
        # INS, DEL, MAT = consume input, then output, then both
        # This alignment: (1,0)→INS→(2,0), (2,0)→DEL→(2,1), (2,1)→MAT→... wait
        # Actually: INS consumes input, DEL consumes output
        # After INS: i=1, o=0. After DEL: i=1, o=1. After MAT: i=2, o=2.
        aln = jnp.array([INS, DEL, MAT], dtype=jnp.int32)
        validate_alignment(aln, Li=2, Lo=2)

        ll = float(aligned_log_forward(jm, in_toks, out_toks, aln))
        assert math.isfinite(ll)

    def test_vs_boss_all_match(self, repo_root, boss_path):
        """All-match alignment on matching sequences should equal diagonal path."""
        import subprocess
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        in_str, out_str = "10", "10"
        result = subprocess.run(
            [boss_path, td_path,
             "--input-chars", in_str, "--output-chars", out_str,
             "-P", params_path, "-L"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"boss failed: {result.stderr}")
        data = json.loads(result.stdout)
        boss_ll = float(data[0][-1]) if isinstance(data[0], list) else float(data[0])

        with open(params_path) as f:
            param_vals = json.load(f)
        machine = Machine.from_file(td_path)
        em = EvaluatedMachine.from_machine(machine, param_vals)
        jm = JAXMachine.from_evaluated(em)

        in_toks = jnp.array(em.tokenize_input(list(in_str)), dtype=jnp.int32)
        out_toks = jnp.array(em.tokenize_output(list(out_str)), dtype=jnp.int32)
        aln = jnp.array([MAT, MAT], dtype=jnp.int32)

        ll_aligned = float(aligned_log_forward(jm, in_toks, out_toks, aln))

        # Aligned forward restricts to diagonal path, so should be <= boss
        assert ll_aligned <= boss_ll + 0.01
        # But for matching sequences with bitnoise (diagonal dominates), should be close
        assert ll_aligned == pytest.approx(boss_ll, abs=0.5)


class TestAlignedNeural:
    """Test neural alignment-constrained DP."""

    @pytest.fixture
    def bitnoise_pm(self, repo_root):
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        machine = Machine.from_file(td_path)
        return ParameterizedMachine.from_machine(machine)

    def test_matches_neural_unconstrained_all_match(self, bitnoise_pm):
        """All-match aligned neural should be <= unconstrained neural."""
        pm = bitnoise_pm
        in_toks = jnp.array(pm.tokenize_input(list("10")), dtype=jnp.int32)
        out_toks = jnp.array(pm.tokenize_output(list("10")), dtype=jnp.int32)
        Li, Lo = 2, 2
        params = {
            "p": jnp.full((Li + 1, Lo + 1), 0.9),
            "q": jnp.full((Li + 1, Lo + 1), 0.1),
        }
        aln = jnp.array([MAT, MAT], dtype=jnp.int32)

        ll_aligned = float(neural_aligned_log_forward(
            pm, in_toks, out_toks, aln, params))
        ll_full = float(neural_log_forward_tok(pm, in_toks, out_toks, params))

        assert math.isfinite(ll_aligned)
        assert ll_aligned <= ll_full + 0.01

    def test_viterbi_le_forward(self, bitnoise_pm):
        """Neural aligned Viterbi <= Forward."""
        pm = bitnoise_pm
        in_toks = jnp.array(pm.tokenize_input(list("10")), dtype=jnp.int32)
        out_toks = jnp.array(pm.tokenize_output(list("10")), dtype=jnp.int32)
        Li, Lo = 2, 2
        params = {
            "p": jnp.full((Li + 1, Lo + 1), 0.9),
            "q": jnp.full((Li + 1, Lo + 1), 0.1),
        }
        aln = jnp.array([MAT, MAT], dtype=jnp.int32)

        ll_fwd = float(neural_aligned_log_forward(
            pm, in_toks, out_toks, aln, params))
        ll_vit = float(neural_aligned_log_viterbi(
            pm, in_toks, out_toks, aln, params))
        assert ll_vit <= ll_fwd + 1e-5

    def test_neural_aligned_with_indels(self, bitnoise_pm):
        """Neural aligned DP with insertions and deletions."""
        pm = bitnoise_pm
        in_toks = jnp.array(pm.tokenize_input(list("10")), dtype=jnp.int32)
        out_toks = jnp.array(pm.tokenize_output(list("01")), dtype=jnp.int32)
        Li, Lo = 2, 2
        params = {
            "p": jnp.full((Li + 1, Lo + 1), 0.9),
            "q": jnp.full((Li + 1, Lo + 1), 0.1),
        }
        aln = jnp.array([INS, DEL, MAT], dtype=jnp.int32)

        ll = float(neural_aligned_log_forward(
            pm, in_toks, out_toks, aln, params))
        assert math.isfinite(ll)

    def test_neural_aligned_matches_standard_aligned(self, repo_root):
        """Neural aligned with constant params should match standard aligned."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        machine = Machine.from_file(td_path)
        pm = ParameterizedMachine.from_machine(machine)
        em = EvaluatedMachine.from_machine(machine, {"p": 0.9, "q": 0.1})
        jm = JAXMachine.from_evaluated(em)

        in_toks = jnp.array(pm.tokenize_input(list("10")), dtype=jnp.int32)
        out_toks = jnp.array(pm.tokenize_output(list("10")), dtype=jnp.int32)
        Li, Lo = 2, 2
        params = {
            "p": jnp.full((Li + 1, Lo + 1), 0.9),
            "q": jnp.full((Li + 1, Lo + 1), 0.1),
        }
        aln = jnp.array([MAT, MAT], dtype=jnp.int32)

        ll_std = float(aligned_log_forward(jm, in_toks, out_toks, aln))
        ll_neural = float(neural_aligned_log_forward(
            pm, in_toks, out_toks, aln, params))

        assert ll_neural == pytest.approx(ll_std, abs=0.01)

    def test_neural_aligned_broadcast_params(self, bitnoise_pm):
        """Neural aligned with broadcast params matches full params."""
        pm = bitnoise_pm
        in_toks = jnp.array(pm.tokenize_input(list("10")), dtype=jnp.int32)
        out_toks = jnp.array(pm.tokenize_output(list("10")), dtype=jnp.int32)
        Li, Lo = 2, 2
        full_params = {
            "p": jnp.full((Li + 1, Lo + 1), 0.9),
            "q": jnp.full((Li + 1, Lo + 1), 0.1),
        }
        bcast_params = {
            "p": jnp.full((1, 1), 0.9),
            "q": jnp.full((1, 1), 0.1),
        }
        aln = jnp.array([MAT, MAT], dtype=jnp.int32)

        ll_full = float(neural_aligned_log_forward(
            pm, in_toks, out_toks, aln, full_params))
        ll_bcast = float(neural_aligned_log_forward(
            pm, in_toks, out_toks, aln, bcast_params))
        assert ll_bcast == pytest.approx(ll_full, abs=1e-5)

    def test_neural_aligned_grad(self, bitnoise_pm):
        """JAX grad through neural aligned forward."""
        pm = bitnoise_pm
        in_toks = jnp.array(pm.tokenize_input(list("10")), dtype=jnp.int32)
        out_toks = jnp.array(pm.tokenize_output(list("10")), dtype=jnp.int32)
        Li, Lo = 2, 2
        q_tensor = jnp.full((Li + 1, Lo + 1), 0.1)
        aln = jnp.array([MAT, MAT], dtype=jnp.int32)

        def loss_fn(p_t):
            return neural_aligned_log_forward(
                pm, in_toks, out_toks, aln, {"p": p_t, "q": q_tensor})

        p_tensor = jnp.full((Li + 1, Lo + 1), 0.9)
        grad_p = jax.grad(loss_fn)(p_tensor)
        assert grad_p.shape == (Li + 1, Lo + 1)
        assert jnp.all(jnp.isfinite(grad_p))
        assert jnp.any(grad_p != 0.0)

    def test_jukescantor_aligned(self, repo_root):
        """Jukes-Cantor with defs, aligned."""
        jc_path = str(repo_root / "preset" / "jukescantor.json")
        machine = Machine.from_file(jc_path)
        pm = ParameterizedMachine.from_machine(machine)

        in_str = "ACGT"
        out_str = "ACGA"  # one substitution
        Li, Lo = 4, 4
        in_toks = jnp.array(pm.tokenize_input(list(in_str)), dtype=jnp.int32)
        out_toks = jnp.array(pm.tokenize_output(list(out_str)), dtype=jnp.int32)
        params = {"t": jnp.full((1, 1), 0.5)}
        aln = jnp.array([MAT, MAT, MAT, MAT], dtype=jnp.int32)

        ll_aligned = float(neural_aligned_log_forward(
            pm, in_toks, out_toks, aln, params))
        ll_full = float(neural_log_forward_tok(pm, in_toks, out_toks,
                                                {"t": jnp.full((Li + 1, Lo + 1), 0.5)}))

        assert math.isfinite(ll_aligned)
        # Aligned (all-match) <= unconstrained
        assert ll_aligned <= ll_full + 0.01
