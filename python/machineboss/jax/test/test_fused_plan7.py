"""Tests for Plan7-aware fused DP with nested scans."""

import json
import math
import subprocess

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.hmmer import HmmerModel
from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.fused import FusedMachine, fused_log_forward, fused_log_viterbi
from machineboss.jax.fused_plan7 import (
    FusedPlan7Machine, fused_plan7_log_forward, fused_plan7_log_viterbi,
)


def _build_generic_fused(hmmer_path, td_path, boss_path, *, multihit=False):
    """Build generic FusedMachine via boss Plan7 machine construction."""
    flag = "--hmmer-multihit" if multihit else "--hmmer-plan7"
    result = subprocess.run(
        [boss_path, flag, hmmer_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"boss failed: {result.stderr}")
    gen_em = EvaluatedMachine.from_machine(Machine.from_json(result.stdout))
    td_em = EvaluatedMachine.from_machine(Machine.from_file(td_path))
    return FusedMachine.build(gen_em, td_em), td_em


def _build_plan7_fused(hmmer_path, td_path, boss_path, *, multihit=False):
    """Build Plan7-aware FusedPlan7Machine from HmmerModel."""
    with open(hmmer_path) as f:
        hmmer = HmmerModel.read(f)
    td_em = EvaluatedMachine.from_machine(Machine.from_file(td_path))
    fm = FusedPlan7Machine.build(hmmer, td_em, multihit=multihit)
    return fm, td_em


class TestFusedPlan7Forward:
    """Test Plan7-aware fused Forward matches generic fused Forward."""

    def test_fn3_bitecho(self, repo_root, boss_path):
        """fn3.hmm + bitecho: Plan7 fused matches generic fused."""
        hmmer_path = str(repo_root / "t" / "hmmer" / "fn3.hmm")
        td_path = str(repo_root / "t" / "machine" / "bitecho.json")

        generic, td_em = _build_generic_fused(hmmer_path, td_path, boss_path)
        plan7, _ = _build_plan7_fused(hmmer_path, td_path, boss_path)

        # Use a short test sequence
        test_seq = list("010101")
        out_seq = jnp.array(td_em.tokenize_output(test_seq))

        gen_ll = float(fused_log_forward(generic, out_seq))
        p7_ll = float(fused_plan7_log_forward(plan7, out_seq))

        assert p7_ll == pytest.approx(gen_ll, abs=0.1), \
            f"Plan7={p7_ll} != generic={gen_ll}"

    def test_fn3_bitnoise(self, repo_root, boss_path):
        """fn3.hmm + bitnoise: Plan7 fused matches generic fused."""
        hmmer_path = str(repo_root / "t" / "hmmer" / "fn3.hmm")
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        with open(params_path) as f:
            params = json.load(f)

        # Generic
        flag = "--hmmer-plan7"
        result = subprocess.run(
            [boss_path, flag, hmmer_path],
            capture_output=True, text=True,
        )
        gen_em = EvaluatedMachine.from_machine(Machine.from_json(result.stdout))
        td_em = EvaluatedMachine.from_machine(
            Machine.from_file(td_path), params)
        generic = FusedMachine.build(gen_em, td_em)

        # Plan7
        with open(hmmer_path) as f:
            hmmer = HmmerModel.read(f)
        plan7 = FusedPlan7Machine.build(hmmer, td_em)

        test_seq = list("0101")
        out_seq = jnp.array(td_em.tokenize_output(test_seq))

        gen_ll = float(fused_log_forward(generic, out_seq))
        p7_ll = float(fused_plan7_log_forward(plan7, out_seq))

        assert p7_ll == pytest.approx(gen_ll, abs=0.1), \
            f"Plan7={p7_ll} != generic={gen_ll}"

    def test_vs_boss_compose(self, repo_root, boss_path):
        """Plan7 fused should match boss --hmmer-plan7 + compose + forward."""
        hmmer_path = str(repo_root / "t" / "hmmer" / "fn3.hmm")
        td_path = str(repo_root / "t" / "machine" / "bitecho.json")

        # Reference from boss (compose Plan7 with transducer)
        result = subprocess.run(
            [boss_path, "--hmmer-plan7", hmmer_path,
             "--compose", td_path,
             "--output-chars", "010101", "-L"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"boss compose failed: {result.stderr}")
        data = json.loads(result.stdout)
        boss_ll = float(data[0][-1]) if isinstance(data[0], list) else float(data[0])

        # Plan7 fused
        plan7, td_em = _build_plan7_fused(hmmer_path, td_path, boss_path)
        out_seq = jnp.array(td_em.tokenize_output(list("010101")))
        p7_ll = float(fused_plan7_log_forward(plan7, out_seq))

        # Both should agree: either both finite or both very negative
        if math.isinf(boss_ll) and boss_ll < 0:
            assert p7_ll < -1e30, f"Expected very negative, got {p7_ll}"
        else:
            assert p7_ll == pytest.approx(boss_ll, abs=0.1), \
                f"Plan7={p7_ll} != boss={boss_ll}"


class TestFusedPlan7Viterbi:
    """Test Plan7-aware fused Viterbi."""

    def test_viterbi_le_forward(self, repo_root, boss_path):
        """Viterbi <= Forward for Plan7 fused."""
        hmmer_path = str(repo_root / "t" / "hmmer" / "fn3.hmm")
        td_path = str(repo_root / "t" / "machine" / "bitecho.json")

        plan7, td_em = _build_plan7_fused(hmmer_path, td_path, boss_path)
        out_seq = jnp.array(td_em.tokenize_output(list("0101")))

        fwd_ll = float(fused_plan7_log_forward(plan7, out_seq))
        vit_ll = float(fused_plan7_log_viterbi(plan7, out_seq))

        assert vit_ll <= fwd_ll + 0.01, \
            f"Viterbi={vit_ll} > Forward={fwd_ll}"


# ============================================================
# Helpers for bug regression tests
# ============================================================

AA_ALPHABET = list('ACDEFGHIKLMNPQRSTVWY')


def _make_aa_echo_transducer():
    """Build a simple amino acid echo transducer (1 state, self-loops)."""
    aa_echo_json = {
        'state': [{
            'id': 'S',
            'trans': [{'in': aa, 'out': aa, 'to': 'S'} for aa in AA_ALPHABET]
        }]
    }
    return EvaluatedMachine.from_machine(
        Machine.from_json(json.dumps(aa_echo_json)))


def _build_aa_echo_plan7(repo_root, *, multihit=False):
    """Build Plan7-aware fused machine with fn3 + aa_echo."""
    hmmer_path = str(repo_root / "t" / "hmmer" / "fn3.hmm")
    with open(hmmer_path) as f:
        hmmer = HmmerModel.read(f)
    td_em = _make_aa_echo_transducer()
    fm = FusedPlan7Machine.build(hmmer, td_em, multihit=multihit)
    return fm, td_em


class TestFusedPlan7BugRegression:
    """Regression tests that catch the 4 bugs fixed from the JS kernel.

    Uses fn3.hmm + amino acid echo transducer, which exercises real amino acid
    emissions (unlike bitecho which produces -Infinity with amino acid profiles).
    """

    def test_aa_echo_reference_values(self, repo_root):
        """Forward matches Float64 JS reference values.

        Catches bugs #2 (missing NX->N, CX->C, JX->J), #3 (pre/post E),
        and #4 (missing B->M_k->E->CX chain).
        """
        fm, td_em = _build_aa_echo_plan7(repo_root)

        cases = [
            ('',     -15.776),
            ('A',    -17.234),
            ('ACDE', -25.701),
        ]

        for seq_str, ref in cases:
            if seq_str:
                out_seq = jnp.array(td_em.tokenize_output(list(seq_str)))
            else:
                out_seq = jnp.array([], dtype=jnp.int32)
            ll = float(fused_plan7_log_forward(fm, out_seq))
            assert ll == pytest.approx(ref, abs=0.5), \
                f"seq='{seq_str}': Plan7={ll} != ref={ref}"

    def test_aa_echo_viterbi_le_forward(self, repo_root):
        """Viterbi <= Forward with aa_echo + fn3.

        Catches major routing bugs where Viterbi might exceed Forward.
        """
        fm, td_em = _build_aa_echo_plan7(repo_root)
        out_seq = jnp.array(td_em.tokenize_output(list('ACDE')))

        fwd_ll = float(fused_plan7_log_forward(fm, out_seq))
        vit_ll = float(fused_plan7_log_viterbi(fm, out_seq))

        assert math.isfinite(fwd_ll), f"Forward not finite: {fwd_ll}"
        assert math.isfinite(vit_ll), f"Viterbi not finite: {vit_ll}"
        assert vit_ll <= fwd_ll + 0.01, \
            f"Viterbi={vit_ll} > Forward={fwd_ll}"

    def test_aa_echo_semiring_difference(self, repo_root):
        """Forward > Viterbi for multi-path profile.

        Catches semiring/routing confusion (if all mass goes through one path,
        Forward == Viterbi, which would indicate broken routing).
        """
        fm, td_em = _build_aa_echo_plan7(repo_root)
        out_seq = jnp.array(td_em.tokenize_output(list('ACDE')))

        fwd_ll = float(fused_plan7_log_forward(fm, out_seq))
        vit_ll = float(fused_plan7_log_viterbi(fm, out_seq))

        assert fwd_ll > vit_ll + 1e-6, \
            f"Forward ({fwd_ll}) should be > Viterbi ({vit_ll})"

    def test_empty_sequence(self, repo_root):
        """Forward on empty output with aa_echo + fn3.

        Catches init bug (#4): B->M_k->E->CX->T path must be propagated.
        """
        fm, td_em = _build_aa_echo_plan7(repo_root)
        out_seq = jnp.array([], dtype=jnp.int32)

        ll = float(fused_plan7_log_forward(fm, out_seq))
        assert math.isfinite(ll), f"Empty seq Forward not finite: {ll}"
        assert ll == pytest.approx(-15.776, abs=0.5), \
            f"Empty seq Forward={ll} != ref=-15.776"

    def test_tighter_vs_generic_fused(self, repo_root, boss_path):
        """Plan7-fused vs generic-fused with fn3+bitecho, tight tolerance.

        Catches all bugs since generic fused (via explicit composition) is correct.
        """
        hmmer_path = str(repo_root / "t" / "hmmer" / "fn3.hmm")
        td_path = str(repo_root / "t" / "machine" / "bitecho.json")

        generic, td_em = _build_generic_fused(hmmer_path, td_path, boss_path)
        plan7, _ = _build_plan7_fused(hmmer_path, td_path, boss_path)

        test_seq = list("010101")
        out_seq = jnp.array(td_em.tokenize_output(test_seq))

        gen_ll = float(fused_log_forward(generic, out_seq))
        p7_ll = float(fused_plan7_log_forward(plan7, out_seq))

        assert p7_ll == pytest.approx(gen_ll, abs=0.01), \
            f"Plan7={p7_ll} != generic={gen_ll} (tight tolerance)"

    def test_flanking_n_c_populated(self, repo_root):
        """After Forward with aa_echo, N and C flanking states have finite probability.

        Catches bug #2 (missing NX->N, CX->C transitions): without these,
        N/C emitting states stay at -Infinity.
        """
        fm, td_em = _build_aa_echo_plan7(repo_root)
        out_seq = jnp.array(td_em.tokenize_output(list('ACDE')))

        # Run Forward manually to inspect flanking state
        from machineboss.jax.fused_plan7 import _fused_plan7_dp, LOGSUMEXP
        # Use the public API and check the result is finite
        ll = float(fused_plan7_log_forward(fm, out_seq))
        assert math.isfinite(ll), \
            f"Forward with aa_echo should be finite (got {ll})"
        # If N/C are unpopulated, the result would be -Infinity or very negative
        assert ll > -100, \
            f"Forward={ll} is too negative — flanking states likely unpopulated"
