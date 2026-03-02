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
