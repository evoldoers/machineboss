"""Tests for fused Plan7+transducer Forward algorithm."""

import json
import subprocess

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.forward import log_forward
from machineboss.jax.fused import FusedMachine, fused_log_forward, fused_log_viterbi


def _boss_loglike(boss_path, *args):
    """Get log-likelihood from boss CLI."""
    cmd = [boss_path] + list(args) + ["-L"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"boss failed: {result.stderr}")
    data = json.loads(result.stdout)
    if isinstance(data, list):
        return float(data[0][-1]) if isinstance(data[0], list) else float(data[0])
    return float(data)


class TestFusedForward:
    """Test fused Forward against explicit composition."""

    def test_generator_identity_match(self, repo_root, boss_path):
        """gen("01") + bitecho, output "01" -> loglike ~ 0."""
        td_path = str(repo_root / "t" / "machine" / "bitecho.json")

        # Reference from boss
        ref_ll = _boss_loglike(
            boss_path,
            "--generate-chars", "01",
            "--compose", td_path,
            "--output-chars", "01"
        )

        # Fused
        gen_result = subprocess.run(
            [boss_path, "--generate-chars", "01"],
            capture_output=True, text=True
        )
        gen_em = EvaluatedMachine.from_machine(Machine.from_json(gen_result.stdout))
        td_em = EvaluatedMachine.from_machine(Machine.from_file(td_path))

        fused = FusedMachine.build(gen_em, td_em)
        out_seq = jnp.array(td_em.tokenize_output(list("01")))
        fused_ll = float(fused_log_forward(fused, out_seq))

        assert ref_ll == pytest.approx(0.0, abs=0.01)
        assert fused_ll == pytest.approx(ref_ll, abs=0.01)

    def test_generator_identity_mismatch(self, repo_root, boss_path):
        """gen("01") + bitecho, output "10" -> very low probability."""
        td_path = str(repo_root / "t" / "machine" / "bitecho.json")

        gen_result = subprocess.run(
            [boss_path, "--generate-chars", "01"],
            capture_output=True, text=True
        )
        gen_em = EvaluatedMachine.from_machine(Machine.from_json(gen_result.stdout))
        td_em = EvaluatedMachine.from_machine(Machine.from_file(td_path))

        fused = FusedMachine.build(gen_em, td_em)
        out_seq = jnp.array(td_em.tokenize_output(list("10")))
        fused_ll = float(fused_log_forward(fused, out_seq))

        assert fused_ll < -30

    def test_generator_noisy(self, repo_root, boss_path):
        """gen("101") + bitnoise(p,q), output "001" -> fused matches boss."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        # Reference from boss
        ref_ll = _boss_loglike(
            boss_path,
            "--generate-chars", "101",
            "--compose", td_path,
            "-P", params_path,
            "--output-chars", "001"
        )

        # Fused
        gen_result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True
        )
        gen_em = EvaluatedMachine.from_machine(Machine.from_json(gen_result.stdout))

        with open(params_path) as f:
            params = json.load(f)
        td_em = EvaluatedMachine.from_machine(Machine.from_file(td_path), params)

        fused = FusedMachine.build(gen_em, td_em)
        out_seq = jnp.array(td_em.tokenize_output(list("001")))
        fused_ll = float(fused_log_forward(fused, out_seq))

        assert fused_ll == pytest.approx(ref_ll, abs=0.01)

    def test_fused_vs_python_forward(self, repo_root, boss_path):
        """Fused Forward should match Python Forward on composed machine."""
        td_path = str(repo_root / "t" / "machine" / "bitecho.json")

        # Compose via boss, then evaluate
        compose_result = subprocess.run(
            [boss_path, "--generate-chars", "01",
             "--compose", td_path, "--evaluate"],
            capture_output=True, text=True
        )
        composed = Machine.from_json(compose_result.stdout)
        composed_em = EvaluatedMachine.from_machine(composed)
        composed_jm = JAXMachine.from_evaluated(composed_em)

        in_seq = jnp.array([], dtype=jnp.int32)
        out_seq = jnp.array(composed_em.tokenize_output(list("01")))
        ref_ll = float(log_forward(composed_jm, in_seq, out_seq))

        # Fused
        gen_result = subprocess.run(
            [boss_path, "--generate-chars", "01"],
            capture_output=True, text=True
        )
        gen_em = EvaluatedMachine.from_machine(Machine.from_json(gen_result.stdout))
        td_em = EvaluatedMachine.from_machine(Machine.from_file(td_path))

        fused = FusedMachine.build(gen_em, td_em)
        out_seq_fused = jnp.array(td_em.tokenize_output(list("01")))
        fused_ll = float(fused_log_forward(fused, out_seq_fused))

        assert fused_ll == pytest.approx(ref_ll, abs=0.01)

    def test_noisy_different_output(self, repo_root, boss_path):
        """gen("101") + bitnoise, output "111" -> fused matches boss."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        ref_ll = _boss_loglike(
            boss_path,
            "--generate-chars", "101",
            "--compose", td_path,
            "-P", params_path,
            "--output-chars", "111"
        )

        gen_result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True
        )
        gen_em = EvaluatedMachine.from_machine(Machine.from_json(gen_result.stdout))

        with open(params_path) as f:
            params = json.load(f)
        td_em = EvaluatedMachine.from_machine(Machine.from_file(td_path), params)

        fused = FusedMachine.build(gen_em, td_em)
        out_seq = jnp.array(td_em.tokenize_output(list("111")))
        fused_ll = float(fused_log_forward(fused, out_seq))

        assert fused_ll == pytest.approx(ref_ll, abs=0.01)

    def test_longer_sequence(self, repo_root, boss_path):
        """Fused with a longer generated sequence."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        # Reference from boss
        ref_ll = _boss_loglike(
            boss_path,
            "--generate-chars", "11001",
            "--compose", td_path,
            "-P", params_path,
            "--output-chars", "11001"
        )

        # Fused
        gen_result = subprocess.run(
            [boss_path, "--generate-chars", "11001"],
            capture_output=True, text=True
        )
        gen_em = EvaluatedMachine.from_machine(Machine.from_json(gen_result.stdout))

        with open(params_path) as f:
            params = json.load(f)
        td_em = EvaluatedMachine.from_machine(Machine.from_file(td_path), params)

        fused = FusedMachine.build(gen_em, td_em)
        out_seq = jnp.array(td_em.tokenize_output(list("11001")))
        fused_ll = float(fused_log_forward(fused, out_seq))

        assert fused_ll == pytest.approx(ref_ll, abs=0.01)


class TestFusedViterbi:
    """Test fused Viterbi against Forward (Viterbi <= Forward always)."""

    def test_viterbi_eq_forward_deterministic(self, repo_root, boss_path):
        """For deterministic gen + identity transducer, Viterbi = Forward."""
        td_path = str(repo_root / "t" / "machine" / "bitecho.json")

        gen_result = subprocess.run(
            [boss_path, "--generate-chars", "01"],
            capture_output=True, text=True
        )
        gen_em = EvaluatedMachine.from_machine(Machine.from_json(gen_result.stdout))
        td_em = EvaluatedMachine.from_machine(Machine.from_file(td_path))

        fused = FusedMachine.build(gen_em, td_em)
        out_seq = jnp.array(td_em.tokenize_output(list("01")))

        fwd_ll = float(fused_log_forward(fused, out_seq))
        vit_ll = float(fused_log_viterbi(fused, out_seq))

        assert vit_ll == pytest.approx(fwd_ll, abs=0.01)
        assert vit_ll == pytest.approx(0.0, abs=0.01)

    def test_viterbi_le_forward_noisy(self, repo_root, boss_path):
        """Viterbi <= Forward for noisy transducer."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        gen_result = subprocess.run(
            [boss_path, "--generate-chars", "101"],
            capture_output=True, text=True
        )
        gen_em = EvaluatedMachine.from_machine(Machine.from_json(gen_result.stdout))

        with open(params_path) as f:
            params = json.load(f)
        td_em = EvaluatedMachine.from_machine(Machine.from_file(td_path), params)

        fused = FusedMachine.build(gen_em, td_em)
        out_seq = jnp.array(td_em.tokenize_output(list("001")))

        fwd_ll = float(fused_log_forward(fused, out_seq))
        vit_ll = float(fused_log_viterbi(fused, out_seq))

        # Viterbi (best path) <= Forward (sum over all paths)
        assert vit_ll <= fwd_ll + 0.01

    def test_viterbi_mismatch(self, repo_root, boss_path):
        """Viterbi should be very negative for impossible output."""
        td_path = str(repo_root / "t" / "machine" / "bitecho.json")

        gen_result = subprocess.run(
            [boss_path, "--generate-chars", "01"],
            capture_output=True, text=True
        )
        gen_em = EvaluatedMachine.from_machine(Machine.from_json(gen_result.stdout))
        td_em = EvaluatedMachine.from_machine(Machine.from_file(td_path))

        fused = FusedMachine.build(gen_em, td_em)
        out_seq = jnp.array(td_em.tokenize_output(list("10")))
        vit_ll = float(fused_log_viterbi(fused, out_seq))

        assert vit_ll < -30
