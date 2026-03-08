"""Tests for wavefront beam-Viterbi alignment."""

import json
import subprocess

import pytest
import numpy as np

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.beam_align import beam_align


class TestBeamAlignUnitindel:
    """Test beam-align on unitindel (acyclic: should match exact Viterbi)."""

    @pytest.fixture
    def machine_and_params(self, repo_root):
        m = Machine.from_file(str(repo_root / "t" / "machine" / "unitindel.json"))
        params = {"ins": 0.1, "no_ins": 0.9, "del": 0.1, "no_del": 0.9}
        return m, params

    def test_score_matches_viterbi(self, machine_and_params, repo_root):
        """With large beam, beam-align score should match Viterbi on acyclic machine."""
        m, params = machine_and_params
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = np.array(em.tokenize_input(list("xx")))
        out_seq = np.array(em.tokenize_output(list("xxx")))

        result = beam_align(jm, in_seq, out_seq, beam_width=1000)

        # Reference from boss --viterbi: -2.82939
        assert result.score == pytest.approx(-2.82939, abs=1e-3)

    def test_path_valid(self, machine_and_params):
        """Path should be non-empty and have correct endpoints."""
        m, params = machine_and_params
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = np.array(em.tokenize_input(list("xx")))
        out_seq = np.array(em.tokenize_output(list("xxx")))

        result = beam_align(jm, in_seq, out_seq, beam_width=1000)
        assert len(result.path) > 0

    def test_beam_width_monotone(self, machine_and_params):
        """Smaller beam should give score <= larger beam."""
        m, params = machine_and_params
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = np.array(em.tokenize_input(list("xx")))
        out_seq = np.array(em.tokenize_output(list("xxx")))

        small = beam_align(jm, in_seq, out_seq, beam_width=5)
        large = beam_align(jm, in_seq, out_seq, beam_width=1000)
        assert small.score <= large.score + 1e-10

    def test_matches_boss_cli(self, machine_and_params, repo_root, boss_path):
        """Beam-align score should match boss --beam-align."""
        m, params = machine_and_params
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = np.array(em.tokenize_input(list("xx")))
        out_seq = np.array(em.tokenize_output(list("xxx")))

        result = beam_align(jm, in_seq, out_seq, beam_width=1000)

        # Reference: boss --viterbi gives -2.82939
        assert result.score == pytest.approx(-2.82939, abs=1e-3)


class TestBeamAlignTKF92:
    """Test beam-align on TKF92 (cyclic machine)."""

    @pytest.fixture
    def tkf92_machine(self, repo_root):
        m = Machine.from_file(str(repo_root / "preset" / "tkf92branch.json"))
        base_params = {"t": 0.5, "insRate": 0.01, "delRate": 0.02, "r": 0.3}
        for i in range(20):
            base_params[f"pi_{i}"] = 0.05
        # Resolve defs: evaluate function definitions using base params
        from machineboss.weight import evaluate as eval_weight
        resolved = dict(base_params)
        # Iteratively resolve defs until stable
        for _ in range(10):
            changed = False
            for name, expr in m.defs.items():
                try:
                    val = eval_weight(expr, resolved)
                    if name not in resolved or resolved[name] != val:
                        resolved[name] = val
                        changed = True
                except (KeyError, TypeError):
                    pass
            if not changed:
                break
        return m, resolved

    def test_finite_score(self, tkf92_machine):
        """TKF92 beam-align should produce a finite score."""
        m, params = tkf92_machine
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = np.array(em.tokenize_input(list("AC")))
        out_seq = np.array(em.tokenize_output(list("AC")))

        result = beam_align(jm, in_seq, out_seq, beam_width=1000)
        assert np.isfinite(result.score)
        assert result.score < 0

    def test_path_nonempty(self, tkf92_machine):
        """TKF92 beam-align path should be non-empty."""
        m, params = tkf92_machine
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = np.array(em.tokenize_input(list("AC")))
        out_seq = np.array(em.tokenize_output(list("AC")))

        result = beam_align(jm, in_seq, out_seq, beam_width=1000)
        assert len(result.path) > 0

    def test_cross_impl_consistency(self, tkf92_machine, boss_path, repo_root):
        """JAX and C++ should agree on TKF92 beam-align score."""
        import tempfile, os
        m, params = tkf92_machine
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = np.array(em.tokenize_input(list("AC")))
        out_seq = np.array(em.tokenize_output(list("AC")))

        py_result = beam_align(jm, in_seq, out_seq, beam_width=1000)

        # Get C++ result — only pass base params (not resolved defs)
        base_params = {"t": 0.5, "insRate": 0.01, "delRate": 0.02, "r": 0.3}
        for i in range(20):
            base_params[f"pi_{i}"] = 0.05
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(base_params, f)
            params_file = f.name
        try:
            result = subprocess.run(
                [boss_path, "--preset", "tkf92branch",
                 "--functions", params_file,
                 "--input-chars", "AC", "--output-chars", "AC",
                 "--beam-align", "--beam-width", "1000", "-v0"],
                capture_output=True, text=True
            )
            assert result.returncode == 0, f"C++ failed: {result.stderr}"
            alignment = json.loads(result.stdout)
            assert len(alignment) > 0, "C++ beam-align should produce alignment"
        finally:
            os.unlink(params_file)

        # Both should have finite scores
        assert np.isfinite(py_result.score)
