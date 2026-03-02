"""Tests for JAX Forward-Backward and expected counts."""

import json
import subprocess

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.fwdback import log_likelihood_with_counts
from machineboss.jax.forward import log_forward


class TestFwdBack:
    """Test Forward-Backward consistency."""

    def test_ll_matches_forward(self, repo_root):
        """Log-likelihood from fwdback should match standalone forward."""
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("101")))

        fwd_ll = float(log_forward(jm, in_seq, out_seq))
        fb_ll, counts = log_likelihood_with_counts(jm, in_seq, out_seq)
        fb_ll = float(fb_ll)

        assert fb_ll == pytest.approx(fwd_ll, abs=0.01)

    def test_counts_nonnegative(self, repo_root):
        """Expected counts should be non-negative."""
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = Machine.from_file(machine_path)
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("10")))
        out_seq = jnp.array(em.tokenize_output(list("10")))

        ll, counts = log_likelihood_with_counts(jm, in_seq, out_seq)
        assert jnp.all(counts >= -0.01)

    def test_bitnoise_counts_vs_boss(self, repo_root, boss_path):
        """Expected per-parameter counts for bitnoise should match C++ boss -C."""
        machine_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        m = Machine.from_file(machine_path)
        with open(params_path) as f:
            params = json.load(f)
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)

        in_seq = jnp.array(em.tokenize_input(list("101")))
        out_seq = jnp.array(em.tokenize_output(list("001")))

        ll, counts = log_likelihood_with_counts(jm, in_seq, out_seq)

        # Verify log-likelihood matches boss
        result = subprocess.run(
            [boss_path, machine_path,
             "--input-chars", "101", "--output-chars", "001",
             "-P", params_path, "-L"],
            capture_output=True, text=True,
        )
        data = json.loads(result.stdout)
        boss_ll = float(data[0][-1]) if isinstance(data[0], list) else float(data[0])
        assert float(ll) == pytest.approx(boss_ll, abs=0.01)

        # Get per-parameter counts from boss -C
        import tempfile, os
        seqpair_data = json.dumps([{
            "input": {"name": "101", "sequence": list("101")},
            "output": {"name": "001", "sequence": list("001")},
        }])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(seqpair_data)
            seqpair_path = f.name
        try:
            result = subprocess.run(
                [boss_path, "-C", machine_path, "-P", params_path, "-D", seqpair_path],
                capture_output=True, text=True,
            )
            boss_counts = json.loads(result.stdout)  # {"p": 2, "q": 1}
        finally:
            os.unlink(seqpair_path)

        # Map JAX per-transition counts to per-parameter counts.
        # Extract weight expressions from the Machine JSON.
        weight_exprs = []
        for state in m.to_json()['state']:
            for t in state.get('trans', []):
                weight_exprs.append(t['weight'])

        # For each parameter, sum the counts of transitions that use it
        jax_param_counts = {}
        for i, expr in enumerate(weight_exprs):
            if isinstance(expr, str) and expr in params:
                jax_param_counts[expr] = jax_param_counts.get(expr, 0.0) + float(counts[i])

        for param, boss_val in boss_counts.items():
            jax_val = jax_param_counts.get(param, 0.0)
            assert jax_val == pytest.approx(boss_val, abs=0.01), \
                f"count for '{param}': JAX={jax_val} != boss={boss_val}"
