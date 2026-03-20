"""Tests for parameterized (neural) DP on TransMachine."""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.jax.types import NEG_INF
from machineboss.jax.jax_weight import ParameterizedMachine
from machineboss.jax.dp_neural import neural_log_forward as existing_neural_forward
from machineboss.jax.trans.parameterized import ParameterizedTransMachine
from machineboss.jax.trans.dp_neural import neural_forward_2d


class TestParameterizedTransMachine:
    """Test ParameterizedTransMachine construction and weight evaluation."""

    def test_from_machine(self, repo_root):
        path = repo_root / "t" / "machine" / "bitnoise.json"
        m = Machine.from_file(str(path))
        ptm = ParameterizedTransMachine.from_machine(m)
        assert ptm.n_states == m.n_states
        assert ptm._n_transitions > 0

    def test_build_log_w(self, repo_root):
        import json
        path = repo_root / "t" / "machine" / "bitnoise.json"
        params_path = repo_root / "t" / "io" / "params.json"
        m = Machine.from_file(str(path))
        with open(params_path) as f:
            params = json.load(f)

        ptm = ParameterizedTransMachine.from_machine(m)

        # Make params broadcastable
        params_2d = {k: jnp.array([[v]]) for k, v in params.items()
                     if k in ptm.free_params}
        log_w = ptm.build_log_w(params_2d, 0, 0)
        assert log_w.shape == (ptm._n_transitions,)
        assert jnp.all(jnp.isfinite(log_w))


class TestNeuralForward2D:
    """Test neural forward matches existing implementation."""

    def test_matches_existing(self, repo_root):
        import json
        path = repo_root / "t" / "machine" / "bitnoise.json"
        params_path = repo_root / "t" / "io" / "params.json"
        m = Machine.from_file(str(path))
        with open(params_path) as f:
            raw_params = json.load(f)

        # Existing
        pm = ParameterizedMachine.from_machine(m)
        # TransMachine-based
        ptm = ParameterizedTransMachine.from_machine(m)

        # Token sequences
        in_tokens = jnp.array(pm.tokenize_input(list("10")))
        out_tokens = jnp.array(pm.tokenize_output(list("01")))

        # Build PSWMs
        Li = len(in_tokens)
        Lo = len(out_tokens)
        in_pswm = jnp.full((Li, pm.n_input_tokens), NEG_INF)
        in_pswm = in_pswm.at[jnp.arange(Li), in_tokens].set(0.0)
        out_pswm = jnp.full((Lo, pm.n_output_tokens), NEG_INF)
        out_pswm = out_pswm.at[jnp.arange(Lo), out_tokens].set(0.0)

        # Position-independent params
        params_2d = {k: jnp.array([[v]]) for k, v in raw_params.items()
                     if k in pm.free_params}

        from machineboss.jax.semiring import LOGSUMEXP

        result_existing = float(existing_neural_forward(
            pm, in_pswm, out_pswm, params_2d))
        result_trans = float(neural_forward_2d(
            ptm, in_pswm, out_pswm, params_2d, LOGSUMEXP))

        assert result_trans == pytest.approx(result_existing, abs=0.1)
