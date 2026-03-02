"""Tests for parameterized 2D DP with position-dependent weight expressions."""

import json
import math

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.seq import PSWMSeq
from machineboss.jax.forward import log_forward
from machineboss.jax.viterbi import log_viterbi
from machineboss.jax.backward import log_backward_matrix
from machineboss.jax.jax_weight import ParameterizedMachine, compile_expr
from machineboss.jax.dp_neural import (
    neural_log_forward, neural_log_viterbi, neural_log_backward_matrix,
)


class TestCompileExpr:
    """Test weight expression compilation to JAX."""

    def test_constant(self):
        fn = compile_expr(0.5)
        assert float(fn({})) == pytest.approx(0.5)

    def test_parameter(self):
        fn = compile_expr("p")
        assert float(fn({"p": jnp.float32(0.9)})) == pytest.approx(0.9)

    def test_multiply(self):
        fn = compile_expr({"*": ["p", 0.5]})
        assert float(fn({"p": jnp.float32(0.8)})) == pytest.approx(0.4)

    def test_add(self):
        fn = compile_expr({"+": [0.3, "q"]})
        assert float(fn({"q": jnp.float32(0.7)})) == pytest.approx(1.0)

    def test_subtract(self):
        fn = compile_expr({"-": [1, "p"]})
        assert float(fn({"p": jnp.float32(0.3)})) == pytest.approx(0.7)

    def test_divide(self):
        fn = compile_expr({"/": ["p", 2]})
        assert float(fn({"p": jnp.float32(0.8)})) == pytest.approx(0.4)

    def test_log(self):
        fn = compile_expr({"log": "p"})
        val = float(fn({"p": jnp.float32(math.e)}))
        assert val == pytest.approx(1.0, abs=1e-5)

    def test_exp(self):
        fn = compile_expr({"exp": 1.0})
        assert float(fn({})) == pytest.approx(math.e, abs=1e-5)

    def test_not(self):
        fn = compile_expr({"not": "p"})
        assert float(fn({"p": jnp.float32(0.3)})) == pytest.approx(0.7)

    def test_pow(self):
        fn = compile_expr({"pow": [2, 3]})
        assert float(fn({})) == pytest.approx(8.0)

    def test_nested(self):
        # p * (1 - q) + 0.1
        expr = {"+": [{"*": ["p", {"-": [1, "q"]}]}, 0.1]}
        fn = compile_expr(expr)
        p, q = 0.8, 0.3
        expected = p * (1 - q) + 0.1
        result = float(fn({"p": jnp.float32(p), "q": jnp.float32(q)}))
        assert result == pytest.approx(expected, abs=1e-5)


class TestParameterizedMachine:
    """Test ParameterizedMachine compilation."""

    def test_from_bitnoise(self, repo_root):
        """Compile bitnoise.json and verify structure."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        machine = Machine.from_file(td_path)
        pm = ParameterizedMachine.from_machine(machine)

        assert pm.n_states == 1
        assert pm.param_names == {"p", "q"}
        assert len(pm.input_tokens) > 1  # has input alphabet
        assert len(pm.output_tokens) > 1  # has output alphabet

    def test_build_log_trans_matches_evaluated(self, repo_root):
        """Constant params should match EvaluatedMachine's log_trans."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")
        with open(params_path) as f:
            params = json.load(f)

        machine = Machine.from_file(td_path)

        # EvaluatedMachine approach
        em = EvaluatedMachine.from_machine(machine, params)
        jm = JAXMachine.from_evaluated(em)

        # ParameterizedMachine approach
        pm = ParameterizedMachine.from_machine(machine)
        jax_params = {k: jnp.float32(v) for k, v in params.items()}
        lt = pm.build_log_trans(jax_params)

        # Compare transition tensors
        assert jm.log_trans is not None
        assert lt.shape == jm.log_trans.shape
        # Allow tolerance for float32 vs float64 differences
        import numpy.testing as npt
        mask = jm.log_trans > -1e30
        npt.assert_allclose(lt[mask], jm.log_trans[mask], atol=0.01)


class TestNeuralForwardConstantParams:
    """Test neural forward with constant (position-independent) params.

    With constant params, neural_log_forward should match log_forward.
    """

    def _make_constant_params(self, param_dict, Li, Lo):
        """Broadcast scalar params to (Li+1, Lo+1) tensors."""
        return {
            name: jnp.full((Li + 1, Lo + 1), val)
            for name, val in param_dict.items()
        }

    def _make_one_hot_pswm(self, tokens, n_tokens):
        """Build one-hot PSWM from token indices (equivalent to TokenSeq)."""
        from machineboss.jax.types import NEG_INF
        L = len(tokens)
        pswm = jnp.full((L, n_tokens), NEG_INF)
        pswm = pswm.at[jnp.arange(L), jnp.array(tokens)].set(0.0)
        return pswm

    def test_bitnoise_forward(self, repo_root):
        """Neural forward with constant params matches standard forward."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")
        with open(params_path) as f:
            params = json.load(f)

        machine = Machine.from_file(td_path)
        em = EvaluatedMachine.from_machine(machine, params)
        jm = JAXMachine.from_evaluated(em)

        in_str = "101"
        out_str = "010"
        in_seq = jnp.array(em.tokenize_input(list(in_str)))
        out_seq = jnp.array(em.tokenize_output(list(out_str)))

        # Standard forward
        ref_ll = float(log_forward(jm, in_seq, out_seq, strategy='simple',
                                    kernel='dense'))

        # Neural forward
        pm = ParameterizedMachine.from_machine(machine)
        Li, Lo = len(in_str), len(out_str)
        in_pswm = self._make_one_hot_pswm(
            em.tokenize_input(list(in_str)), pm.n_input_tokens)
        out_pswm = self._make_one_hot_pswm(
            em.tokenize_output(list(out_str)), pm.n_output_tokens)
        const_params = self._make_constant_params(params, Li, Lo)

        neural_ll = float(neural_log_forward(pm, in_pswm, out_pswm, const_params))

        assert neural_ll == pytest.approx(ref_ll, abs=0.01), \
            f"neural={neural_ll} != ref={ref_ll}"

    def test_bitnoise_viterbi(self, repo_root):
        """Neural Viterbi with constant params matches standard Viterbi."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")
        with open(params_path) as f:
            params = json.load(f)

        machine = Machine.from_file(td_path)
        em = EvaluatedMachine.from_machine(machine, params)
        jm = JAXMachine.from_evaluated(em)

        in_str = "10"
        out_str = "01"
        in_seq = jnp.array(em.tokenize_input(list(in_str)))
        out_seq = jnp.array(em.tokenize_output(list(out_str)))

        ref_vit = float(log_viterbi(jm, in_seq, out_seq, strategy='simple',
                                     kernel='dense'))

        pm = ParameterizedMachine.from_machine(machine)
        Li, Lo = len(in_str), len(out_str)
        in_pswm = self._make_one_hot_pswm(
            em.tokenize_input(list(in_str)), pm.n_input_tokens)
        out_pswm = self._make_one_hot_pswm(
            em.tokenize_output(list(out_str)), pm.n_output_tokens)
        const_params = self._make_constant_params(params, Li, Lo)

        neural_vit = float(neural_log_viterbi(pm, in_pswm, out_pswm, const_params))

        assert neural_vit == pytest.approx(ref_vit, abs=0.01), \
            f"neural={neural_vit} != ref={ref_vit}"

    def test_bitnoise_backward_consistency(self, repo_root):
        """Backward[start] should equal Forward log-likelihood."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")
        with open(params_path) as f:
            params = json.load(f)

        machine = Machine.from_file(td_path)
        em = EvaluatedMachine.from_machine(machine, params)

        in_str = "10"
        out_str = "01"

        pm = ParameterizedMachine.from_machine(machine)
        Li, Lo = len(in_str), len(out_str)
        in_pswm = self._make_one_hot_pswm(
            em.tokenize_input(list(in_str)), pm.n_input_tokens)
        out_pswm = self._make_one_hot_pswm(
            em.tokenize_output(list(out_str)), pm.n_output_tokens)
        const_params = self._make_constant_params(params, Li, Lo)

        fwd_ll = float(neural_log_forward(pm, in_pswm, out_pswm, const_params))
        bp = neural_log_backward_matrix(pm, in_pswm, out_pswm, const_params)
        bwd_ll = float(bp[0, 0, 0])  # backward at start state

        assert bwd_ll == pytest.approx(fwd_ll, abs=0.01), \
            f"backward={bwd_ll} != forward={fwd_ll}"

    def test_viterbi_le_forward(self, repo_root):
        """Viterbi <= Forward invariant."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")
        with open(params_path) as f:
            params = json.load(f)

        machine = Machine.from_file(td_path)
        em = EvaluatedMachine.from_machine(machine, params)

        in_str = "101"
        out_str = "010"

        pm = ParameterizedMachine.from_machine(machine)
        Li, Lo = len(in_str), len(out_str)
        in_pswm = self._make_one_hot_pswm(
            em.tokenize_input(list(in_str)), pm.n_input_tokens)
        out_pswm = self._make_one_hot_pswm(
            em.tokenize_output(list(out_str)), pm.n_output_tokens)
        const_params = self._make_constant_params(params, Li, Lo)

        fwd_ll = float(neural_log_forward(pm, in_pswm, out_pswm, const_params))
        vit_ll = float(neural_log_viterbi(pm, in_pswm, out_pswm, const_params))

        assert vit_ll <= fwd_ll + 0.01, \
            f"Viterbi={vit_ll} > Forward={fwd_ll}"


class TestNeuralForwardVaryingParams:
    """Test neural forward with position-varying parameters."""

    def test_position_dependent_params(self, repo_root):
        """Different params at each position should give different results."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")

        machine = Machine.from_file(td_path)
        em = EvaluatedMachine.from_machine(machine, {"p": 0.99, "q": 0.01})
        pm = ParameterizedMachine.from_machine(machine)

        in_str = "10"
        out_str = "01"
        Li, Lo = len(in_str), len(out_str)

        from machineboss.jax.types import NEG_INF
        in_pswm = jnp.full((Li, pm.n_input_tokens), NEG_INF)
        in_toks = em.tokenize_input(list(in_str))
        in_pswm = in_pswm.at[jnp.arange(Li), jnp.array(in_toks)].set(0.0)

        out_pswm = jnp.full((Lo, pm.n_output_tokens), NEG_INF)
        out_toks = em.tokenize_output(list(out_str))
        out_pswm = out_pswm.at[jnp.arange(Lo), jnp.array(out_toks)].set(0.0)

        # Constant params (high p = low noise)
        const_params = {
            "p": jnp.full((Li + 1, Lo + 1), 0.99),
            "q": jnp.full((Li + 1, Lo + 1), 0.01),
        }
        ll_const = float(neural_log_forward(pm, in_pswm, out_pswm, const_params))

        # Varying params (high noise everywhere)
        noisy_params = {
            "p": jnp.full((Li + 1, Lo + 1), 0.5),
            "q": jnp.full((Li + 1, Lo + 1), 0.5),
        }
        ll_noisy = float(neural_log_forward(pm, in_pswm, out_pswm, noisy_params))

        # With high noise (p=0.5), the likelihood should be different from low noise
        assert ll_const != pytest.approx(ll_noisy, abs=0.1), \
            f"Different params gave same result: {ll_const} vs {ll_noisy}"

    def test_jax_grad_through_params(self, repo_root):
        """Verify JAX can differentiate through position-dependent params."""
        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")

        machine = Machine.from_file(td_path)
        em = EvaluatedMachine.from_machine(machine, {"p": 0.99, "q": 0.01})
        pm = ParameterizedMachine.from_machine(machine)

        # Use matching sequences so "p" parameter is used on the diagonal path
        in_str = "10"
        out_str = "10"
        Li, Lo = len(in_str), len(out_str)

        from machineboss.jax.types import NEG_INF
        in_pswm = jnp.full((Li, pm.n_input_tokens), NEG_INF)
        in_toks = em.tokenize_input(list(in_str))
        in_pswm = in_pswm.at[jnp.arange(Li), jnp.array(in_toks)].set(0.0)

        out_pswm = jnp.full((Lo, pm.n_output_tokens), NEG_INF)
        out_toks = em.tokenize_output(list(out_str))
        out_pswm = out_pswm.at[jnp.arange(Lo), jnp.array(out_toks)].set(0.0)

        # Differentiate w.r.t. the p parameter tensor
        p_tensor = jnp.full((Li + 1, Lo + 1), 0.9)
        q_tensor = jnp.full((Li + 1, Lo + 1), 0.1)

        def loss_fn(p_t):
            params = {"p": p_t, "q": q_tensor}
            return neural_log_forward(pm, in_pswm, out_pswm, params)

        grad_p = jax.grad(loss_fn)(p_tensor)

        # Gradient should exist and be finite
        assert grad_p.shape == (Li + 1, Lo + 1)
        assert jnp.all(jnp.isfinite(grad_p)), \
            f"Non-finite gradients: {grad_p}"
        # Gradient should be non-zero (changing p affects the likelihood)
        assert jnp.any(grad_p != 0.0), "Gradient is all zeros"

    def test_vs_boss_forward(self, repo_root, boss_path):
        """Compare neural forward (constant params) vs C++ boss -L."""
        import subprocess

        td_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")
        with open(params_path) as f:
            params = json.load(f)

        in_str = "101"
        out_str = "010"

        # C++ reference
        result = subprocess.run(
            [boss_path, td_path,
             "--input-chars", in_str,
             "--output-chars", out_str,
             "-P", params_path, "-L"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"boss failed: {result.stderr}")
        data = json.loads(result.stdout)
        boss_ll = float(data[0][-1]) if isinstance(data[0], list) else float(data[0])

        # Neural forward
        machine = Machine.from_file(td_path)
        em = EvaluatedMachine.from_machine(machine, params)
        pm = ParameterizedMachine.from_machine(machine)

        Li, Lo = len(in_str), len(out_str)
        from machineboss.jax.types import NEG_INF
        in_pswm = jnp.full((Li, pm.n_input_tokens), NEG_INF)
        in_toks = em.tokenize_input(list(in_str))
        in_pswm = in_pswm.at[jnp.arange(Li), jnp.array(in_toks)].set(0.0)

        out_pswm = jnp.full((Lo, pm.n_output_tokens), NEG_INF)
        out_toks = em.tokenize_output(list(out_str))
        out_pswm = out_pswm.at[jnp.arange(Lo), jnp.array(out_toks)].set(0.0)

        const_params = {
            name: jnp.full((Li + 1, Lo + 1), val)
            for name, val in params.items()
        }

        neural_ll = float(neural_log_forward(pm, in_pswm, out_pswm, const_params))

        assert neural_ll == pytest.approx(boss_ll, abs=0.01), \
            f"neural={neural_ll} != boss={boss_ll}"
