"""Systematic cross-validation between TransMachine and existing implementations.

Ensures every algorithm in trans/ produces identical results to the
corresponding algorithm in the original jax/ package.
"""

import json
import subprocess

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.semiring import LOGSUMEXP, MAXPLUS
from machineboss.jax.forward import log_forward
from machineboss.jax.viterbi import log_viterbi
from machineboss.jax.backward import log_backward_matrix
from machineboss.jax.trans.machine import TransMachine
from machineboss.jax.trans.dp_1d import forward_1d, backward_1d, viterbi_1d
from machineboss.jax.trans.dp_2d import forward_2d, backward_2d, viterbi_2d


def _boss_loglike(boss_path, machine_file, input_str=None, output_str=None,
                  params_file=None):
    args = [boss_path, machine_file]
    if input_str:
        args += ["--input-chars", input_str]
    if output_str:
        args += ["--output-chars", output_str]
    if params_file:
        args += ["-P", params_file]
    args.append("-L")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"boss failed: {result.stderr}")
    data = json.loads(result.stdout)
    if isinstance(data, list):
        return float(data[0][-1]) if isinstance(data[0], list) else float(data[0])
    return float(data)


class TestDispatcherAcceptsTransMachine:
    """Test that top-level dispatchers accept TransMachine directly."""

    def test_log_forward_with_transmachine(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        result_tm = float(log_forward(bitecho_tm, in_seq, out_seq))
        result_jm = float(log_forward(bitecho_jm, in_seq, out_seq))

        assert result_tm == pytest.approx(result_jm, abs=0.01)

    def test_log_viterbi_with_transmachine(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        result_tm = float(log_viterbi(bitecho_tm, in_seq, out_seq))
        result_jm = float(log_viterbi(bitecho_jm, in_seq, out_seq))

        assert result_tm == pytest.approx(result_jm, abs=0.01)

    def test_log_backward_with_transmachine(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        bp_tm = log_backward_matrix(bitecho_tm, in_seq, out_seq)
        bp_jm = log_backward_matrix(bitecho_jm, in_seq, out_seq)

        # Compare only finite values
        finite_mask = (bp_tm > -1e30) | (bp_jm > -1e30)
        if jnp.any(finite_mask):
            assert jnp.allclose(bp_tm[finite_mask], bp_jm[finite_mask], atol=0.1)


class TestCrossValidation2D:
    """Cross-validate 2D algorithms between trans/ and existing."""

    def test_forward_bitecho_all_strategies(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        existing = float(log_forward(bitecho_jm, in_seq, out_seq))
        trans_simple = float(forward_2d(bitecho_tm, in_seq, out_seq, strategy='simple'))
        trans_optimal = float(forward_2d(bitecho_tm, in_seq, out_seq, strategy='optimal'))

        assert trans_simple == pytest.approx(existing, abs=0.01)
        assert trans_optimal == pytest.approx(existing, abs=0.01)

    def test_viterbi_bitecho(self, bitecho_tm, bitecho_jm, bitecho_em):
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        existing = float(log_viterbi(bitecho_jm, in_seq, out_seq))
        trans_result = float(viterbi_2d(bitecho_tm, in_seq, out_seq))

        assert trans_result == pytest.approx(existing, abs=0.01)

    def test_forward_matches_boss(self, repo_root, boss_path, bitecho_tm, bitecho_em):
        machine_path = str(repo_root / "t" / "machine" / "bitecho.json")
        in_seq = jnp.array(bitecho_em.tokenize_input(list("101")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("101")))

        result = float(forward_2d(bitecho_tm, in_seq, out_seq))
        boss_ll = _boss_loglike(boss_path, machine_path,
                                input_str="101", output_str="101")
        assert result == pytest.approx(boss_ll, abs=0.01)

    def test_bitnoise_forward(self, repo_root, boss_path, bitnoise_tm):
        machine_path = str(repo_root / "t" / "machine" / "bitnoise.json")
        params_path = str(repo_root / "t" / "io" / "params.json")

        with open(params_path) as f:
            params = json.load(f)

        em = EvaluatedMachine.from_machine(
            Machine.from_file(machine_path), params)
        tok_map_in = {s: i for i, s in enumerate(em.input_tokens)}
        tok_map_out = {s: i for i, s in enumerate(em.output_tokens)}

        in_seq = jnp.array([tok_map_in['1'], tok_map_in['0'], tok_map_in['1']])
        out_seq = jnp.array([tok_map_out['0'], tok_map_out['1'], tok_map_out['1']])

        result = float(forward_2d(bitnoise_tm, in_seq, out_seq, strategy='simple'))
        boss_ll = _boss_loglike(boss_path, machine_path,
                                input_str="101", output_str="011",
                                params_file=params_path)
        assert result == pytest.approx(boss_ll, abs=0.1)

    def test_backward_consistency(self, bitecho_tm, bitecho_em):
        """Forward and backward should give same log-likelihood."""
        in_seq = jnp.array(bitecho_em.tokenize_input(list("10")))
        out_seq = jnp.array(bitecho_em.tokenize_output(list("10")))

        fwd = float(forward_2d(bitecho_tm, in_seq, out_seq, strategy='simple'))
        bp = backward_2d(bitecho_tm, in_seq, out_seq, strategy='simple')
        bwd_ll = float(bp[0, 0, 0])

        assert fwd == pytest.approx(bwd_ll, abs=0.1)
