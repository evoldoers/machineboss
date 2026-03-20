"""Tests for TransMachine class."""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.trans.machine import TransMachine


class TestTransMachineConstruction:
    """Test TransMachine constructors."""

    def test_from_machine(self, bitecho_machine):
        tm = TransMachine.from_machine(bitecho_machine)
        assert tm.n_states == bitecho_machine.n_states
        assert tm.n_transitions > 0
        assert tm.is_transducer()

    def test_from_evaluated(self, bitecho_machine):
        em = EvaluatedMachine.from_machine(bitecho_machine)
        tm = TransMachine.from_evaluated(em)
        assert tm.n_states == em.n_states
        assert tm.n_in == len(em.input_tokens)
        assert tm.n_out == len(em.output_tokens)

    def test_from_jax_machine(self, bitecho_jm):
        tm = TransMachine.from_jax_machine(bitecho_jm)
        assert tm.n_states == bitecho_jm.n_states
        assert tm.n_in == bitecho_jm.n_input_tokens
        assert tm.n_out == bitecho_jm.n_output_tokens

    def test_masks_partition(self, bitecho_tm):
        """Masks should partition all transitions."""
        tm = bitecho_tm
        total = (tm.silent_mask.astype(int) + tm.emit_in_mask.astype(int) +
                 tm.emit_out_mask.astype(int) + tm.emit_both_mask.astype(int))
        assert jnp.all(total == 1)


class TestTransMachineRoundTrip:
    """Test round-trip conversions."""

    def test_to_jax_machine(self, bitecho_machine):
        em = EvaluatedMachine.from_machine(bitecho_machine)
        tm = TransMachine.from_evaluated(em)
        jm = tm.to_jax_machine()
        assert jm.n_states == tm.n_states
        assert jm.n_input_tokens == tm.n_in
        assert jm.n_output_tokens == tm.n_out
        assert jm.log_trans is not None

    def test_to_machine(self, bitecho_machine):
        em = EvaluatedMachine.from_machine(bitecho_machine)
        tm = TransMachine.from_evaluated(em)
        m2 = tm.to_machine()
        assert m2.n_states == bitecho_machine.n_states

    def test_jax_machine_roundtrip_preserves_weights(self, bitecho_machine):
        em = EvaluatedMachine.from_machine(bitecho_machine)
        jm = JAXMachine.from_evaluated(em)
        tm = TransMachine.from_jax_machine(jm)
        jm2 = tm.to_jax_machine()

        # Compare sparse arrays
        assert jnp.allclose(jm.log_weights, jm2.log_weights)
        assert jnp.array_equal(jm.src_states, jm2.src_states)
        assert jnp.array_equal(jm.dst_states, jm2.dst_states)

    def test_jax_machine_dense_matches(self, bitecho_machine):
        em = EvaluatedMachine.from_machine(bitecho_machine)
        jm = JAXMachine.from_evaluated(em)
        tm = TransMachine.from_jax_machine(jm)
        jm2 = tm.to_jax_machine()

        assert jnp.allclose(jm.log_trans, jm2.log_trans, atol=1e-5)


class TestTransMachinePytree:
    """Test JAX pytree registration."""

    def test_flatten_unflatten(self, bitecho_tm):
        children, aux = bitecho_tm.tree_flatten()
        tm2 = TransMachine.tree_unflatten(aux, children)
        assert tm2.n_states == bitecho_tm.n_states
        assert jnp.array_equal(tm2.src, bitecho_tm.src)
        assert jnp.array_equal(tm2.log_w, bitecho_tm.log_w)

    def test_jit_compatible(self, bitecho_tm):
        @jax.jit
        def f(tm):
            return tm.log_w.sum()
        result = f(bitecho_tm)
        assert jnp.isfinite(result)

    def test_machine_type(self, bitecho_tm):
        assert bitecho_tm.is_transducer()
        assert not bitecho_tm.is_generator()
        assert not bitecho_tm.is_recognizer()
