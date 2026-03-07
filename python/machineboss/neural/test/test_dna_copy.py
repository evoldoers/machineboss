"""Tests for neural DNA copy transducer."""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import jax.random as jr

from machineboss.machine import Machine
from machineboss.jax.jax_weight import ParameterizedMachine
from machineboss.jax.dp_neural import neural_log_forward_tok, neural_log_viterbi_tok
from machineboss.neural.dna_copy import (
    make_dna_copy_machine, onehot_dna, tokenize_dna,
)
from machineboss.neural.simulator import (
    simulate_dna_pair, simulate_batch, _find_homopolymer_mask,
)


class TestDNACopyMachine:
    """Test machine structure."""

    def test_state_count(self):
        m = make_dna_copy_machine()
        assert m.n_states == 6

    def test_alphabets(self):
        m = make_dna_copy_machine()
        assert m.input_alphabet() == ["A", "C", "G", "T"]
        assert m.output_alphabet() == ["A", "C", "G", "T"]

    def test_free_params(self):
        m = make_dna_copy_machine()
        pm = ParameterizedMachine.from_machine(m)
        assert pm.free_params == {"t", "pIns", "pDel"}

    def test_state_names(self):
        m = make_dna_copy_machine()
        names = [s.name for s in m.state]
        assert names == ["begin", "wait", "match", "insert", "delete", "end"]


class TestDNACopyForward:
    """Test forward/viterbi with constant parameters."""

    @pytest.fixture
    def pm(self):
        return ParameterizedMachine.from_machine(make_dna_copy_machine())

    def _const_params(self, t=0.1, pIns=0.05, pDel=0.05):
        return {
            "t": jnp.array([[t]]),
            "pIns": jnp.array([[pIns]]),
            "pDel": jnp.array([[pDel]]),
        }

    def test_forward_finite(self, pm):
        in_tok = jnp.array(pm.tokenize_input(list("ACGT")), dtype=jnp.int32)
        out_tok = jnp.array(pm.tokenize_output(list("ACGT")), dtype=jnp.int32)
        ll = neural_log_forward_tok(pm, in_tok, out_tok, self._const_params())
        assert jnp.isfinite(ll)
        assert float(ll) < 0

    def test_identical_higher_ll(self, pm):
        in_tok = jnp.array(pm.tokenize_input(list("ACGTACGT")), dtype=jnp.int32)
        out_same = jnp.array(pm.tokenize_output(list("ACGTACGT")), dtype=jnp.int32)
        out_diff = jnp.array(pm.tokenize_output(list("TGCATGCA")), dtype=jnp.int32)
        params = self._const_params(t=0.1)
        ll_same = neural_log_forward_tok(pm, in_tok, out_same, params)
        ll_diff = neural_log_forward_tok(pm, in_tok, out_diff, params)
        assert float(ll_same) > float(ll_diff)

    def test_viterbi_le_forward(self, pm):
        in_tok = jnp.array(pm.tokenize_input(list("ACGT")), dtype=jnp.int32)
        out_tok = jnp.array(pm.tokenize_output(list("ACGT")), dtype=jnp.int32)
        params = self._const_params()
        ll_fwd = neural_log_forward_tok(pm, in_tok, out_tok, params)
        ll_vit = neural_log_viterbi_tok(pm, in_tok, out_tok, params)
        assert float(ll_vit) <= float(ll_fwd) + 1e-5


class TestSimulator:
    """Test DNA pair simulator."""

    def test_produces_valid_dna(self):
        anc, desc = simulate_dna_pair(jr.PRNGKey(0), length=50)
        assert len(anc) == 50
        assert all(b in "ACGT" for b in anc)
        assert all(b in "ACGT" for b in desc)
        assert len(desc) > 0

    def test_batch(self):
        pairs = simulate_batch(jr.PRNGKey(1), n_pairs=5, length=30)
        assert len(pairs) == 5
        for anc, desc in pairs:
            assert len(anc) == 30
            assert len(desc) > 0

    def test_homopolymer_mask(self):
        # AAACGT -> first 3 positions are homopolymer
        seq = jnp.array([0, 0, 0, 1, 2, 3])  # A,A,A,C,G,T
        mask = _find_homopolymer_mask(seq, min_run=3)
        assert bool(mask[0]) and bool(mask[1]) and bool(mask[2])
        assert not bool(mask[3]) and not bool(mask[4]) and not bool(mask[5])

    def test_high_hp_multiplier_increases_divergence(self):
        """Higher hp_multiplier should cause more divergence in hp regions."""
        rng = jr.PRNGKey(42)
        # Long sequence with embedded homopolymer
        pairs_low = simulate_batch(rng, 20, length=50, hp_multiplier=1.0,
                                   base_sub_rate=0.1, base_indel_rate=0.05)
        pairs_high = simulate_batch(rng, 20, length=50, hp_multiplier=5.0,
                                    base_sub_rate=0.1, base_indel_rate=0.05)
        # Higher multiplier should produce more length variation on average
        len_diffs_low = [abs(len(a) - len(d)) for a, d in pairs_low]
        len_diffs_high = [abs(len(a) - len(d)) for a, d in pairs_high]
        # This is statistical, but with 20 pairs it should be robust
        assert sum(len_diffs_high) >= sum(len_diffs_low)


class TestOnehot:
    def test_shape(self):
        oh = onehot_dna("ACGT")
        assert oh.shape == (4, 4)

    def test_values(self):
        oh = onehot_dna("A")
        assert float(oh[0, 0]) == 1.0
        assert float(oh[0, 1]) == 0.0


class TestCNN:
    """Test CNN and gradient flow (requires flax)."""

    @pytest.fixture
    def pm(self):
        return ParameterizedMachine.from_machine(make_dna_copy_machine())

    def test_cnn_output_shapes(self):
        nn = pytest.importorskip("flax.linen")
        from machineboss.neural.dna_copy import DNACopyCNN
        model = DNACopyCNN()
        x = onehot_dna("ACGTACGT")  # (8, 4)
        params = model.init(jax.random.PRNGKey(0), x)
        t, pIns, pDel = model.apply(params, x)
        assert t.shape == (8,)
        assert pIns.shape == (8,)
        assert pDel.shape == (8,)

    def test_cnn_value_ranges(self):
        nn = pytest.importorskip("flax.linen")
        from machineboss.neural.dna_copy import DNACopyCNN
        model = DNACopyCNN()
        x = onehot_dna("ACGTACGTACGT")
        params = model.init(jax.random.PRNGKey(0), x)
        t, pIns, pDel = model.apply(params, x)
        assert jnp.all(t > 0.01)
        assert jnp.all(pIns >= 0) and jnp.all(pIns <= 0.3)
        assert jnp.all(pDel >= 0) and jnp.all(pDel <= 0.3)

    def test_gradient_flows(self, pm):
        nn = pytest.importorskip("flax.linen")
        from machineboss.neural.dna_copy import DNACopyCNN, make_loss_fn

        model = DNACopyCNN()
        x = onehot_dna("ACGT")
        cnn_params = model.init(jax.random.PRNGKey(0), x)
        in_tok = jnp.array(pm.tokenize_input(list("ACGT")), dtype=jnp.int32)
        out_tok = jnp.array(pm.tokenize_output(list("ACGT")), dtype=jnp.int32)

        loss_fn = make_loss_fn(pm)
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(cnn_params, model, x, in_tok, out_tok)

        # Check that gradients are non-trivial (not all zero)
        flat_grads = jax.tree.leaves(grads)
        total_norm = sum(float(jnp.sum(g ** 2)) for g in flat_grads)
        assert total_norm > 0

    def test_training_loss_decreases(self, pm):
        nn = pytest.importorskip("flax.linen")
        optax = pytest.importorskip("optax")
        from machineboss.neural.dna_copy import DNACopyCNN, make_loss_fn

        model = DNACopyCNN(hidden=16)
        anc, desc = "ACGTACGT", "ACGTACGT"
        x = onehot_dna(anc)
        in_tok = jnp.array(pm.tokenize_input(list(anc)), dtype=jnp.int32)
        out_tok = jnp.array(pm.tokenize_output(list(desc)), dtype=jnp.int32)

        cnn_params = model.init(jax.random.PRNGKey(0), x)
        loss_fn = make_loss_fn(pm)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(cnn_params)

        losses = []
        for _ in range(10):
            loss, grads = jax.value_and_grad(loss_fn)(
                cnn_params, model, x, in_tok, out_tok)
            losses.append(float(loss))
            updates, opt_state = optimizer.update(grads, opt_state)
            cnn_params = optax.apply_updates(cnn_params, updates)

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
