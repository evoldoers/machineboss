"""Tests for neural TKF92 protein transducer."""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import jax.random as jr

from machineboss.machine import Machine
from machineboss.jax.jax_weight import ParameterizedMachine
from machineboss.jax.dp_neural import neural_log_forward_tok, neural_log_viterbi_tok
from machineboss.neural.tkf92 import make_tkf92_machine, AA_ALPHA, N_AA
from machineboss.neural.stockholm import parse_stockholm, StockholmMSA


class TestTKF92Machine:
    """Test machine structure."""

    def test_state_count(self):
        m = make_tkf92_machine()
        assert m.n_states == 5

    def test_state_names(self):
        m = make_tkf92_machine()
        names = [s.name for s in m.state]
        assert names == ["begin", "match", "insert", "delete", "end"]

    def test_alphabets(self):
        m = make_tkf92_machine()
        assert m.input_alphabet() == AA_ALPHA
        assert m.output_alphabet() == AA_ALPHA
        assert len(AA_ALPHA) == 20

    def test_free_params(self):
        m = make_tkf92_machine()
        pm = ParameterizedMachine.from_machine(m)
        expected = {"t", "insRate", "delRate", "r"}
        expected |= {f"pi_{aa}" for aa in AA_ALPHA}
        assert pm.free_params == expected

    def test_n_transitions(self):
        m = make_tkf92_machine()
        # 5-state canonical WFST: 4 source states (begin, match, insert, delete)
        # each with 4 outgoing column types: 20*20 mat + 20 ins + 20 del + 1 fin = 441.
        # End state has no outgoing transitions.
        assert m.n_transitions == 4 * (20 * 20 + 20 + 20 + 1)


class TestTKF92Forward:
    """Test forward/viterbi with constant parameters."""

    @pytest.fixture
    def pm(self):
        return ParameterizedMachine.from_machine(make_tkf92_machine())

    def _const_params(self, t=0.5, insRate=0.01, delRate=0.02, r=0.3):
        # Uniform amino acid frequencies
        params = {
            "t": jnp.array([[t]]),
            "insRate": jnp.array([[insRate]]),
            "delRate": jnp.array([[delRate]]),
            "r": jnp.array([[r]]),
        }
        for aa in AA_ALPHA:
            params[f"pi_{aa}"] = jnp.array([[1.0 / N_AA]])
        return params

    def test_forward_finite(self, pm):
        seq = "ACDE"
        in_tok = jnp.array(pm.tokenize_input(list(seq)), dtype=jnp.int32)
        out_tok = jnp.array(pm.tokenize_output(list(seq)), dtype=jnp.int32)
        ll = neural_log_forward_tok(pm, in_tok, out_tok, self._const_params())
        assert jnp.isfinite(ll)
        assert float(ll) < 0

    def test_identical_higher_ll(self, pm):
        in_tok = jnp.array(pm.tokenize_input(list("ACDE")), dtype=jnp.int32)
        out_same = jnp.array(pm.tokenize_output(list("ACDE")), dtype=jnp.int32)
        out_diff = jnp.array(pm.tokenize_output(list("FGHI")), dtype=jnp.int32)
        params = self._const_params(t=0.1)
        ll_same = neural_log_forward_tok(pm, in_tok, out_same, params)
        ll_diff = neural_log_forward_tok(pm, in_tok, out_diff, params)
        assert float(ll_same) > float(ll_diff)

    def test_viterbi_le_forward(self, pm):
        in_tok = jnp.array(pm.tokenize_input(list("AC")), dtype=jnp.int32)
        out_tok = jnp.array(pm.tokenize_output(list("AC")), dtype=jnp.int32)
        params = self._const_params()
        ll_fwd = neural_log_forward_tok(pm, in_tok, out_tok, params)
        ll_vit = neural_log_viterbi_tok(pm, in_tok, out_tok, params)
        assert float(ll_vit) <= float(ll_fwd) + 1e-5

    def test_r_zero_reduces_transitions(self, pm):
        """With r=0, no fragment self-loops, should still be finite."""
        in_tok = jnp.array(pm.tokenize_input(list("AC")), dtype=jnp.int32)
        out_tok = jnp.array(pm.tokenize_output(list("AC")), dtype=jnp.int32)
        params = self._const_params(r=0.0)
        ll = neural_log_forward_tok(pm, in_tok, out_tok, params)
        assert jnp.isfinite(ll)


class TestStockholm:
    """Test Stockholm format parser."""

    SIMPLE_STO = """\
# STOCKHOLM 1.0
#=GF ID TestFamily
seq1 ACDE-FG
seq2 A-DEFGH
seq3 ACDKLMN
//
"""

    def test_parse_basic(self):
        msa = parse_stockholm(self.SIMPLE_STO)
        assert msa.id == "TestFamily"
        assert msa.n_seqs == 3
        assert msa.alignment_length == 7
        assert msa.names == ["seq1", "seq2", "seq3"]

    def test_pick_pair(self):
        msa = parse_stockholm(self.SIMPLE_STO)
        s1, s2 = msa.pick_pair(0, 1)
        # Double-gap columns removed, single gaps kept in remaining columns
        assert len(s1) == len(s2)
        # No column should have both sequences gapped
        for c1, c2 in zip(s1, s2):
            assert not (c1 in "-." and c2 in "-.")

    def test_ungapped_pair(self):
        msa = parse_stockholm(self.SIMPLE_STO)
        s1, s2 = msa.ungapped_pair(0, 1)
        assert "-" not in s1
        assert "-" not in s2

    def test_onehot_shape(self):
        msa = parse_stockholm(self.SIMPLE_STO)
        oh = msa.to_onehot()
        assert oh.shape == (3, 7, 21)

    def test_onehot_gap(self):
        msa = parse_stockholm(self.SIMPLE_STO)
        oh = msa.to_onehot()
        # seq1 position 4 is '-' -> index 20
        assert float(oh[0, 4, 20]) == 1.0

    INTERLEAVED_STO = """\
# STOCKHOLM 1.0
seq1 ACDE
seq2 FGHI
#
seq1 KLMN
seq2 PQRS
//
"""

    def test_interleaved(self):
        msa = parse_stockholm(self.INTERLEAVED_STO)
        assert msa.aligned_seqs[0] == "ACDEKLMN"
        assert msa.aligned_seqs[1] == "FGHIPQRS"


class TestMSATransformer:
    """Test MSA transformer (requires flax)."""

    def test_output_shape(self):
        nn = pytest.importorskip("flax.linen")
        from machineboss.neural.msa_transformer import MSATransformer

        model = MSATransformer(d_model=32, n_heads=4, n_layers=1)
        x = jnp.ones((3, 10, 21))  # 3 seqs, length 10
        params = model.init(jax.random.PRNGKey(0), x)
        out = model.apply(params, x)
        assert out.shape == (10, 32)  # (L, d_model)


class TestTKF92Heads:
    """Test parameter heads (requires flax)."""

    def test_output_shapes(self):
        nn = pytest.importorskip("flax.linen")
        from machineboss.neural.tkf92 import TKF92Heads

        heads = TKF92Heads()
        x = jnp.ones((10, 64))  # (L, d_model)
        params = heads.init(jax.random.PRNGKey(0), x)
        t, insRate, delRate, r, pi = heads.apply(params, x)
        assert t.shape == (10,)
        assert insRate.shape == (10,)
        assert delRate.shape == (10,)
        assert r.shape == (10,)
        assert pi.shape == (10, 20)

    def test_output_constraints(self):
        nn = pytest.importorskip("flax.linen")
        from machineboss.neural.tkf92 import TKF92Heads

        heads = TKF92Heads()
        x = jnp.ones((10, 64))
        params = heads.init(jax.random.PRNGKey(0), x)
        t, insRate, delRate, r, pi = heads.apply(params, x)

        assert jnp.all(t > 0.01)
        assert jnp.all(insRate > 0)
        assert jnp.all(delRate > insRate)  # mu > lambda
        assert jnp.all(r >= 0) and jnp.all(r <= 1)
        assert jnp.allclose(jnp.sum(pi, axis=-1), 1.0, atol=1e-5)


class TestTKF92EndToEnd:
    """End-to-end gradient test (requires flax)."""

    def test_gradient_flows(self):
        nn = pytest.importorskip("flax.linen")
        from machineboss.neural.msa_transformer import MSATransformer
        from machineboss.neural.tkf92 import TKF92Heads, heads_to_dp_params
        from machineboss.jax.dp_neural import neural_log_forward_tok

        machine = make_tkf92_machine()
        pm = ParameterizedMachine.from_machine(machine)

        d_model = 32
        transformer = MSATransformer(d_model=d_model, n_heads=4, n_layers=1)
        heads = TKF92Heads()

        # Tiny MSA: 2 seqs, length 3
        msa = jnp.ones((2, 3, 21)) / 21.0
        rng = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(rng)
        t_params = transformer.init(k1, msa)
        emb = transformer.apply(t_params, msa)
        h_params = heads.init(k2, emb)

        in_tok = jnp.array(pm.tokenize_input(list("AC")), dtype=jnp.int32)
        out_tok = jnp.array(pm.tokenize_output(list("AC")), dtype=jnp.int32)

        def loss_fn(all_params):
            emb = transformer.apply(all_params["t"], msa)
            t, insRate, delRate, r, pi = heads.apply(all_params["h"], emb)
            dp_params = heads_to_dp_params(t, insRate, delRate, r, pi)
            ll = neural_log_forward_tok(pm, in_tok, out_tok, dp_params)
            return -ll

        all_params = {"t": t_params, "h": h_params}
        grads = jax.grad(loss_fn)(all_params)

        # Check gradients are non-trivial
        flat_grads = jax.tree.leaves(grads)
        total_norm = sum(float(jnp.sum(g ** 2)) for g in flat_grads)
        assert total_norm > 0

    def test_training_loss_decreases(self):
        nn = pytest.importorskip("flax.linen")
        optax = pytest.importorskip("optax")
        from machineboss.neural.msa_transformer import MSATransformer
        from machineboss.neural.tkf92 import TKF92Heads, heads_to_dp_params
        from machineboss.jax.dp_neural import neural_log_forward_tok

        machine = make_tkf92_machine()
        pm = ParameterizedMachine.from_machine(machine)

        d_model = 32
        transformer = MSATransformer(d_model=d_model, n_heads=4, n_layers=1)
        heads = TKF92Heads()

        # Tiny MSA: 2 seqs, length 3
        msa = jnp.ones((2, 3, 21)) / 21.0
        rng = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(rng)
        t_params = transformer.init(k1, msa)
        emb = transformer.apply(t_params, msa)
        h_params = heads.init(k2, emb)

        in_tok = jnp.array(pm.tokenize_input(list("AC")), dtype=jnp.int32)
        out_tok = jnp.array(pm.tokenize_output(list("AC")), dtype=jnp.int32)

        def loss_fn(all_params):
            emb = transformer.apply(all_params["t"], msa)
            t, insRate, delRate, r, pi = heads.apply(all_params["h"], emb)
            dp_params = heads_to_dp_params(t, insRate, delRate, r, pi)
            ll = neural_log_forward_tok(pm, in_tok, out_tok, dp_params)
            return -ll

        all_params = {"t": t_params, "h": h_params}
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(all_params)

        losses = []
        for _ in range(8):
            loss, grads = jax.value_and_grad(loss_fn)(all_params)
            losses.append(float(loss))
            updates, opt_state = optimizer.update(grads, opt_state)
            all_params = optax.apply_updates(all_params, updates)

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
