"""Tests for HMMER3 file parser."""

import pytest
from machineboss.hmmer import HmmerModel


class TestHmmerParser:
    def test_read_fn3(self, hmmer_file):
        with open(hmmer_file) as f:
            model = HmmerModel.read(f)
        assert len(model.alph) == 20
        assert len(model.nodes) == 86
        assert len(model.ins0_emit) == 20
        assert len(model.null_emit) == 20

    def test_node_emissions_sum_approx_one(self, hmmer_file):
        with open(hmmer_file) as f:
            model = HmmerModel.read(f)
        for i, node in enumerate(model.nodes):
            s = sum(node.match_emit)
            assert s == pytest.approx(1.0, abs=0.01), f"Node {i} match emit sum: {s}"

    def test_transition_probs(self, hmmer_file):
        with open(hmmer_file) as f:
            model = HmmerModel.read(f)
        # First node transitions should sum to ~1
        n = model.nodes[0]
        s = n.m_to_m + n.m_to_i + n.m_to_d
        assert s == pytest.approx(1.0, abs=0.01)


class TestHmmerMachine:
    def test_core_machine_states(self, hmmer_file):
        with open(hmmer_file) as f:
            model = HmmerModel.read(f)
        m = model.machine(local=True)
        assert m.n_states == 5 * 86 + 4  # 434

    def test_core_machine_global_states(self, hmmer_file):
        with open(hmmer_file) as f:
            model = HmmerModel.read(f)
        m = model.machine(local=False)
        assert m.n_states == 434
        assert m.state[0].name == "B"

    def test_plan7_machine_states(self, hmmer_file):
        with open(hmmer_file) as f:
            model = HmmerModel.read(f)
        m = model.plan7_machine()
        assert m.n_states == 434 + 8  # 442
        assert m.state[0].name == "S"
        assert m.state[-1].name == "T"

    def test_plan7_multihit_j_has_transitions(self, hmmer_file):
        with open(hmmer_file) as f:
            model = HmmerModel.read(f)
        m = model.plan7_machine(multihit=True)
        # Find J state
        j_states = [s for s in m.state if s.name == "J"]
        assert len(j_states) == 1
        assert len(j_states[0].trans) == 20  # one emit per amino acid

    def test_plan7_singlehit_j_empty(self, hmmer_file):
        with open(hmmer_file) as f:
            model = HmmerModel.read(f)
        m = model.plan7_machine(multihit=False)
        j_states = [s for s in m.state if s.name == "J"]
        assert len(j_states) == 1
        assert len(j_states[0].trans) == 0

    def test_output_alphabet_amino(self, hmmer_file):
        with open(hmmer_file) as f:
            model = HmmerModel.read(f)
        m = model.machine(local=True)
        alpha = m.output_alphabet()
        assert "A" in alpha
        assert "W" in alpha
        assert len(alpha) == 20
