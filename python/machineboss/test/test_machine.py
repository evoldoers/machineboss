"""Tests for Machine construction and JSON I/O."""

import json
import pytest
from machineboss.machine import Machine, MachineState, MachineTransition
from machineboss.eval import EvaluatedMachine


class TestMachineJSON:
    def test_roundtrip_simple(self):
        m = Machine(state=[
            MachineState(name="start", trans=[
                MachineTransition(dest=1, output="A", weight=0.5),
                MachineTransition(dest=1, output="B", weight=0.5),
            ]),
            MachineState(name="end"),
        ])
        j = m.to_json()
        m2 = Machine.from_json(j)
        assert m2.n_states == 2
        assert m2.state[0].name == "start"
        assert len(m2.state[0].trans) == 2
        assert m2.state[1].name == "end"
        assert len(m2.state[1].trans) == 0

    def test_from_json_string(self):
        s = '{"state":[{"id":"s","trans":[{"to":1}]},{"id":"e","trans":[]}]}'
        m = Machine.from_json(s)
        assert m.n_states == 2
        assert m.state[0].name == "s"

    def test_output_alphabet(self):
        m = Machine(state=[
            MachineState(trans=[
                MachineTransition(dest=0, output="C"),
                MachineTransition(dest=0, output="A"),
                MachineTransition(dest=0, output="B"),
                MachineTransition(dest=1),
            ]),
            MachineState(),
        ])
        assert m.output_alphabet() == ["A", "B", "C"]

    def test_input_alphabet(self):
        m = Machine(state=[
            MachineState(trans=[
                MachineTransition(dest=0, input="1"),
                MachineTransition(dest=0, input="0"),
                MachineTransition(dest=1),
            ]),
            MachineState(),
        ])
        assert m.input_alphabet() == ["0", "1"]


class TestMachineFromFile:
    def test_load_bitecho(self, repo_root):
        path = repo_root / "t" / "machine" / "bitecho.json"
        if not path.exists():
            pytest.skip("test data not found")
        m = Machine.from_file(str(path))
        assert m.n_states > 0
        assert m.input_alphabet() == ["0", "1"]
        assert m.output_alphabet() == ["0", "1"]

    def test_load_bitnoise(self, repo_root):
        path = repo_root / "t" / "machine" / "bitnoise.json"
        if not path.exists():
            pytest.skip("test data not found")
        m = Machine.from_file(str(path))
        assert m.n_states > 0


class TestTransition:
    def test_silent(self):
        t = MachineTransition(dest=1)
        assert t.is_silent

    def test_not_silent(self):
        t = MachineTransition(dest=1, output="A")
        assert not t.is_silent

    def test_to_json_minimal(self):
        t = MachineTransition(dest=1)
        assert t.to_json() == {"to": 1}

    def test_to_json_full(self):
        t = MachineTransition(dest=2, input="A", output="B", weight=0.5)
        j = t.to_json()
        assert j["to"] == 2
        assert j["in"] == "A"
        assert j["out"] == "B"
        assert j["weight"] == 0.5


# Machines with parameters and their evaluation params
PARAM_MACHINES = {
    "unitindel": {"ins": 0.1, "no_ins": 0.9, "del": 0.1, "no_del": 0.9},
    "bitnoise": {"p": 0.9, "q": 0.1},
    "bsc": {"e": 0.1},
    "bitstutter": {},  # has numeric weights, no params needed
}
# Machines with no parameters (all weights numeric or 1)
SIMPLE_MACHINES = ["bitecho", "bitstutter"]


class TestIdempotentUnevaluated:
    """Unevaluated roundtrip: from_json(to_json(from_file(path))) twice."""

    @pytest.mark.parametrize("name", SIMPLE_MACHINES)
    def test_unevaluated_roundtrip(self, repo_root, name):
        path = repo_root / "t" / "machine" / f"{name}.json"
        if not path.exists():
            pytest.skip("test data not found")
        m1 = Machine.from_file(str(path))
        j1 = m1.to_json()
        m2 = Machine.from_json(j1)
        j2 = m2.to_json()
        assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True)


class TestIdempotentEvaluated:
    """Evaluated roundtrip: Machine -> EvaluatedMachine -> to_machine() -> to_json() twice."""

    @pytest.mark.parametrize("name,params", PARAM_MACHINES.items())
    def test_evaluated_roundtrip(self, repo_root, name, params):
        path = repo_root / "t" / "machine" / f"{name}.json"
        if not path.exists():
            pytest.skip("test data not found")
        m = Machine.from_file(str(path))
        em = EvaluatedMachine.from_machine(m, params)
        m1 = em.to_machine()
        j1 = m1.to_json()
        # Second round
        em2 = EvaluatedMachine.from_machine(m1)
        m2 = em2.to_machine()
        j2 = m2.to_json()
        assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True)

    def test_evaluated_preserves_structure(self, repo_root):
        path = repo_root / "t" / "machine" / "bitecho.json"
        if not path.exists():
            pytest.skip("test data not found")
        m = Machine.from_file(str(path))
        em = EvaluatedMachine.from_machine(m)
        m2 = em.to_machine()
        assert m2.n_states == m.n_states
        assert m2.input_alphabet() == m.input_alphabet()
        assert m2.output_alphabet() == m.output_alphabet()


class TestIdempotentJAX:
    """JAXMachine roundtrip: Machine -> Evaluated -> JAXMachine -> to_machine() -> to_json() twice."""

    @pytest.mark.parametrize("name,params", PARAM_MACHINES.items())
    def test_jax_roundtrip(self, repo_root, name, params):
        pytest.importorskip("jax")
        from machineboss.jax.types import JAXMachine

        path = repo_root / "t" / "machine" / f"{name}.json"
        if not path.exists():
            pytest.skip("test data not found")
        m = Machine.from_file(str(path))
        em = EvaluatedMachine.from_machine(m, params)
        jm = JAXMachine.from_evaluated(em)
        m1 = jm.to_machine()
        j1 = m1.to_json()
        # Second round
        em2 = EvaluatedMachine.from_machine(m1)
        jm2 = JAXMachine.from_evaluated(em2)
        m2 = jm2.to_machine()
        j2 = m2.to_json()
        assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True)

    def test_jax_preserves_tokens(self, repo_root):
        pytest.importorskip("jax")
        from machineboss.jax.types import JAXMachine

        path = repo_root / "t" / "machine" / "bitecho.json"
        if not path.exists():
            pytest.skip("test data not found")
        m = Machine.from_file(str(path))
        em = EvaluatedMachine.from_machine(m)
        jm = JAXMachine.from_evaluated(em)
        assert jm.input_token_list == ["", "0", "1"]
        assert jm.output_token_list == ["", "0", "1"]
