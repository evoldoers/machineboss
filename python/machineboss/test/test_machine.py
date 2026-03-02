"""Tests for Machine construction and JSON I/O."""

import json
import pytest
from machineboss.machine import Machine, MachineState, MachineTransition


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
