"""Integration tests for Boss CLI wrapper (requires bin/boss)."""

import json
import pytest
from machineboss.boss import Boss
from machineboss.machine import Machine


@pytest.fixture
def boss(boss_path):
    return Boss(executable=boss_path)


class TestBossWrapper:
    def test_help(self, boss):
        # --help returns exit code 0 with usage info
        out = boss.run("--help")
        assert "options" in out.lower()

    def test_generate_chars(self, boss):
        result = boss.run_json("--generate-chars", "ACG")
        assert "state" in result

    def test_load_machine(self, boss, repo_root):
        path = str(repo_root / "t" / "machine" / "bitecho.json")
        m = boss.load_machine(path)
        assert m.n_states > 0

    def test_hmmer_global(self, boss, hmmer_file):
        m = boss.load_machine("--hmmer-global", hmmer_file)
        assert m.n_states == 434

    def test_hmmer_plan7(self, boss, hmmer_file):
        m = boss.load_machine("--hmmer-plan7", hmmer_file)
        assert m.n_states == 442
        assert m.state[0].name == "S"
        assert m.state[-1].name == "T"

    def test_hmmer_multihit(self, boss, hmmer_file):
        m = boss.load_machine("--hmmer-multihit", hmmer_file)
        assert m.n_states == 442
