"""Pytest fixtures for TransMachine tests."""

import pytest
from pathlib import Path

import jax.numpy as jnp

from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine
from machineboss.jax.trans.machine import TransMachine


@pytest.fixture
def repo_root():
    return Path(__file__).parent.parent.parent.parent.parent.parent


@pytest.fixture
def boss_path(repo_root):
    p = repo_root / "bin" / "boss"
    if not p.is_file():
        pytest.skip("bin/boss not found")
    return str(p)


@pytest.fixture
def test_data_dir(repo_root):
    return repo_root / "t"


@pytest.fixture
def bitecho_machine(repo_root):
    """Identity transducer for {0, 1}."""
    path = repo_root / "t" / "machine" / "bitecho.json"
    return Machine.from_file(str(path))


@pytest.fixture
def bitnoise_machine(repo_root):
    """Noisy channel transducer for {0, 1}."""
    path = repo_root / "t" / "machine" / "bitnoise.json"
    return Machine.from_file(str(path))


@pytest.fixture
def bitecho_tm(bitecho_machine):
    return TransMachine.from_machine(bitecho_machine)


@pytest.fixture
def bitnoise_tm(repo_root):
    import json
    path = repo_root / "t" / "machine" / "bitnoise.json"
    params_path = repo_root / "t" / "io" / "params.json"
    m = Machine.from_file(str(path))
    with open(params_path) as f:
        params = json.load(f)
    return TransMachine.from_machine(m, params)


@pytest.fixture
def bitecho_jm(bitecho_machine):
    em = EvaluatedMachine.from_machine(bitecho_machine)
    return JAXMachine.from_evaluated(em)


@pytest.fixture
def bitnoise_jm(repo_root):
    import json
    path = repo_root / "t" / "machine" / "bitnoise.json"
    params_path = repo_root / "t" / "io" / "params.json"
    m = Machine.from_file(str(path))
    with open(params_path) as f:
        params = json.load(f)
    em = EvaluatedMachine.from_machine(m, params)
    return JAXMachine.from_evaluated(em)


@pytest.fixture
def bitecho_em(bitecho_machine):
    return EvaluatedMachine.from_machine(bitecho_machine)
