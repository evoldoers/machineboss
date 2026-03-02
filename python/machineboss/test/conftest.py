"""Pytest fixtures for machineboss tests."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def repo_root():
    """Path to the machineboss repository root."""
    return Path(__file__).parent.parent.parent.parent


@pytest.fixture
def boss_path(repo_root):
    """Path to bin/boss executable."""
    p = repo_root / "bin" / "boss"
    if not p.is_file():
        pytest.skip("bin/boss not found; build with 'make' first")
    return str(p)


@pytest.fixture
def test_data_dir(repo_root):
    """Path to t/ test data directory."""
    return repo_root / "t"


@pytest.fixture
def hmmer_file(test_data_dir):
    """Path to fn3.hmm test file."""
    p = test_data_dir / "hmmer" / "fn3.hmm"
    if not p.is_file():
        pytest.skip("t/hmmer/fn3.hmm not found")
    return str(p)
