"""Pytest fixtures for JAX DP algorithm tests."""

import pytest
from pathlib import Path


@pytest.fixture
def repo_root():
    return Path(__file__).parent.parent.parent.parent.parent


@pytest.fixture
def boss_path(repo_root):
    p = repo_root / "bin" / "boss"
    if not p.is_file():
        pytest.skip("bin/boss not found")
    return str(p)


@pytest.fixture
def test_data_dir(repo_root):
    return repo_root / "t"
