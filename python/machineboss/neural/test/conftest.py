"""Pytest fixtures for neural transducer tests."""

import pytest
from pathlib import Path


@pytest.fixture
def repo_root():
    return Path(__file__).parent.parent.parent.parent.parent


@pytest.fixture
def rng_key():
    jax = pytest.importorskip("jax")
    return jax.random.PRNGKey(42)
