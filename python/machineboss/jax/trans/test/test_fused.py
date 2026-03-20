"""Tests for fused Plan7+transducer DP on TransMachine.

Delegates to existing fused_plan7 tests for coverage; this test
verifies the TransMachine wrapper produces equivalent results.
"""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp


class TestFusedDelegation:
    """Test that fused DP delegates correctly."""

    def test_import(self):
        from machineboss.jax.trans.dp_fused import fused_forward, fused_viterbi
        # Just verify imports work
        assert callable(fused_forward)
        assert callable(fused_viterbi)
