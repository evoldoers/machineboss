"""Tests for weight expression parse/eval/deriv."""

import math
import pytest
from machineboss.weight import (
    evaluate, differentiate, params,
    multiply, add, divide, subtract, log_of, exp_of,
    ZERO, ONE, is_zero, is_one,
)


class TestEvaluate:
    def test_int_constant(self):
        assert evaluate(42) == 42.0

    def test_float_constant(self):
        assert evaluate(3.14) == pytest.approx(3.14)

    def test_zero(self):
        assert evaluate(0) == 0.0

    def test_one(self):
        assert evaluate(1) == 1.0

    def test_param(self):
        assert evaluate("p", {"p": 0.3}) == pytest.approx(0.3)

    def test_param_missing(self):
        with pytest.raises(KeyError):
            evaluate("q", {"p": 0.3})

    def test_multiply(self):
        assert evaluate({"*": [2, 3]}) == pytest.approx(6.0)

    def test_add(self):
        assert evaluate({"+": [2, 3]}) == pytest.approx(5.0)

    def test_subtract(self):
        assert evaluate({"-": [5, 3]}) == pytest.approx(2.0)

    def test_divide(self):
        assert evaluate({"/": [6, 3]}) == pytest.approx(2.0)

    def test_log(self):
        assert evaluate({"log": 1}) == pytest.approx(0.0)
        assert evaluate({"log": math.e}) == pytest.approx(1.0)

    def test_exp(self):
        assert evaluate({"exp": 0}) == pytest.approx(1.0)
        assert evaluate({"exp": 1}) == pytest.approx(math.e)

    def test_pow(self):
        assert evaluate({"pow": [2, 10]}) == pytest.approx(1024.0)

    def test_not(self):
        assert evaluate({"not": 0.3}) == pytest.approx(0.7)

    def test_nested(self):
        # (p + q) * 2
        expr = {"*": [{"+": ["p", "q"]}, 2]}
        assert evaluate(expr, {"p": 3, "q": 4}) == pytest.approx(14.0)


class TestDifferentiate:
    def test_constant(self):
        assert is_zero(differentiate(5, "p"))

    def test_param(self):
        assert is_one(differentiate("p", "p"))
        assert is_zero(differentiate("q", "p"))

    def test_sum(self):
        # d(p + 2)/dp = 1
        d = differentiate({"+": ["p", 2]}, "p")
        assert evaluate(d, {"p": 1}) == pytest.approx(1.0)

    def test_product(self):
        # d(p * q)/dp = q
        d = differentiate({"*": ["p", "q"]}, "p")
        assert evaluate(d, {"p": 2, "q": 3}) == pytest.approx(3.0)

    def test_log(self):
        # d(log(p))/dp = 1/p
        d = differentiate({"log": "p"}, "p")
        assert evaluate(d, {"p": 2.0}) == pytest.approx(0.5)

    def test_exp(self):
        # d(exp(p))/dp = exp(p)
        d = differentiate({"exp": "p"}, "p")
        assert evaluate(d, {"p": 1.0}) == pytest.approx(math.e)


class TestParams:
    def test_constant(self):
        assert params(42) == set()

    def test_param(self):
        assert params("p") == {"p"}

    def test_nested(self):
        assert params({"*": ["p", {"+": ["q", 1]}]}) == {"p", "q"}


class TestConstructors:
    def test_multiply_zero(self):
        assert is_zero(multiply(ZERO, "p"))
        assert is_zero(multiply("p", ZERO))

    def test_multiply_one(self):
        assert multiply(ONE, "p") == "p"
        assert multiply("p", ONE) == "p"

    def test_add_zero(self):
        assert add(ZERO, "p") == "p"
        assert add("p", ZERO) == "p"
