"""Weight expression parser, evaluator, and symbolic differentiator.

Mirrors the C++ WeightAlgebra class. Weight expressions are JSON values:
- Numbers (int/float): constants
- Strings: parameter names (e.g. "p", "rate")
- Dicts with operator keys: {"*": [a, b]}, {"+": [a, b]}, {"/": [a, b]},
  {"-": [a, b]}, {"log": x}, {"exp": x}, {"pow": [base, exp]}, {"not": x}
"""

from __future__ import annotations

import math
from typing import Any

# Type alias
WeightExpr = Any  # int | float | str | dict


# Constants
ZERO = 0
ONE = 1


def is_zero(w: WeightExpr) -> bool:
    return w == 0 or w == 0.0


def is_one(w: WeightExpr) -> bool:
    return w == 1 or w == 1.0


def multiply(a: WeightExpr, b: WeightExpr) -> WeightExpr:
    if is_zero(a) or is_zero(b):
        return ZERO
    if is_one(a):
        return b
    if is_one(b):
        return a
    return {"*": [a, b]}


def add(a: WeightExpr, b: WeightExpr) -> WeightExpr:
    if is_zero(a):
        return b
    if is_zero(b):
        return a
    return {"+": [a, b]}


def subtract(a: WeightExpr, b: WeightExpr) -> WeightExpr:
    if is_zero(b):
        return a
    return {"-": [a, b]}


def divide(a: WeightExpr, b: WeightExpr) -> WeightExpr:
    if is_zero(a):
        return ZERO
    if is_one(b):
        return a
    return {"/": [a, b]}


def power(base: WeightExpr, exp: WeightExpr) -> WeightExpr:
    if is_zero(exp):
        return ONE
    if is_one(exp):
        return base
    return {"pow": [base, exp]}


def log_of(x: WeightExpr) -> WeightExpr:
    return {"log": x}


def exp_of(x: WeightExpr) -> WeightExpr:
    return {"exp": x}


def negate(x: WeightExpr) -> WeightExpr:
    """1 - x"""
    return subtract(ONE, x)


def reciprocal(x: WeightExpr) -> WeightExpr:
    """1 / x"""
    return divide(ONE, x)


def evaluate(w: WeightExpr, params: dict[str, float] | None = None) -> float:
    """Evaluate a weight expression to a float given parameter values."""
    if params is None:
        params = {}

    if isinstance(w, (int, float)):
        return float(w)
    if isinstance(w, str):
        if w in params:
            return params[w]
        raise KeyError(f"Unknown parameter: {w}")
    if isinstance(w, dict):
        if "*" in w:
            args = w["*"]
            return evaluate(args[0], params) * evaluate(args[1], params)
        if "+" in w:
            args = w["+"]
            return evaluate(args[0], params) + evaluate(args[1], params)
        if "-" in w:
            args = w["-"]
            return evaluate(args[0], params) - evaluate(args[1], params)
        if "/" in w:
            args = w["/"]
            return evaluate(args[0], params) / evaluate(args[1], params)
        if "pow" in w:
            args = w["pow"]
            return evaluate(args[0], params) ** evaluate(args[1], params)
        if "log" in w:
            return math.log(evaluate(w["log"], params))
        if "exp" in w:
            return math.exp(evaluate(w["exp"], params))
        if "not" in w:
            return 1.0 - evaluate(w["not"], params)
        raise ValueError(f"Unknown weight expression operator: {list(w.keys())}")
    raise TypeError(f"Unsupported weight expression type: {type(w)}")


def differentiate(w: WeightExpr, param: str) -> WeightExpr:
    """Symbolic differentiation of weight expression with respect to a parameter."""
    if isinstance(w, (int, float)):
        return ZERO
    if isinstance(w, str):
        return ONE if w == param else ZERO
    if isinstance(w, dict):
        if "*" in w:
            a, b = w["*"]
            # Product rule: d(a*b) = da*b + a*db
            da, db = differentiate(a, param), differentiate(b, param)
            return add(multiply(da, b), multiply(a, db))
        if "+" in w:
            a, b = w["+"]
            return add(differentiate(a, param), differentiate(b, param))
        if "-" in w:
            a, b = w["-"]
            return subtract(differentiate(a, param), differentiate(b, param))
        if "/" in w:
            a, b = w["/"]
            # Quotient rule: d(a/b) = (da*b - a*db) / b^2
            da, db = differentiate(a, param), differentiate(b, param)
            return divide(
                subtract(multiply(da, b), multiply(a, db)),
                multiply(b, b)
            )
        if "log" in w:
            x = w["log"]
            # d(log x) = dx / x
            return divide(differentiate(x, param), x)
        if "exp" in w:
            x = w["exp"]
            # d(exp x) = exp(x) * dx
            return multiply(w, differentiate(x, param))
        if "not" in w:
            return subtract(ZERO, differentiate(w["not"], param))
        if "pow" in w:
            base, exp_ = w["pow"]
            # d(a^b) = a^b * (db*log(a) + b*da/a) [general case]
            da = differentiate(base, param)
            db = differentiate(exp_, param)
            return multiply(w, add(
                multiply(db, log_of(base)),
                divide(multiply(exp_, da), base)
            ))
    return ZERO


def params(w: WeightExpr) -> set[str]:
    """Return the set of parameter names in a weight expression."""
    if isinstance(w, (int, float)):
        return set()
    if isinstance(w, str):
        return {w}
    if isinstance(w, dict):
        result = set()
        for v in w.values():
            if isinstance(v, list):
                for item in v:
                    result |= params(item)
            else:
                result |= params(v)
        return result
    return set()
