from __future__ import annotations

from typing import Any


def eval_unary(op: str, value: Any) -> Any:
    if op == "-":
        return -value
    if op == "~":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("~ needs an int")
        return ~value
    if op in {"!", "not"}:
        return not bool(value)
    raise ValueError(f"unsupported unary operator: {op}")


def eval_binary(op: str, left: Any, right: Any) -> Any:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return left / right
    if op == "%":
        return left % right
    if op == "//":
        return left // right
    if op == "**":
        return left**right
    if op in {"&", "|", "^", "<<", ">>"}:
        if isinstance(left, bool) or isinstance(right, bool) or not isinstance(left, int) or not isinstance(right, int):
            raise TypeError("bitwise operators need ints")
        if op == "&":
            return left & right
        if op == "|":
            return left | right
        if op == "^":
            return left ^ right
        if op == "<<":
            if right > 64:
                raise OverflowError("shift count too large")
            return left << right
        return left >> right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op in {"and", "&&"}:
        return bool(left) and bool(right)
    if op in {"or", "||"}:
        return bool(left) or bool(right)
    raise ValueError(f"unsupported binary operator: {op}")
