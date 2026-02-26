from __future__ import annotations

from typing import Any


def eval_unary(op: str, value: Any) -> Any:
    if op == "-":
        return -value
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
