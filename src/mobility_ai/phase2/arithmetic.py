"""Bounded arithmetic expressions, without Python execution or object access."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

_OPERATORS: dict[type[ast.operator], Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_FUNCTIONS: dict[str, Callable] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def evaluate_arithmetic(expression: str) -> int | float:
    if len(expression) > 512:
        raise ValueError("Expression exceeds 512 characters")
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression is too complex")

    def evaluate(node: ast.AST) -> int | float:
        result: int | float
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            result = node.value
        elif isinstance(node, ast.Name) and node.id in _CONSTANTS:
            result = _CONSTANTS[node.id]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            result = value if isinstance(node.op, ast.UAdd) else -value
        elif isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 100:
                    raise ValueError("Exponent exceeds 100")
                result = math.pow(left, right)
            elif type(node.op) in _OPERATORS:
                result = _OPERATORS[type(node.op)](left, right)
            else:
                raise ValueError("Operator is not allowed")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
            and len(node.args) <= 2
        ):
            result = _FUNCTIONS[node.func.id](*(evaluate(arg) for arg in node.args))
        else:
            raise ValueError("Only numbers, arithmetic, and approved math functions are allowed")
        if abs(result) > 1e100 or not math.isfinite(result):
            raise ValueError("Result is outside the supported numeric range")
        return result

    return evaluate(tree.body)
