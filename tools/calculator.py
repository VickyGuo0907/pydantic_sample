"""Safe mathematical expression evaluator using AST parsing."""

from __future__ import annotations

import ast
import operator
from typing import Callable

# Allowed binary operators mapped to their stdlib implementations.
_BINARY_OPS: dict[type, Callable[[int | float, int | float], int | float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators mapped to their stdlib implementations.
_UNARY_OPS: dict[type, Callable[[int | float], int | float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> int | float:
    """Recursively evaluate an AST node, allowing only safe math operations.

    Args:
        node: An AST node from a parsed expression.

    Returns:
        The numeric result of evaluating the node.

    Raises:
        ValueError: If the node contains unsupported operations or types.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp):
        op_func = _BINARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return op_func(left, right)

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_eval_node(node.operand))

    raise ValueError(
        f"Unsupported expression element: {type(node).__name__}. "
        "Only numeric literals and basic arithmetic (+, -, *, /, //, %, **) are allowed."
    )


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Uses AST parsing to allow only numeric literals and basic arithmetic.
    Never uses eval(). Rejects any function calls, imports, or variable references.

    Args:
        expression: A math expression string like "2 + 3 * 4".

    Returns:
        The result as a string, or an error message prefixed with "Error:" if
        evaluation fails.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree)
        # Drop trailing .0 for whole-number float results (e.g. 10 / 2 → "5")
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except (ValueError, SyntaxError, TypeError) as exc:
        return f"Error: {exc}"
