from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from alpha.ast_nodes import BinaryOpNode, ExprNode, FunctionCallNode, IdentifierNode, NumberNode, UnaryOpNode
from features.operators import safe_divide
from features.registry import OperatorRegistry


@dataclass(slots=True)
class EvaluationContext:
    numeric_fields: Mapping[str, pd.DataFrame]
    registry: OperatorRegistry
    group_fields: Mapping[str, pd.DataFrame] = field(default_factory=dict)


class ExpressionEvaluator:
    def __init__(self, context: EvaluationContext) -> None:
        self.context = context

    def evaluate(self, node: ExprNode):
        if isinstance(node, NumberNode):
            return node.value
        if isinstance(node, IdentifierNode):
            if node.name in self.context.numeric_fields:
                return self.context.numeric_fields[node.name]
            if node.name in self.context.group_fields:
                return self.context.group_fields[node.name]
            raise KeyError(f"Field '{node.name}' is unavailable in the evaluation context.")
        if isinstance(node, UnaryOpNode):
            operand = self.evaluate(node.operand)
            return +operand if node.operator == "+" else -operand
        if isinstance(node, BinaryOpNode):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            if node.operator == "+":
                return left + right
            if node.operator == "-":
                return left - right
            if node.operator == "*":
                return left * right
            if node.operator == "/":
                return safe_divide(left, right)
            raise ValueError(f"Unsupported binary operator '{node.operator}'.")
        if isinstance(node, FunctionCallNode):
            spec = self.context.registry.get(node.name)
            args = [self.evaluate(argument) for argument in node.args]
            return spec.function(*args)
        raise TypeError(f"Unsupported expression node type: {type(node)!r}")


def evaluate_expression(
    node: ExprNode,
    fields: Mapping[str, pd.DataFrame],
    registry: OperatorRegistry,
    group_fields: Mapping[str, pd.DataFrame] | None = None,
):
    return ExpressionEvaluator(
        context=EvaluationContext(numeric_fields=fields, registry=registry, group_fields=group_fields or {}),
    ).evaluate(node)
