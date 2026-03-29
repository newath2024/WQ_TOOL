from __future__ import annotations

from dataclasses import dataclass

from alpha.ast_nodes import (
    BinaryOpNode,
    ExprNode,
    FunctionCallNode,
    IdentifierNode,
    NumberNode,
    UnaryOpNode,
    node_complexity,
    node_depth,
    to_expression,
)
from features.registry import IDEMPOTENT_WRAPPERS, OperatorRegistry, WINDOWED_OPERATORS


@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]
    output_type: str | None = None


class ExpressionValidator:
    def __init__(
        self,
        registry: OperatorRegistry,
        allowed_fields: set[str],
        max_depth: int,
        group_fields: set[str] | None = None,
        field_types: dict[str, str] | None = None,
        complexity_limit: int | None = None,
    ) -> None:
        self.registry = registry
        self.allowed_fields = allowed_fields
        self.max_depth = max_depth
        self.group_fields = group_fields or set()
        self.field_types = field_types or {}
        self.complexity_limit = complexity_limit

    def validate(self, node: ExprNode) -> ValidationResult:
        errors: list[str] = []
        if node_depth(node) > self.max_depth:
            errors.append(f"Expression depth exceeds max depth {self.max_depth}.")
        if self.complexity_limit is not None and node_complexity(node) > self.complexity_limit:
            errors.append(f"Expression complexity exceeds limit {self.complexity_limit}.")
        inferred = self._infer_node_type(node, errors)
        if inferred is not None and inferred != "matrix":
            errors.append("Expression must evaluate to a matrix-valued signal.")
        self._validate_redundancy(node, errors)
        return ValidationResult(is_valid=not errors, errors=errors, output_type=inferred)

    def _infer_node_type(self, node: ExprNode, errors: list[str]) -> str | None:
        if isinstance(node, NumberNode):
            return "scalar"
        if isinstance(node, IdentifierNode):
            if node.name in self.field_types:
                return self.field_types[node.name]
            if node.name in self.group_fields:
                return "group"
            if node.name in self.allowed_fields:
                return "matrix"
            errors.append(f"Unknown field '{node.name}'.")
            return None
        if isinstance(node, UnaryOpNode):
            operand_type = self._infer_node_type(node.operand, errors)
            if operand_type not in {"matrix", "scalar"}:
                errors.append(f"Unary operator '{node.operator}' requires a numeric operand.")
                return None
            return operand_type
        if isinstance(node, BinaryOpNode):
            return self._infer_binary_type(node, errors)
        if isinstance(node, FunctionCallNode):
            if not self.registry.contains(node.name):
                errors.append(f"Unknown operator '{node.name}'.")
                return None
            spec = self.registry.get(node.name)
            if not spec.supports_arg_count(len(node.args)):
                errors.append(
                    f"Operator '{node.name}' expects between {spec.min_args} and {spec.max_args} args, "
                    f"received {len(node.args)}."
                )
                return None
            arg_types: list[str] = []
            for argument in node.args:
                inferred = self._infer_node_type(argument, errors)
                if inferred is None:
                    return None
                arg_types.append(inferred)
            compatible = spec.compatible_signatures(tuple(arg_types))
            if not compatible:
                errors.append(
                    f"Operator '{node.name}' is incompatible with input types {tuple(arg_types)}."
                )
                return None
            self._validate_operator_parameters(node, errors)
            return spec.resolve_output_type(tuple(arg_types))
        errors.append(f"Unsupported node type: {type(node)!r}")
        return None

    def _infer_binary_type(self, node: BinaryOpNode, errors: list[str]) -> str | None:
        left_type = self._infer_node_type(node.left, errors)
        right_type = self._infer_node_type(node.right, errors)
        if left_type is None or right_type is None:
            return None
        if left_type == "group" or right_type == "group":
            errors.append("Group field can only be used as the grouping argument in group operators.")
            return None
        if left_type == "scalar" and right_type == "scalar":
            return "scalar"
        if left_type in {"matrix", "scalar"} and right_type in {"matrix", "scalar"}:
            return "matrix"
        errors.append(f"Binary operator '{node.operator}' received incompatible types {(left_type, right_type)}.")
        return None

    def _validate_operator_parameters(self, node: FunctionCallNode, errors: list[str]) -> None:
        if node.name in WINDOWED_OPERATORS and len(node.args) >= 2:
            window_node = node.args[-1]
            if not isinstance(window_node, NumberNode) or int(window_node.value) != window_node.value:
                errors.append(f"Operator '{node.name}' requires an integer lookback window.")
            elif int(window_node.value) <= 0:
                errors.append(f"Operator '{node.name}' requires a positive lookback window.")
        if node.name == "clip" and len(node.args) == 3:
            lower, upper = node.args[1], node.args[2]
            if not isinstance(lower, NumberNode) or not isinstance(upper, NumberNode):
                errors.append("Clip bounds must be numeric literals.")
            elif lower.value >= upper.value:
                errors.append("Clip lower bound must be less than upper bound.")

    def _validate_redundancy(self, node: ExprNode, errors: list[str]) -> None:
        if isinstance(node, BinaryOpNode):
            left_expr = to_expression(node.left)
            right_expr = to_expression(node.right)
            if node.operator in {"-", "/"} and left_expr == right_expr:
                errors.append(f"Redundant binary expression '{left_expr}{node.operator}{right_expr}'.")
            if node.operator in {"+", "-"} and isinstance(node.right, NumberNode) and node.right.value == 0:
                errors.append("Redundant arithmetic with zero literal.")
            if node.operator == "*" and isinstance(node.right, NumberNode) and node.right.value == 1:
                errors.append("Redundant multiplication by one.")
        if isinstance(node, FunctionCallNode):
            if len(node.args) == 1 and isinstance(node.args[0], FunctionCallNode):
                child = node.args[0]
                if node.name == child.name and node.name in IDEMPOTENT_WRAPPERS:
                    errors.append(f"Redundant nested wrapper '{node.name}'.")
            if node.name == "rank" and len(node.args) == 1 and isinstance(node.args[0], FunctionCallNode):
                if node.args[0].name == "zscore":
                    errors.append("Redundant normalization chain 'rank(zscore(x))'.")
            if node.name == "zscore" and len(node.args) == 1 and isinstance(node.args[0], FunctionCallNode):
                if node.args[0].name == "rank":
                    errors.append("Redundant normalization chain 'zscore(rank(x))'.")
        for child in _iter_children(node):
            self._validate_redundancy(child, errors)


def validate_expression(
    node: ExprNode,
    registry: OperatorRegistry,
    allowed_fields: set[str],
    max_depth: int,
    group_fields: set[str] | None = None,
    field_types: dict[str, str] | None = None,
    complexity_limit: int | None = None,
) -> ValidationResult:
    return ExpressionValidator(
        registry=registry,
        allowed_fields=allowed_fields,
        max_depth=max_depth,
        group_fields=group_fields,
        field_types=field_types,
        complexity_limit=complexity_limit,
    ).validate(node)


def _iter_children(node: ExprNode) -> tuple[ExprNode, ...]:
    if isinstance(node, UnaryOpNode):
        return (node.operand,)
    if isinstance(node, BinaryOpNode):
        return (node.left, node.right)
    if isinstance(node, FunctionCallNode):
        return node.args
    return ()
