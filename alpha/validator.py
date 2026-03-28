from __future__ import annotations

from dataclasses import dataclass

from alpha.ast_nodes import BinaryOpNode, ExprNode, FunctionCallNode, IdentifierNode, NumberNode, UnaryOpNode, node_depth
from features.registry import GROUP_OPERATORS, OperatorRegistry, WINDOWED_OPERATORS


@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]


class ExpressionValidator:
    def __init__(
        self,
        registry: OperatorRegistry,
        allowed_fields: set[str],
        max_depth: int,
        group_fields: set[str] | None = None,
    ) -> None:
        self.registry = registry
        self.allowed_fields = allowed_fields
        self.max_depth = max_depth
        self.group_fields = group_fields or set()

    def validate(self, node: ExprNode) -> ValidationResult:
        errors: list[str] = []
        if node_depth(node) > self.max_depth:
            errors.append(f"Expression depth exceeds max depth {self.max_depth}.")
        self._validate_node(node, errors, expect_group_identifier=False)
        return ValidationResult(is_valid=not errors, errors=errors)

    def _validate_node(
        self,
        node: ExprNode,
        errors: list[str],
        expect_group_identifier: bool,
    ) -> None:
        if isinstance(node, NumberNode):
            return
        if isinstance(node, IdentifierNode):
            if expect_group_identifier:
                if node.name not in self.group_fields:
                    errors.append(f"Unknown group field '{node.name}'.")
                return
            if node.name not in self.allowed_fields:
                if node.name in self.group_fields:
                    errors.append(
                        f"Group field '{node.name}' can only be used as the grouping argument in group operators."
                    )
                else:
                    errors.append(f"Unknown field '{node.name}'.")
            return
        if isinstance(node, UnaryOpNode):
            self._validate_node(node.operand, errors, expect_group_identifier=False)
            return
        if isinstance(node, BinaryOpNode):
            self._validate_node(node.left, errors, expect_group_identifier=False)
            self._validate_node(node.right, errors, expect_group_identifier=False)
            return
        if isinstance(node, FunctionCallNode):
            if not self.registry.contains(node.name):
                errors.append(f"Unknown operator '{node.name}'.")
                return
            spec = self.registry.get(node.name)
            if not spec.min_args <= len(node.args) <= spec.max_args:
                errors.append(
                    f"Operator '{node.name}' expects between {spec.min_args} and {spec.max_args} args, "
                    f"received {len(node.args)}."
                )

            if node.name in GROUP_OPERATORS:
                if len(node.args) != 2:
                    errors.append(f"Operator '{node.name}' requires exactly 2 args.")
                    return
                self._validate_node(node.args[0], errors, expect_group_identifier=False)
                self._validate_node(node.args[1], errors, expect_group_identifier=True)
                return

            for argument in node.args:
                self._validate_node(argument, errors, expect_group_identifier=False)

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
            return
        errors.append(f"Unsupported node type: {type(node)!r}")


def validate_expression(
    node: ExprNode,
    registry: OperatorRegistry,
    allowed_fields: set[str],
    max_depth: int,
    group_fields: set[str] | None = None,
) -> ValidationResult:
    return ExpressionValidator(
        registry=registry,
        allowed_fields=allowed_fields,
        max_depth=max_depth,
        group_fields=group_fields,
    ).validate(node)
