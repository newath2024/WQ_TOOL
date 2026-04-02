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

_CROSS_SECTIONAL_NESTING_OPERATORS = frozenset(
    {
        "rank",
        "zscore",
        "group_rank",
        "group_zscore",
        "group_neutralize",
        "normalize",
    }
)
_PRICE_FIELDS = frozenset({"close", "open", "high", "low", "vwap"})
_QUANTITY_FIELDS = frozenset({"volume", "adv20", "sharesout"})
_CURRENCY_FIELDS = frozenset({"sales", "assets", "debt", "equity", "ebitda", "net_income", "revenue", "cash", "capex"})
_RATIO_FIELDS = frozenset({"returns"})
_MARKET_VALUE_FIELDS = frozenset({"cap"})
_UNIT_CLASS_BY_FIELD = {
    **{name: "price" for name in _PRICE_FIELDS},
    **{name: "quantity" for name in _QUANTITY_FIELDS},
    **{name: "currency" for name in _CURRENCY_FIELDS},
    **{name: "ratio" for name in _RATIO_FIELDS},
    **{name: "market_value" for name in _MARKET_VALUE_FIELDS},
}
_DIMENSIONLESS_FUNCTIONS = frozenset(
    {
        "rank",
        "zscore",
        "group_rank",
        "group_zscore",
        "group_neutralize",
        "normalize",
    }
)


@dataclass(slots=True)
class ValidationIssue:
    reason_code: str
    detail: str


@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]
    output_type: str | None = None
    issues: tuple[ValidationIssue, ...] = ()
    primary_reason_code: str | None = None
    detail: str | None = None


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
        issues: list[ValidationIssue] = []
        if node_depth(node) > self.max_depth:
            self._add_issue(
                errors,
                issues,
                "validation_depth_exceeded",
                f"Expression depth exceeds max depth {self.max_depth}.",
            )
        if self.complexity_limit is not None and node_complexity(node) > self.complexity_limit:
            self._add_issue(
                errors,
                issues,
                "complexity_exceeded",
                f"Expression complexity exceeds limit {self.complexity_limit}.",
            )
        self._validate_cross_sectional_inside_time_series(node, errors, issues)
        inferred = self._infer_node_type(node, errors, issues)
        if inferred is not None and inferred != "matrix":
            self._add_issue(
                errors,
                issues,
                "validation_semantic_invalid",
                "Expression must evaluate to a matrix-valued signal.",
            )
        self._validate_unit_compatibility(node, errors, issues)
        self._validate_redundancy(node, errors, issues)
        primary_issue = issues[0] if issues else None
        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            output_type=inferred,
            issues=tuple(issues),
            primary_reason_code=primary_issue.reason_code if primary_issue else None,
            detail=primary_issue.detail if primary_issue else None,
        )

    def _infer_node_type(
        self,
        node: ExprNode,
        errors: list[str],
        issues: list[ValidationIssue],
    ) -> str | None:
        if isinstance(node, NumberNode):
            return "scalar"
        if isinstance(node, IdentifierNode):
            if node.name in self.field_types:
                return self.field_types[node.name]
            if node.name in self.group_fields:
                return "group"
            if node.name in self.allowed_fields:
                return "matrix"
            self._add_issue(
                errors,
                issues,
                "validation_disallowed_field",
                f"Unknown field '{node.name}'.",
            )
            return None
        if isinstance(node, UnaryOpNode):
            operand_type = self._infer_node_type(node.operand, errors, issues)
            if operand_type not in {"matrix", "scalar"}:
                self._add_issue(
                    errors,
                    issues,
                    "validation_field_type_resolution_failed",
                    f"Unary operator '{node.operator}' requires a numeric operand.",
                )
                return None
            return operand_type
        if isinstance(node, BinaryOpNode):
            return self._infer_binary_type(node, errors, issues)
        if isinstance(node, FunctionCallNode):
            if not self.registry.contains(node.name):
                self._add_issue(
                    errors,
                    issues,
                    "validation_unknown_operator",
                    f"Unknown operator '{node.name}'.",
                )
                return None
            spec = self.registry.get(node.name)
            if not spec.supports_arg_count(len(node.args)):
                self._add_issue(
                    errors,
                    issues,
                    "validation_operator_arity_mismatch",
                    f"Operator '{node.name}' expects between {spec.min_args} and {spec.max_args} args, "
                    f"received {len(node.args)}.",
                )
                return None
            arg_types: list[str] = []
            for argument in node.args:
                inferred = self._infer_node_type(argument, errors, issues)
                if inferred is None:
                    return None
                arg_types.append(inferred)
            compatible = spec.compatible_signatures(tuple(arg_types))
            if not compatible:
                self._add_issue(
                    errors,
                    issues,
                    "validation_unsupported_combination",
                    f"Operator '{node.name}' is incompatible with input types {tuple(arg_types)}.",
                )
                return None
            self._validate_operator_parameters(node, spec, errors, issues)
            return spec.resolve_output_type(tuple(arg_types))
        self._add_issue(
            errors,
            issues,
            "validation_unknown_error",
            f"Unsupported node type: {type(node)!r}",
        )
        return None

    def _infer_binary_type(
        self,
        node: BinaryOpNode,
        errors: list[str],
        issues: list[ValidationIssue],
    ) -> str | None:
        left_type = self._infer_node_type(node.left, errors, issues)
        right_type = self._infer_node_type(node.right, errors, issues)
        if left_type is None or right_type is None:
            return None
        if left_type == "group" or right_type == "group":
            self._add_issue(
                errors,
                issues,
                "validation_invalid_group_field",
                "Group field can only be used as the grouping argument in group operators.",
            )
            return None
        if left_type == "scalar" and right_type == "scalar":
            return "scalar"
        if left_type in {"matrix", "scalar"} and right_type in {"matrix", "scalar"}:
            return "matrix"
        self._add_issue(
            errors,
            issues,
            "validation_field_type_resolution_failed",
            f"Binary operator '{node.operator}' received incompatible types {(left_type, right_type)}.",
        )
        return None

    def _validate_operator_parameters(
        self,
        node: FunctionCallNode,
        spec,
        errors: list[str],
        issues: list[ValidationIssue],
    ) -> None:
        if node.name in WINDOWED_OPERATORS and len(node.args) >= 2:
            window_node = node.args[-1]
            if not isinstance(window_node, NumberNode) or int(window_node.value) != window_node.value:
                self._add_issue(
                    errors,
                    issues,
                    "validation_semantic_invalid",
                    f"Operator '{node.name}' requires an integer lookback window.",
                )
            elif int(window_node.value) <= 0:
                self._add_issue(
                    errors,
                    issues,
                    "validation_semantic_invalid",
                    f"Operator '{node.name}' requires a positive lookback window.",
                )
        if node.name == "clip" and len(node.args) == 3:
            lower, upper = node.args[1], node.args[2]
            if not isinstance(lower, NumberNode) or not isinstance(upper, NumberNode):
                self._add_issue(
                    errors,
                    issues,
                    "validation_semantic_invalid",
                    "Clip bounds must be numeric literals.",
                )
            elif lower.value >= upper.value:
                self._add_issue(
                    errors,
                    issues,
                    "validation_semantic_invalid",
                    "Clip lower bound must be less than upper bound.",
                )
        if spec.has_tag("requires_positive_input") and node.args:
            literal_value = _literal_numeric_value(node.args[0])
            if literal_value is not None and literal_value <= 0:
                self._add_issue(
                    errors,
                    issues,
                    "validation_semantic_invalid",
                    f"Operator '{node.name}' requires a positive input.",
                )

    def _validate_redundancy(
        self,
        node: ExprNode,
        errors: list[str],
        issues: list[ValidationIssue],
    ) -> None:
        if isinstance(node, BinaryOpNode):
            left_expr = to_expression(node.left)
            right_expr = to_expression(node.right)
            if node.operator in {"-", "/"} and left_expr == right_expr:
                self._add_issue(
                    errors,
                    issues,
                    "redundant_expression",
                    f"Redundant binary expression '{left_expr}{node.operator}{right_expr}'.",
                )
            if node.operator in {"+", "-"} and isinstance(node.right, NumberNode) and node.right.value == 0:
                self._add_issue(
                    errors,
                    issues,
                    "redundant_expression",
                    "Redundant arithmetic with zero literal.",
                )
            if node.operator == "*" and isinstance(node.right, NumberNode) and node.right.value == 1:
                self._add_issue(
                    errors,
                    issues,
                    "redundant_expression",
                    "Redundant multiplication by one.",
                )
        if isinstance(node, FunctionCallNode):
            if len(node.args) == 1 and isinstance(node.args[0], FunctionCallNode):
                child = node.args[0]
                if node.name == child.name and node.name in IDEMPOTENT_WRAPPERS:
                    self._add_issue(
                        errors,
                        issues,
                        "redundant_expression",
                        f"Redundant nested wrapper '{node.name}'.",
                    )
            if node.name == "rank" and len(node.args) == 1 and isinstance(node.args[0], FunctionCallNode):
                if node.args[0].name == "zscore":
                    self._add_issue(
                        errors,
                        issues,
                        "redundant_expression",
                        "Redundant normalization chain 'rank(zscore(x))'.",
                    )
            if node.name == "zscore" and len(node.args) == 1 and isinstance(node.args[0], FunctionCallNode):
                if node.args[0].name == "rank":
                    self._add_issue(
                        errors,
                        issues,
                        "redundant_expression",
                        "Redundant normalization chain 'zscore(rank(x))'.",
                    )
        for child in _iter_children(node):
            self._validate_redundancy(child, errors, issues)

    def _validate_unit_compatibility(
        self,
        node: ExprNode,
        errors: list[str],
        issues: list[ValidationIssue],
    ) -> None:
        if isinstance(node, BinaryOpNode) and node.operator in {"+", "-"}:
            left_unit = self._infer_unit_class(node.left)
            right_unit = self._infer_unit_class(node.right)
            if left_unit is not None and right_unit is not None and left_unit != right_unit:
                self._add_issue(
                    errors,
                    issues,
                    "validation_unit_incompatible",
                    (
                        f"Binary operator '{node.operator}' cannot combine "
                        f"'{left_unit}' and '{right_unit}' units."
                    ),
                )
        for child in _iter_children(node):
            self._validate_unit_compatibility(child, errors, issues)

    def _infer_unit_class(self, node: ExprNode) -> str | None:
        if isinstance(node, IdentifierNode):
            return _UNIT_CLASS_BY_FIELD.get(node.name)
        if isinstance(node, UnaryOpNode):
            return self._infer_unit_class(node.operand)
        if isinstance(node, BinaryOpNode):
            if node.operator in {"*", "/"}:
                return None
            if node.operator not in {"+", "-"}:
                return None
            left_unit = self._infer_unit_class(node.left)
            right_unit = self._infer_unit_class(node.right)
            if left_unit is None or right_unit is None or left_unit != right_unit:
                return None
            return left_unit
        if isinstance(node, FunctionCallNode):
            if node.name.startswith("ts_") or node.name in _DIMENSIONLESS_FUNCTIONS:
                return None
            if node.name == "abs" and len(node.args) == 1:
                return self._infer_unit_class(node.args[0])
        return None

    def _validate_cross_sectional_inside_time_series(
        self,
        node: ExprNode,
        errors: list[str],
        issues: list[ValidationIssue],
        *,
        enclosing_time_series_operator: str | None = None,
    ) -> None:
        if isinstance(node, FunctionCallNode):
            if (
                enclosing_time_series_operator is not None
                and node.name in _CROSS_SECTIONAL_NESTING_OPERATORS
            ):
                self._add_issue(
                    errors,
                    issues,
                    "validation_invalid_nesting",
                    (
                        f"Cross-sectional operator '{node.name}' cannot be nested inside "
                        f"time-series operator '{enclosing_time_series_operator}'."
                    ),
                )
            next_enclosing_time_series_operator = (
                node.name
                if node.name.startswith("ts_")
                else enclosing_time_series_operator
            )
            for child in node.args:
                self._validate_cross_sectional_inside_time_series(
                    child,
                    errors,
                    issues,
                    enclosing_time_series_operator=next_enclosing_time_series_operator,
                )
            return
        for child in _iter_children(node):
            self._validate_cross_sectional_inside_time_series(
                child,
                errors,
                issues,
                enclosing_time_series_operator=enclosing_time_series_operator,
            )

    @staticmethod
    def _add_issue(
        errors: list[str],
        issues: list[ValidationIssue],
        reason_code: str,
        detail: str,
    ) -> None:
        errors.append(detail)
        issues.append(ValidationIssue(reason_code=reason_code, detail=detail))


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


def has_nesting_violation(node: ExprNode) -> bool:
    """Fast check: is any cross-sectional operator nested inside a ts_* operator?

    This is a lightweight alternative to the full validator, optimised for use
    in hot generation loops where we only care about the nesting constraint.
    """
    return _scan_nesting(node, inside_ts=False)


def _scan_nesting(node: ExprNode, *, inside_ts: bool) -> bool:
    if isinstance(node, FunctionCallNode):
        if inside_ts and node.name in _CROSS_SECTIONAL_NESTING_OPERATORS:
            return True
        child_context = node.name.startswith("ts_") or inside_ts
        return any(_scan_nesting(child, inside_ts=child_context) for child in node.args)
    for child in _iter_children(node):
        if _scan_nesting(child, inside_ts=inside_ts):
            return True
    return False


def _iter_children(node: ExprNode) -> tuple[ExprNode, ...]:
    if isinstance(node, UnaryOpNode):
        return (node.operand,)
    if isinstance(node, BinaryOpNode):
        return (node.left, node.right)
    if isinstance(node, FunctionCallNode):
        return node.args
    return ()


def _literal_numeric_value(node: ExprNode) -> float | None:
    if isinstance(node, NumberNode):
        return float(node.value)
    if isinstance(node, UnaryOpNode) and node.operator == "-" and isinstance(node.operand, NumberNode):
        return -float(node.operand.value)
    return None
