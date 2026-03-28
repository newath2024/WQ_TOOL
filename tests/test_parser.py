from __future__ import annotations

import pytest

from alpha.ast_nodes import FunctionCallNode
from alpha.parser import parse_expression
from alpha.validator import validate_expression
from features.registry import build_registry


def test_parse_nested_expression() -> None:
    node = parse_expression("rank(delta(close, 5))")
    assert isinstance(node, FunctionCallNode)
    assert node.name == "rank"


def test_parser_rejects_unsafe_constructs() -> None:
    with pytest.raises(ValueError):
        parse_expression("close.__class__")


def test_validator_rejects_unknown_fields_and_depth() -> None:
    registry = build_registry(["rank", "delta"])
    node = parse_expression("rank(delta(foo, 5))")
    result = validate_expression(node, registry, {"close"}, max_depth=2)

    assert not result.is_valid
    assert any("Unknown field 'foo'" in error for error in result.errors)
    assert any("depth exceeds" in error for error in result.errors)


def test_validator_handles_group_operator_field_types() -> None:
    registry = build_registry(["group_neutralize"])

    valid = validate_expression(
        parse_expression("group_neutralize(close, sector)"),
        registry,
        {"close"},
        max_depth=4,
        group_fields={"sector"},
    )
    invalid = validate_expression(
        parse_expression("close + sector"),
        registry,
        {"close"},
        max_depth=4,
        group_fields={"sector"},
    )

    assert valid.is_valid
    assert not invalid.is_valid
    assert any("can only be used as the grouping argument" in error for error in invalid.errors)
