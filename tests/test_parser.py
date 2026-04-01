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
    assert {issue.reason_code for issue in result.issues} == {
        "validation_invalid_nesting",
        "validation_disallowed_field",
    }


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
    assert invalid.primary_reason_code == "validation_invalid_group_field"


def test_validator_maps_unknown_operator_reason_code() -> None:
    registry = build_registry(["rank"])
    result = validate_expression(
        parse_expression("mystery(close)"),
        registry,
        {"close"},
        max_depth=4,
    )

    assert not result.is_valid
    assert result.primary_reason_code == "validation_unknown_operator"


def test_validator_maps_operator_arity_mismatch_reason_code() -> None:
    registry = build_registry(["rank"])
    result = validate_expression(
        parse_expression("rank(close, 1)"),
        registry,
        {"close"},
        max_depth=4,
    )

    assert not result.is_valid
    assert result.primary_reason_code == "validation_operator_arity_mismatch"


def test_validator_maps_unsupported_combination_reason_code() -> None:
    registry = build_registry(["group_neutralize"])
    result = validate_expression(
        parse_expression("group_neutralize(close, close)"),
        registry,
        {"close"},
        max_depth=4,
        group_fields={"sector"},
        field_types={"close": "matrix", "sector": "group"},
    )

    assert not result.is_valid
    assert result.primary_reason_code == "validation_unsupported_combination"


def test_validator_maps_field_type_resolution_failure_reason_code() -> None:
    registry = build_registry(["rank"])
    result = validate_expression(
        parse_expression("-sector"),
        registry,
        {"close"},
        max_depth=4,
        group_fields={"sector"},
    )

    assert not result.is_valid
    assert result.primary_reason_code == "validation_field_type_resolution_failed"


def test_validator_maps_invalid_nesting_reason_code() -> None:
    registry = build_registry(["rank", "ts_mean"])
    result = validate_expression(
        parse_expression("rank(ts_mean(close, 2))"),
        registry,
        {"close"},
        max_depth=1,
    )

    assert not result.is_valid
    assert result.primary_reason_code == "validation_invalid_nesting"


@pytest.mark.parametrize(
    ("expression", "operators"),
    [
        ("rank(ts_mean(close, 10))", ["rank", "ts_mean"]),
        ("group_rank(ts_delta(close, 5), industry)", ["group_rank", "ts_delta"]),
        ("ts_rank(ts_delta(close, 1), 20)", ["ts_rank", "ts_delta"]),
        ("rank(close)", ["rank"]),
    ],
)
def test_validator_allows_cross_sectional_ops_outside_time_series(
    expression: str,
    operators: list[str],
) -> None:
    registry = build_registry(operators)
    result = validate_expression(
        parse_expression(expression),
        registry,
        {"close", "volume", "returns"},
        max_depth=10,
        group_fields={"industry", "sector"},
    )

    assert result.is_valid


@pytest.mark.parametrize(
    ("expression", "operators", "cross_sectional_operator", "time_series_operator"),
    [
        ("ts_mean(rank(close), 10)", ["ts_mean", "rank"], "rank", "ts_mean"),
        ("ts_corr(zscore(close), volume, 10)", ["ts_corr", "zscore"], "zscore", "ts_corr"),
        (
            "ts_delta(group_neutralize(close, sector), 5)",
            ["ts_delta", "group_neutralize"],
            "group_neutralize",
            "ts_delta",
        ),
        ("ts_std_dev(rank(returns), 20)", ["ts_std_dev", "rank"], "rank", "ts_std_dev"),
        ("ts_decay_linear(zscore(close), 5)", ["ts_decay_linear", "zscore"], "zscore", "ts_decay_linear"),
    ],
)
def test_validator_rejects_cross_sectional_ops_nested_inside_time_series(
    expression: str,
    operators: list[str],
    cross_sectional_operator: str,
    time_series_operator: str,
) -> None:
    registry = build_registry(operators)
    result = validate_expression(
        parse_expression(expression),
        registry,
        {"close", "volume", "returns"},
        max_depth=10,
        group_fields={"industry", "sector"},
    )

    assert not result.is_valid
    assert result.primary_reason_code == "validation_invalid_nesting"
    assert result.detail is not None
    assert cross_sectional_operator in result.detail
    assert time_series_operator in result.detail


@pytest.mark.parametrize(
    ("expression", "unit_left", "unit_right"),
    [
        ("close + volume", "price", "quantity"),
        ("high - sharesout", "price", "quantity"),
        ("sales + volume", "currency", "quantity"),
    ],
)
def test_validator_rejects_incompatible_unit_addition_and_subtraction(
    expression: str,
    unit_left: str,
    unit_right: str,
) -> None:
    registry = build_registry(
        [
            "rank",
            "zscore",
            "ts_mean",
            "ts_corr",
            "ts_delta",
            "ts_std_dev",
            "ts_decay_linear",
        ]
    )
    result = validate_expression(
        parse_expression(expression),
        registry,
        {
            "close",
            "open",
            "high",
            "low",
            "vwap",
            "volume",
            "adv20",
            "sharesout",
            "sales",
            "returns",
        },
        max_depth=10,
    )

    assert not result.is_valid
    assert result.primary_reason_code == "validation_unit_incompatible"
    assert result.detail is not None
    assert unit_left in result.detail
    assert unit_right in result.detail


@pytest.mark.parametrize(
    ("expression", "operators"),
    [
        ("close + open", []),
        ("close - vwap", []),
        ("close * volume", []),
        ("close / volume", []),
        ("rank(close) + rank(volume)", ["rank"]),
        ("ts_mean(close, 10) + volume", ["ts_mean"]),
    ],
)
def test_validator_allows_compatible_or_skipped_unit_combinations(
    expression: str,
    operators: list[str],
) -> None:
    registry = build_registry([*operators, "rank", "zscore", "ts_mean"])
    result = validate_expression(
        parse_expression(expression),
        registry,
        {
            "close",
            "open",
            "high",
            "low",
            "vwap",
            "volume",
            "adv20",
            "sharesout",
            "sales",
            "returns",
        },
        max_depth=10,
    )

    assert "validation_unit_incompatible" not in {issue.reason_code for issue in result.issues}
    assert result.is_valid


def test_validator_maps_semantic_invalid_reason_code() -> None:
    registry = build_registry(["ts_mean"])
    result = validate_expression(
        parse_expression("ts_mean(close, 0)"),
        registry,
        {"close"},
        max_depth=4,
    )

    assert not result.is_valid
    assert result.primary_reason_code == "validation_semantic_invalid"


def test_registry_keeps_legacy_aliases_but_exposes_brain_canonical_names() -> None:
    registry = build_registry(["delta", "correlation", "covariance", "decay_linear", "ts_mean", "ts_std"])

    assert registry.contains("delta")
    assert registry.contains("ts_delta")
    assert registry.contains("correlation")
    assert registry.contains("ts_corr")
    assert registry.contains("covariance")
    assert registry.contains("ts_covariance")
    assert registry.contains("decay_linear")
    assert registry.contains("ts_decay_linear")
    assert registry.contains("ts_std")
    assert registry.contains("ts_std_dev")
