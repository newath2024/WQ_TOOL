from __future__ import annotations

import numpy as np

from alpha.evaluator import evaluate_expression
from alpha.parser import parse_expression
from alpha.validator import validate_expression
from features.registry import build_registry


def test_evaluator_handles_nested_expression(sample_wide_fields) -> None:
    registry = build_registry(["rank", "delta"])
    node = parse_expression("rank(delta(close, 1))")
    validation = validate_expression(node, registry, {"close"}, max_depth=4)

    assert validation.is_valid
    result = evaluate_expression(node, sample_wide_fields.fields, registry)

    assert result.shape == sample_wide_fields["close"].shape
    assert np.isclose(result.iloc[10].max(skipna=True), 1.0)


def test_evaluator_handles_group_neutralization(sample_wide_fields) -> None:
    close = sample_wide_fields["close"].iloc[:5]
    registry = build_registry(["group_neutralize"])
    node = parse_expression("group_neutralize(close, sector)")
    validation = validate_expression(
        node,
        registry,
        {"close"},
        max_depth=4,
        group_fields={"sector"},
    )

    assert validation.is_valid
    result = evaluate_expression(node, sample_wide_fields.fields, registry, group_fields=sample_wide_fields.group_fields)

    assert np.isclose(result.loc[close.index[0], ["AAA", "BBB"]].mean(), 0.0)
    assert np.isclose(result.loc[close.index[0], ["CCC", "DDD"]].mean(), 0.0)
