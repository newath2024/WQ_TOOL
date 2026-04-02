from __future__ import annotations

from dataclasses import replace

import pytest

from alpha.ast_nodes import FunctionCallNode
from alpha.parser import parse_expression
from alpha.validator import validate_expression
from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import build_registry
from generator.genome import (
    ComplexityGene,
    FeatureGene,
    Genome,
    HorizonGene,
    RegimeGene,
    TransformGene,
    TurnoverGene,
    WrapperGene,
)
from generator.genome_builder import GenomeBuilder
from generator.grammar import MOTIF_LIBRARY, MotifGrammar


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
        "validation_depth_exceeded",
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


def test_validator_maps_depth_exceeded_reason_code() -> None:
    registry = build_registry(["sign", "rank", "ts_decay_linear", "ts_delta", "ts_mean"])
    result = validate_expression(
        parse_expression("sign(rank(ts_decay_linear((ts_delta(close,5)*ts_mean(volume,10)),10)))"),
        registry,
        {"close", "volume"},
        max_depth=5,
    )

    assert not result.is_valid
    assert result.primary_reason_code == "validation_depth_exceeded"
    assert "validation_invalid_nesting" not in {issue.reason_code for issue in result.issues}


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


def _build_generator_test_field_registry() -> FieldRegistry:
    def make_field(
        name: str,
        *,
        category: str,
        field_type: str = "matrix",
        field_score: float = 0.9,
    ) -> FieldSpec:
        return FieldSpec(
            name=name,
            dataset="test",
            field_type=field_type,
            coverage=1.0,
            alpha_usage_count=5,
            category=category,
            runtime_available=True,
            field_score=field_score,
            category_weight=0.8,
        )

    return FieldRegistry(
        fields={
            "open": make_field("open", category="price", field_score=0.92),
            "high": make_field("high", category="price", field_score=0.91),
            "low": make_field("low", category="price", field_score=0.90),
            "close": make_field("close", category="price", field_score=0.95),
            "volume": make_field("volume", category="volume", field_score=0.94),
            "adv20": make_field("adv20", category="liquidity", field_score=0.89),
            "returns": make_field("returns", category="price", field_score=0.88),
            "sales": make_field("sales", category="fundamental", field_score=0.93),
            "assets": make_field("assets", category="fundamental", field_score=0.87),
            "equity": make_field("equity", category="fundamental", field_score=0.86),
            "target_price": make_field("target_price", category="analyst", field_score=0.84),
            "model_score": make_field("model_score", category="model", field_score=0.83),
            "beta": make_field("beta", category="risk", field_score=0.82),
            "sector": make_field("sector", category="group", field_type="vector", field_score=0.80),
            "industry": make_field("industry", category="group", field_type="vector", field_score=0.79),
        }
    )


def _build_generator_test_config() -> GenerationConfig:
    return GenerationConfig(
        allowed_fields=[
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adv20",
            "returns",
            "sales",
            "assets",
            "equity",
            "target_price",
            "model_score",
            "beta",
        ],
        allowed_operators=[
            "ts_delta",
            "ts_mean",
            "ts_std_dev",
            "ts_decay_linear",
            "ts_rank",
            "ts_sum",
            "ts_corr",
            "ts_covariance",
            "rank",
            "zscore",
            "sign",
            "abs",
            "log",
            "group_neutralize",
        ],
        lookbacks=[2, 3, 5, 10],
        max_depth=8,
        complexity_limit=20,
        template_count=8,
        grammar_count=8,
        mutation_count=4,
        normalization_wrappers=["rank", "zscore", "sign"],
        random_seed=17,
    )


def _build_regression_genome(motif: str, *, conditioning_mode: str = "none") -> Genome:
    conditioning_field = "adv20" if conditioning_mode == "liquidity_gate" else ""
    if motif == "quality_score":
        feature_gene = FeatureGene(
            primary_field="sales",
            primary_family="fundamental",
            secondary_field="assets",
            secondary_family="fundamental",
            auxiliary_field="beta",
            auxiliary_family="risk",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(motif=motif, primitive_transform="")
    elif motif == "price_volume_divergence":
        feature_gene = FeatureGene(
            primary_field="close",
            primary_family="price",
            secondary_field="volume",
            secondary_family="volume",
            auxiliary_field="beta",
            auxiliary_family="risk",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(motif=motif, primitive_transform="ts_corr")
    elif motif == "momentum":
        feature_gene = FeatureGene(
            primary_field="close",
            primary_family="price",
            auxiliary_field="adv20",
            auxiliary_family="liquidity",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(motif=motif, primitive_transform="ts_delta")
    elif motif == "mean_reversion":
        feature_gene = FeatureGene(
            primary_field="close",
            primary_family="price",
            auxiliary_field="adv20",
            auxiliary_family="liquidity",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(motif=motif, primitive_transform="ts_mean")
    elif motif == "volatility_adjusted_momentum":
        feature_gene = FeatureGene(
            primary_field="close",
            primary_family="price",
            auxiliary_field="adv20",
            auxiliary_family="liquidity",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(
            motif=motif,
            primitive_transform="ts_delta",
            secondary_transform="ts_std_dev",
        )
    elif motif == "spread":
        feature_gene = FeatureGene(
            primary_field="close",
            primary_family="price",
            secondary_field="volume",
            secondary_family="volume",
            auxiliary_field="adv20",
            auxiliary_family="liquidity",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(
            motif=motif,
            primitive_transform="ts_mean",
            secondary_transform="ts_mean",
        )
    elif motif == "ratio":
        feature_gene = FeatureGene(
            primary_field="close",
            primary_family="price",
            secondary_field="volume",
            secondary_family="volume",
            auxiliary_field="adv20",
            auxiliary_family="liquidity",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(
            motif=motif,
            primitive_transform="ts_mean",
            secondary_transform="ts_mean",
        )
    elif motif == "conditional_momentum":
        feature_gene = FeatureGene(
            primary_field="close",
            primary_family="price",
            secondary_field="volume",
            secondary_family="volume",
            auxiliary_field="beta",
            auxiliary_family="risk",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(
            motif=motif,
            primitive_transform="ts_delta",
            secondary_transform="ts_mean",
        )
    elif motif == "residualized_signal":
        feature_gene = FeatureGene(
            primary_field="close",
            primary_family="price",
            secondary_field="volume",
            secondary_family="volume",
            auxiliary_field="adv20",
            auxiliary_family="liquidity",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(
            motif=motif,
            primitive_transform="ts_mean",
            secondary_transform="ts_mean",
        )
    elif motif == "regime_conditioned_signal":
        feature_gene = FeatureGene(
            primary_field="close",
            primary_family="price",
            secondary_field="volume",
            secondary_family="volume",
            auxiliary_field="beta",
            auxiliary_family="risk",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(
            motif=motif,
            primitive_transform="ts_delta",
            secondary_transform="ts_std_dev",
        )
    elif motif == "group_relative_signal":
        feature_gene = FeatureGene(
            primary_field="close",
            primary_family="price",
            auxiliary_field="adv20",
            auxiliary_family="liquidity",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(motif=motif, primitive_transform="ts_mean")
    elif motif == "liquidity_conditioned_signal":
        feature_gene = FeatureGene(
            primary_field="close",
            primary_family="price",
            secondary_field="volume",
            secondary_family="volume",
            auxiliary_field="adv20",
            auxiliary_family="liquidity",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(
            motif=motif,
            primitive_transform="ts_delta",
            secondary_transform="ts_mean",
        )
    else:
        feature_gene = FeatureGene(
            primary_field="close",
            primary_family="price",
            secondary_field="volume",
            secondary_family="volume",
            auxiliary_field="adv20",
            auxiliary_family="liquidity",
            group_field="sector",
            liquidity_field="adv20",
        )
        transform_gene = TransformGene(
            motif=motif,
            primitive_transform="ts_mean",
            secondary_transform="ts_mean",
        )
    return Genome(
        feature_gene=feature_gene,
        transform_gene=transform_gene,
        horizon_gene=HorizonGene(fast_window=3, slow_window=5, context_window=10),
        wrapper_gene=WrapperGene(),
        regime_gene=RegimeGene(
            conditioning_mode=conditioning_mode,
            conditioning_field=conditioning_field,
        ),
        turnover_gene=TurnoverGene(
            smoothing_operator="ts_decay_linear",
            smoothing_window=5,
            turnover_hint=-0.3,
        ),
        complexity_gene=ComplexityGene(target_depth=4, binary_branching=2, wrapper_budget=0),
        source_mode="nesting_regression",
    )


def _assert_no_cross_sectional_inside_time_series(expression: str) -> None:
    config = _build_generator_test_config()
    field_registry = _build_generator_test_field_registry()
    result = validate_expression(
        parse_expression(expression),
        build_registry(config.allowed_operators),
        set(config.allowed_fields),
        max_depth=20,
        group_fields={
            spec.name
            for spec in field_registry.fields.values()
            if spec.operator_type == "group"
        },
    )
    assert "validation_invalid_nesting" not in {issue.reason_code for issue in result.issues}


@pytest.mark.parametrize(
    ("motif", "conditioning_mode"),
    [
        ("momentum", "none"),
        ("mean_reversion", "none"),
        ("volatility_adjusted_momentum", "none"),
        ("spread", "none"),
        ("ratio", "none"),
        ("quality_score", "none"),
        ("price_volume_divergence", "none"),
        ("conditional_momentum", "none"),
        ("residualized_signal", "none"),
        ("regime_conditioned_signal", "none"),
        ("group_relative_signal", "none"),
        ("liquidity_conditioned_signal", "none"),
        ("spread", "volatility_gate"),
        ("spread", "liquidity_gate"),
        ("quality_score", "group_neutralize"),
        ("group_relative_signal", "group_neutralize"),
    ],
)
def test_grammar_render_keeps_cross_sectional_ops_outside_time_series_after_smoothing(
    motif: str,
    conditioning_mode: str,
) -> None:
    grammar = MotifGrammar()
    render = grammar.render(_build_regression_genome(motif, conditioning_mode=conditioning_mode))

    _assert_no_cross_sectional_inside_time_series(render.expression)


def test_random_smoothed_genome_render_never_nests_cross_sectional_ops_inside_time_series() -> None:
    config = _build_generator_test_config()
    field_registry = _build_generator_test_field_registry()
    registry = build_registry(config.allowed_operators)
    group_fields = {
        spec.name
        for spec in field_registry.fields.values()
        if spec.operator_type == "group"
    }
    builder = GenomeBuilder(
        generation_config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=registry,
        field_registry=field_registry,
        seed=23,
    )
    grammar = MotifGrammar()
    preferred_family_by_motif = {
        "price_volume_divergence": "price",
        "quality_score": "fundamental",
        "conditional_momentum": "price",
        "spread": "price",
        "ratio": "price",
        "residualized_signal": "price",
        "regime_conditioned_signal": "price",
        "group_relative_signal": "price",
        "liquidity_conditioned_signal": "price",
        "momentum": "price",
        "mean_reversion": "price",
        "volatility_adjusted_momentum": "price",
    }

    total_checked = 0
    for motif in MOTIF_LIBRARY:
        for _ in range(30):
            genome = builder.build_parent_seeded_genome(
                motif=motif,
                primary_family=preferred_family_by_motif.get(motif, "price"),
                source_mode="nesting_regression",
            )
            smoothed_genome = replace(
                genome,
                turnover_gene=TurnoverGene(
                    smoothing_operator="ts_decay_linear",
                    smoothing_window=max(2, genome.horizon_gene.slow_window or genome.horizon_gene.fast_window),
                    turnover_hint=genome.turnover_gene.turnover_hint or -0.3,
                ),
            )
            render = grammar.render(smoothed_genome)
            result = validate_expression(
                parse_expression(render.expression),
                registry,
                set(config.allowed_fields),
                max_depth=20,
                group_fields=group_fields,
            )
            assert "validation_invalid_nesting" not in {issue.reason_code for issue in result.issues}, render.expression
            total_checked += 1

    assert total_checked >= 300
