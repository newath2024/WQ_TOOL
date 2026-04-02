from __future__ import annotations

import pytest

from alpha.parser import parse_expression
from alpha.validator import ValidationResult, has_nesting_violation
from core.config import AdaptiveGenerationConfig, GenerationConfig, RepairPolicyConfig
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import build_default_registry, build_registry
from generator.engine import AlphaGenerationEngine
from generator.genome import (
    ComplexityGene,
    FeatureGene,
    Genome,
    GenomeRenderResult,
    HorizonGene,
    RegimeGene,
    TransformGene,
    TurnoverGene,
    WrapperGene,
)
from generator.grammar import MotifGrammar
from generator.repair_policy import RepairPolicy


def _build_field_registry() -> FieldRegistry:
    return FieldRegistry(
        fields={
            "close": FieldSpec(
                name="close",
                dataset="test",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=1,
                category="price",
                runtime_available=True,
                field_score=1.0,
                category_weight=1.0,
            ),
            "volume": FieldSpec(
                name="volume",
                dataset="test",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=1,
                category="volume",
                runtime_available=True,
                field_score=1.0,
                category_weight=1.0,
            ),
            "sector": FieldSpec(
                name="sector",
                dataset="test",
                field_type="vector",
                coverage=1.0,
                alpha_usage_count=1,
                category="group",
                runtime_available=True,
                field_score=1.0,
                category_weight=1.0,
            ),
        }
    )


def _build_generation_config() -> GenerationConfig:
    return GenerationConfig(
        allowed_fields=["close", "volume", "sector"],
        allowed_operators=["rank", "zscore", "ts_mean", "ts_delta", "ts_decay_linear"],
        lookbacks=[5, 10, 20],
        max_depth=8,
        complexity_limit=20,
        template_count=4,
        grammar_count=4,
        mutation_count=4,
        normalization_wrappers=["rank", "zscore", "sign"],
        random_seed=7,
    )


def _build_genome(
    *,
    motif: str = "momentum",
    pre_wrappers: tuple[str, ...] = (),
    post_wrappers: tuple[str, ...] = (),
    smoothing_operator: str = "",
    smoothing_window: int = 0,
) -> Genome:
    return Genome(
        feature_gene=FeatureGene(
            primary_field="close",
            primary_family="price",
            secondary_field="volume",
            secondary_family="volume",
            group_field="sector",
        ),
        transform_gene=TransformGene(motif=motif, primitive_transform="ts_delta", secondary_transform="ts_mean"),
        horizon_gene=HorizonGene(fast_window=5, slow_window=10, context_window=10),
        wrapper_gene=WrapperGene(pre_wrappers=pre_wrappers, post_wrappers=post_wrappers),
        regime_gene=RegimeGene(),
        turnover_gene=TurnoverGene(smoothing_operator=smoothing_operator, smoothing_window=smoothing_window),
        complexity_gene=ComplexityGene(),
    )


def _build_render_result(genome: Genome, expression: str) -> GenomeRenderResult:
    return GenomeRenderResult(
        genome=genome,
        expression=expression,
        normalized_expression=expression,
        family_signature="test-family",
        operator_path=genome.operator_path,
        field_names=genome.field_names,
        field_families=genome.field_families,
        wrappers=genome.wrapper_gene.all_wrappers(),
        horizon_bucket=genome.horizon_bucket,
        turnover_bucket=genome.turnover_bucket,
        complexity_bucket=genome.complexity_bucket,
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("rank(ts_delta(close,5))", False),
        ("ts_mean(rank(close),5)", True),
        ("ts_delta(zscore(volume),3)", True),
        ("(ts_delta(close,5)+rank(volume))", False),
        ("ts_mean((rank(close)-close),5)", True),
    ],
)
def test_has_nesting_violation(expression: str, expected: bool) -> None:
    assert has_nesting_violation(parse_expression(expression)) is expected


def test_grammar_keeps_safe_rank_wrapper_outside_time_series_signal() -> None:
    render = MotifGrammar().render(_build_genome(post_wrappers=("rank",)))

    assert render.expression == "rank(ts_delta(close,10))"
    assert has_nesting_violation(parse_expression(render.expression)) is False


def test_grammar_keeps_safe_rank_wrapper_after_smoothing() -> None:
    render = MotifGrammar().render(
        _build_genome(
            post_wrappers=("rank",),
            smoothing_operator="ts_decay_linear",
            smoothing_window=5,
        )
    )

    assert render.expression == "rank(ts_decay_linear(ts_delta(close,10),5))"
    assert has_nesting_violation(parse_expression(render.expression)) is False


def test_engine_rejects_invalid_nesting_before_full_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_generation_config()
    engine = AlphaGenerationEngine(
        config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(config.allowed_operators),
        field_registry=_build_field_registry(),
    )

    def fail_validation(**_: object) -> ValidationResult:
        raise AssertionError("validate_expression should not be called for invalid nesting")

    monkeypatch.setattr("generator.engine.validate_expression", fail_validation)

    result = engine._build_candidate_result("ts_mean(rank(close),5)", mode="test", parent_ids=())  # noqa: SLF001

    assert result.candidate is None
    assert result.failure_reason == "validation_invalid_nesting"


def test_repair_policy_strips_cross_sectional_wrappers_when_nesting_can_be_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    repair = RepairPolicy(
        generation_config=_build_generation_config(),
        repair_config=RepairPolicyConfig(enabled=True),
        field_registry=_build_field_registry(),
        registry=build_default_registry(),
    )
    genome = _build_genome(pre_wrappers=("rank",), post_wrappers=("zscore",))

    def fake_render(candidate: Genome) -> GenomeRenderResult:
        if candidate.wrapper_gene.pre_wrappers or candidate.wrapper_gene.post_wrappers:
            return _build_render_result(candidate, "ts_mean(rank(close),5)")
        return _build_render_result(candidate, "rank(ts_mean(close,5))")

    monkeypatch.setattr(repair._grammar, "render", fake_render)

    repaired, actions = repair.repair(genome)

    assert repaired.wrapper_gene.pre_wrappers == ()
    assert repaired.wrapper_gene.post_wrappers == ()
    assert "nesting_repair" in actions
    assert "nesting_repair_failed" not in actions


def test_repair_policy_reports_failure_and_returns_original_genome(monkeypatch: pytest.MonkeyPatch) -> None:
    repair = RepairPolicy(
        generation_config=_build_generation_config(),
        repair_config=RepairPolicyConfig(enabled=True),
        field_registry=_build_field_registry(),
        registry=build_default_registry(),
    )
    genome = _build_genome(pre_wrappers=("rank",))

    def always_invalid(candidate: Genome) -> GenomeRenderResult:
        return _build_render_result(candidate, "ts_mean(rank(close),5)")

    monkeypatch.setattr(repair._grammar, "render", always_invalid)

    repaired, actions = repair.repair(genome)

    assert repaired == genome
    assert "nesting_repair_failed" in actions
    assert "nesting_repair" not in actions
