from __future__ import annotations

import json

import pytest

from alpha.ast_nodes import FunctionCallNode, NumberNode, node_depth
from alpha.parser import parse_expression
from alpha.validator import validate_expression
from core.config import AdaptiveGenerationConfig, EliteMotifConfig, QualityOptimizationConfig, load_config
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import build_registry
from generator.engine import AlphaCandidate, GenerationSessionStats
from generator.guardrails import GenerationGuardrails
from memory.pattern_memory import RegionLearningContext
from services.brain_batch_service import BrainBatchService
from services.models import CommandEnvironment
from services.quality_polisher import QualityPolishStats, QualityPolisher, quality_parent_score
from storage.models import BrainResultRecord, StageMetricRecord, SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository


def test_quality_parent_score_rewards_quality_and_penalizes_risk() -> None:
    strong = quality_parent_score(
        {"fitness": 0.40, "sharpe": 0.50, "returns": 0.08, "turnover": 0.30, "drawdown": 0.10}
    )
    weak = quality_parent_score(
        {"fitness": 0.05, "sharpe": 0.08, "returns": 0.01, "turnover": 1.40, "drawdown": 0.70}
    )

    assert strong > weak


def test_operator_substitution_changes_operator_not_args() -> None:
    root = parse_expression("ts_mean(anl69_eps_best_eeps_nxt_yr,20)")
    variants = QualityPolisher(None)._operator_substitution_variants(root, registry=_quality_registry())
    expressions = {variant.expression for variant in variants}

    assert "ts_rank(anl69_eps_best_eeps_nxt_yr,20)" in expressions
    assert "ts_decay_linear(anl69_eps_best_eeps_nxt_yr,20)" in expressions
    assert "ts_mean(anl69_eps_best_eeps_nxt_yr,10)" not in expressions
    for expression in expressions:
        parsed = parse_expression(expression)
        assert isinstance(parsed, FunctionCallNode)
        assert parsed.name != "ts_mean"
        assert parsed.args[0].name == "anl69_eps_best_eeps_nxt_yr"
        assert isinstance(parsed.args[1], NumberNode)
        assert int(parsed.args[1].value) == 20


def test_neutralization_add_wraps_correctly() -> None:
    root = parse_expression("ts_mean(anl69_eps_best_eeps_nxt_yr,20)")
    variants = QualityPolisher(None)._neutralization_variants(
        root,
        registry=_quality_registry(),
        field_registry=_field_registry(),
        generation_config=_quality_generation_config(),
    )

    assert variants
    assert variants[0].expression == "rank(group_neutralize(ts_mean(anl69_eps_best_eeps_nxt_yr,20),subindustry))"


def test_neutralization_remove_unwraps_correctly() -> None:
    root = parse_expression("rank(group_neutralize(ts_mean(anl69_eps_best_eeps_nxt_yr,20),subindustry))")
    variants = QualityPolisher(None)._neutralization_variants(
        root,
        registry=_quality_registry(),
        field_registry=_field_registry(),
        generation_config=_quality_generation_config(),
    )

    assert any(
        variant.transform == "neutralize_remove"
        and variant.expression == "rank(ts_mean(anl69_eps_best_eeps_nxt_yr,20))"
        for variant in variants
    )


def test_composite_structure_respects_depth_limit() -> None:
    root = parse_expression("ts_mean(anl69_eps_best_eeps_nxt_yr,20)")
    variants = QualityPolisher(None)._composite_structure_variants(
        root,
        registry=_quality_registry(),
        lookbacks=[10, 20, 60],
        max_depth=4,
    )

    assert variants
    assert all(node_depth(parse_expression(variant.expression)) <= 4 for variant in variants)


def test_field_substitution_keeps_structure() -> None:
    root = parse_expression("ts_mean(anl69_eps_best_eeps_nxt_yr,20)")
    variants = QualityPolisher(None)._field_substitution_variants(
        root,
        field_registry=_field_registry(),
        allowed_fields=set(_field_registry().fields),
    )

    expressions = {variant.expression for variant in variants}
    assert "ts_mean(anl69_epss_best_eeps_nxt_yr,20)" in expressions
    assert all(expression.startswith("ts_mean(") and expression.endswith(",20)") for expression in expressions)


def test_structural_variants_pass_validator() -> None:
    polisher = QualityPolisher(None)
    root = parse_expression("ts_mean(anl69_eps_best_eeps_nxt_yr,20)")
    registry = _quality_registry()
    field_registry = _field_registry()
    generation_config = _quality_generation_config()
    structural_variants = [
        polisher._operator_substitution_variants(root, registry=registry)[0],
        polisher._neutralization_variants(
            root,
            registry=registry,
            field_registry=field_registry,
            generation_config=generation_config,
        )[0],
        polisher._cross_sectional_wrapper_variants(root, registry=registry)[0],
        polisher._composite_structure_variants(root, registry=registry, lookbacks=[10, 20, 60], max_depth=7)[0],
        polisher._field_substitution_variants(
            root,
            field_registry=field_registry,
            allowed_fields=set(field_registry.fields),
        )[0],
    ]

    for variant in structural_variants:
        _assert_valid_expression(variant.expression, registry=registry, field_registry=field_registry, max_depth=7)


def test_budget_allocation_respects_percentages() -> None:
    variants = _structural_variant_batch(limit=20)
    counts = _budget_group_counts(variants)
    expected = {
        "surface": 6,
        "operator_substitution": 4,
        "neutralization": 3,
        "cross_section": 3,
        "composite": 2,
        "field_substitution": 2,
    }

    assert len(variants) == 20
    for group, target in expected.items():
        assert abs(counts.get(group, 0) - target) <= 2


def test_no_surface_only_variants_dominate() -> None:
    variants = _structural_variant_batch(limit=20)
    counts = _budget_group_counts(variants)

    assert counts["surface"] <= 6
    assert sum(count for group, count in counts.items() if group != "surface") >= 14


def test_duplicate_variants_skipped() -> None:
    repository = SQLiteRepository(":memory:")
    duplicate_expression = "ts_decay_linear(anl69_eps_best_eeps_nxt_yr,20)"
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            expression="ts_mean(anl69_eps_best_eeps_nxt_yr,20)",
            fields_used=("anl69_eps_best_eeps_nxt_yr",),
            operators_used=("ts_mean",),
        )
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-1",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )
        repository.save_alpha_candidates(
            "run-quality",
            [
                AlphaCandidate(
                    alpha_id="existing-duplicate",
                    expression=duplicate_expression,
                    normalized_expression=duplicate_expression,
                    generation_mode="quality_polish",
                    parent_ids=("parent-1",),
                    complexity=4,
                    created_at="2026-04-22T01:30:00+00:00",
                    fields_used=("anl69_eps_best_eeps_nxt_yr",),
                    operators_used=("ts_decay_linear",),
                    depth=3,
                    generation_metadata={},
                )
            ],
        )
        generation_config = _quality_generation_config()
        result = QualityPolisher(repository).generate(
            config=_structural_quality_config(limit=12),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=_quality_registry(),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-quality",
            round_index=2,
            count=12,
            allowed_fields=set(_field_registry().fields),
            lane_operator_allowlist=set(generation_config.allowed_operators),
        )
    finally:
        repository.close()

    assert duplicate_expression not in {candidate.normalized_expression for candidate in result.candidates}
    assert result.stats.skipped_existing_normalized >= 1


def test_repository_lists_latest_completed_quality_polish_parent() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-1", expression="rank(ts_mean(close,2))")
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-old",
            fitness=0.10,
            sharpe=0.10,
            simulated_at="2026-04-22T00:00:00+00:00",
        )
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-new",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )

        rows = repository.list_quality_polish_parent_rows(run_id="run-quality", limit=10)
    finally:
        repository.close()

    assert len(rows) == 1
    assert rows[0]["alpha_id"] == "parent-1"
    assert rows[0]["result_job_id"] == "job-new"
    assert rows[0]["result_fitness"] == pytest.approx(0.30)


def test_repository_lists_quality_polish_usage_rows_with_legacy_fallback() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        repository.save_alpha_candidates(
            "run-quality",
            [
                AlphaCandidate(
                    alpha_id="legacy-polish",
                    expression="zscore(ts_mean(close,3))",
                    normalized_expression="zscore(ts_mean(close,3))",
                    generation_mode="quality_polish",
                    parent_ids=("parent-1",),
                    complexity=5,
                    created_at="2026-04-22T01:30:00+00:00",
                    fields_used=("close",),
                    operators_used=("zscore", "ts_mean"),
                    depth=4,
                    generation_metadata={
                        "polish_parent_alpha_id": "parent-1",
                        "polish_transform": "wrap_zscore",
                    },
                )
            ],
        )

        usage = repository.list_quality_polish_usage_keys("run-quality")
    finally:
        repository.close()

    assert len(usage["usage_rows"]) == 1
    row = usage["usage_rows"][0]
    assert row["polish_signature"]
    assert row["polish_parent_transform_key"] == "parent-1:wrap_zscore"
    assert row["polish_parent_alpha_id"] == "parent-1"
    assert row["polish_transform"] == "wrap_zscore"
    assert row["polish_transform_group"] == "wrap_zscore"
    assert row["normalized_expression"] == "zscore(ts_mean(close,3))"
    assert row["polish_round_index"] is None
    assert row["created_at"] == "2026-04-22T01:30:00+00:00"


def test_quality_polisher_generates_valid_non_duplicate_variants() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-1", expression="rank(ts_mean(close,2))")
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-1",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )

        generation_config = load_config("config/dev.yaml").generation
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(min_completed_parent_count=1, max_polish_candidates_per_round=4),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized={"rank(ts_mean(close,2))"},
            run_id="run-quality",
            round_index=2,
            count=4,
        )
    finally:
        repository.close()

    assert result.candidates
    assert result.stats.generated_count == len(result.candidates)
    assert result.stats.attempt_count >= result.stats.success_count
    assert all(candidate.generation_mode == "quality_polish" for candidate in result.candidates)
    assert all(candidate.generation_metadata["mutation_mode"] == "quality_polish" for candidate in result.candidates)
    assert all(candidate.generation_metadata["polish_parent_alpha_id"] == "parent-1" for candidate in result.candidates)
    assert all(candidate.generation_metadata["polish_signature"] for candidate in result.candidates)
    assert all(candidate.generation_metadata["polish_parent_transform_key"] for candidate in result.candidates)
    assert not any(candidate.normalized_expression == "rank(ts_mean(close,2))" for candidate in result.candidates)
    assert not any(candidate.generation_metadata["polish_transform"].startswith("sign") for candidate in result.candidates)
    assert not any("group_neutralize" in candidate.generation_metadata["polish_transform"] for candidate in result.candidates)
    assert "wrap_rank" not in result.stats.transform_counts
    assert not any(transform.startswith("smooth_") for transform in result.stats.transform_attempt_counts)
    assert result.stats.disabled_transform_counts["smooth_ts_mean"] >= 1


def test_quality_polisher_emits_turnover_repair_variants_for_high_turnover_parent() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-1", expression="rank(close)")
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-high-turnover",
            fitness=0.25,
            sharpe=0.30,
            simulated_at="2026-04-22T01:00:00+00:00",
            turnover=0.95,
        )

        generation_config = load_config("config/dev.yaml").generation
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(min_completed_parent_count=1, max_polish_candidates_per_round=4),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized={"rank(close)"},
            run_id="run-quality",
            round_index=2,
            count=4,
        )
    finally:
        repository.close()

    turnover_candidates = [
        candidate for candidate in result.candidates if candidate.generation_metadata.get("repair_reason") == "turnover"
    ]
    assert turnover_candidates
    assert result.stats.turnover_repair_generated >= 1
    assert result.stats.turnover_repair_attempt_count >= result.stats.turnover_repair_success_count >= 1
    assert result.stats.turnover_repair_transform_counts
    assert all(candidate.generation_metadata["repair_target_profile"] == "low_turnover" for candidate in turnover_candidates)


def test_quality_polisher_soft_skips_parent_transform_used_in_recent_round() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-1", expression="ts_mean(close,2)")
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-1",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )
        repository.save_alpha_candidates(
            "run-quality",
            [
                AlphaCandidate(
                    alpha_id="old-polish",
                    expression="zscore(ts_mean(close,3))",
                    normalized_expression="zscore(ts_mean(close,3))",
                    generation_mode="quality_polish",
                    parent_ids=("parent-1",),
                    complexity=5,
                    created_at="2026-04-22T01:30:00+00:00",
                    fields_used=("close",),
                    operators_used=("zscore", "rank", "ts_mean"),
                    depth=4,
                    generation_metadata={
                        "polish_parent_alpha_id": "parent-1",
                        "polish_transform": "wrap_zscore",
                        "polish_parent_transform_key": "parent-1:wrap_zscore",
                        "polish_round_index": 9,
                    },
                )
            ],
        )

        generation_config = load_config("config/dev.yaml").generation
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(min_completed_parent_count=1, max_polish_candidates_per_round=4),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-quality",
            round_index=10,
            count=4,
        )
    finally:
        repository.close()

    assert result.stats.skipped_used_parent_transform >= 1
    assert not any(
        candidate.generation_metadata["polish_transform"] == "wrap_zscore"
        for candidate in result.candidates
    )


def test_quality_polisher_allows_parent_transform_outside_recent_window() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-1", expression="ts_mean(close,2)")
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-1",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )
        repository.save_alpha_candidates(
            "run-quality",
            [
                AlphaCandidate(
                    alpha_id="old-polish",
                    expression="zscore(ts_mean(close,3))",
                    normalized_expression="zscore(ts_mean(close,3))",
                    generation_mode="quality_polish",
                    parent_ids=("parent-1",),
                    complexity=5,
                    created_at="2026-04-22T01:30:00+00:00",
                    fields_used=("close",),
                    operators_used=("zscore", "ts_mean"),
                    depth=4,
                    generation_metadata={
                        "polish_parent_alpha_id": "parent-1",
                        "polish_transform": "wrap_zscore",
                        "polish_parent_transform_key": "parent-1:wrap_zscore",
                        "polish_round_index": 1,
                    },
                )
            ],
        )

        generation_config = load_config("config/dev.yaml").generation
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(min_completed_parent_count=1, max_polish_candidates_per_round=4),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-quality",
            round_index=10,
            count=4,
        )
    finally:
        repository.close()

    assert result.stats.skipped_used_parent_transform == 0
    assert any(candidate.generation_metadata["polish_transform"] == "wrap_zscore" for candidate in result.candidates)


def test_quality_polisher_hard_skips_exact_signature_even_when_old() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-1", expression="ts_mean(close,2)")
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-1",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )
        repository.save_alpha_candidates(
            "run-quality",
            [
                AlphaCandidate(
                    alpha_id="old-polish",
                    expression="rank(ts_mean(close,2))",
                    normalized_expression="rank(ts_mean(close,2))",
                    generation_mode="quality_polish",
                    parent_ids=("parent-1",),
                    complexity=5,
                    created_at="2026-04-22T01:30:00+00:00",
                    fields_used=("close",),
                    operators_used=("rank", "ts_mean"),
                    depth=4,
                    generation_metadata={
                        "polish_parent_alpha_id": "parent-1",
                        "polish_transform": "wrap_rank",
                        "polish_round_index": 1,
                    },
                )
            ],
        )

        generation_config = load_config("config/dev.yaml").generation
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(min_completed_parent_count=1, max_polish_candidates_per_round=4),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-quality",
            round_index=10,
            count=4,
        )
    finally:
        repository.close()

    assert result.stats.skipped_used_signature >= 1
    assert not any(candidate.generation_metadata["polish_transform"] == "wrap_rank" for candidate in result.candidates)


def test_quality_polisher_legacy_usage_without_round_index_does_not_soft_lock_parent_transform() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-1", expression="ts_mean(close,2)")
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-1",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )
        repository.save_alpha_candidates(
            "run-quality",
            [
                AlphaCandidate(
                    alpha_id="legacy-polish",
                    expression="zscore(ts_mean(close,3))",
                    normalized_expression="zscore(ts_mean(close,3))",
                    generation_mode="quality_polish",
                    parent_ids=("parent-1",),
                    complexity=5,
                    created_at="2026-04-22T01:30:00+00:00",
                    fields_used=("close",),
                    operators_used=("zscore", "ts_mean"),
                    depth=4,
                    generation_metadata={
                        "polish_parent_alpha_id": "parent-1",
                        "polish_transform": "wrap_zscore",
                    },
                )
            ],
        )

        generation_config = load_config("config/dev.yaml").generation
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(min_completed_parent_count=1, max_polish_candidates_per_round=4),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-quality",
            round_index=10,
            count=4,
        )
    finally:
        repository.close()

    assert result.stats.skipped_used_parent_transform == 0
    assert any(candidate.generation_metadata["polish_transform"] == "wrap_zscore" for candidate in result.candidates)


def test_quality_polisher_cools_down_failing_transform_from_recent_metrics() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-1", expression="ts_mean(close,2)")
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-1",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )
        repository.save_stage_metrics(
            [
                StageMetricRecord(
                    run_id="run-quality",
                    round_index=9,
                    stage="generation",
                    metrics_json=json.dumps(
                        {
                            "quality_polish_transform_attempt_counts": {
                                "wrap_zscore": 3,
                                "window_perturb": 4,
                            },
                            "quality_polish_transform_counts": {
                                "wrap_zscore": 0,
                                "window_perturb": 4,
                            },
                        }
                    ),
                    created_at="2026-04-22T01:30:00+00:00",
                )
            ]
        )

        generation_config = load_config("config/dev.yaml").generation
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(min_completed_parent_count=1, max_polish_candidates_per_round=4),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-quality",
            round_index=10,
            count=4,
        )
    finally:
        repository.close()

    assert result.stats.transform_scores["wrap_zscore"] == pytest.approx(0.0)
    assert result.stats.transform_scores["window_perturb"] == pytest.approx(1.0)
    assert result.stats.transform_cooldown_counts["wrap_zscore"] >= 1
    assert not any(candidate.generation_metadata["polish_transform"] == "wrap_zscore" for candidate in result.candidates)


def test_quality_polisher_emits_smooth_variants_when_wrap_transforms_are_exhausted() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-1", expression="close")
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-1",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )
        repository.save_stage_metrics(
            [
                StageMetricRecord(
                    run_id="run-quality",
                    round_index=9,
                    stage="generation",
                    metrics_json=json.dumps(
                        {
                            "quality_polish_transform_attempt_counts": {
                                "wrap_rank": 3,
                                "wrap_zscore": 3,
                                "smooth_ts_mean": 3,
                                "smooth_ts_decay_linear": 3,
                            },
                            "quality_polish_transform_counts": {
                                "wrap_rank": 0,
                                "wrap_zscore": 0,
                                "smooth_ts_mean": 0,
                                "smooth_ts_decay_linear": 0,
                            },
                        }
                    ),
                    created_at="2026-04-22T01:30:00+00:00",
                )
            ]
        )

        generation_config = load_config("config/dev.yaml").generation
        generation_config.lookbacks = [2, 5, 10]
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(
                min_completed_parent_count=1,
                max_polish_candidates_per_round=6,
                enabled_transforms=[
                    "wrap_rank",
                    "wrap_zscore",
                    "smooth_ts_mean",
                    "smooth_ts_decay_linear",
                ],
                disabled_transforms=["cleanup_redundant_wrapper", "smooth_ts_rank"],
                cooldown_exempt_transform_groups=["smooth_ts_mean", "smooth_ts_decay_linear"],
                max_variants_per_parent_by_transform={
                    "wrap_rank": 1,
                    "wrap_zscore": 1,
                    "smooth_ts_mean": 2,
                    "smooth_ts_decay_linear": 2,
                },
            ),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized={"close"},
            run_id="run-quality",
            round_index=10,
            count=6,
        )
    finally:
        repository.close()

    smooth_groups = {
        candidate.generation_metadata["polish_transform_group"]
        for candidate in result.candidates
    }
    assert "smooth_ts_mean" in smooth_groups
    assert "smooth_ts_decay_linear" in smooth_groups
    assert any(
        str(candidate.generation_metadata["polish_transform"]).startswith("smooth_ts_decay_linear_zscore")
        for candidate in result.candidates
    )
    assert result.stats.transform_cooldown_counts["wrap_rank"] >= 1
    assert result.stats.transform_cooldown_counts["wrap_zscore"] >= 1
    assert result.stats.transform_cooldown_counts["smooth_ts_mean"] == 0
    assert result.stats.transform_cooldown_counts["smooth_ts_decay_linear"] == 0
    assert result.stats.cooldown_exempt_groups == ["smooth_ts_mean", "smooth_ts_decay_linear"]
    assert all(candidate.normalized_expression != "close" for candidate in result.candidates)
    assert "redundant_expression" not in result.stats.failure_counts


def test_quality_polisher_scans_past_saturated_top_parent() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-1", expression="close")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-2", expression="volume")
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-1",
            fitness=0.60,
            sharpe=0.70,
            simulated_at="2026-04-22T01:00:00+00:00",
        )
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-2",
            job_id="job-2",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:01:00+00:00",
        )
        repository.save_alpha_candidates(
            "run-quality",
            [
                AlphaCandidate(
                    alpha_id="used-rank",
                    expression="rank(close)",
                    normalized_expression="rank(close)",
                    generation_mode="quality_polish",
                    parent_ids=("parent-1",),
                    complexity=2,
                    created_at="2026-04-22T01:30:00+00:00",
                    fields_used=("close",),
                    operators_used=("rank",),
                    depth=2,
                    generation_metadata={
                        "polish_parent_alpha_id": "parent-1",
                        "polish_transform": "wrap_rank",
                        "polish_round_index": 8,
                    },
                ),
                AlphaCandidate(
                    alpha_id="used-zscore",
                    expression="zscore(close)",
                    normalized_expression="zscore(close)",
                    generation_mode="quality_polish",
                    parent_ids=("parent-1",),
                    complexity=2,
                    created_at="2026-04-22T01:31:00+00:00",
                    fields_used=("close",),
                    operators_used=("zscore",),
                    depth=2,
                    generation_metadata={
                        "polish_parent_alpha_id": "parent-1",
                        "polish_transform": "wrap_zscore",
                        "polish_round_index": 8,
                    },
                ),
            ],
        )

        generation_config = load_config("config/dev.yaml").generation
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(
                min_completed_parent_count=1,
                max_polish_candidates_per_round=2,
                max_polish_parents_per_round=1,
                parent_scan_multiplier=2,
                variants_per_parent=2,
                enabled_transforms=["wrap_rank", "wrap_zscore"],
                disabled_transforms=[
                    "cleanup_redundant_wrapper",
                    "smooth_ts_mean",
                    "smooth_ts_decay_linear",
                    "smooth_ts_rank",
                    "window_perturb",
                ],
                max_variants_per_parent_by_transform={"wrap_rank": 1, "wrap_zscore": 1},
            ),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-quality",
            round_index=10,
            count=2,
        )
    finally:
        repository.close()

    assert result.stats.scanned_parent_count == 2
    assert result.stats.saturated_parent_count >= 1
    assert result.candidates
    assert all("volume" in candidate.expression for candidate in result.candidates)


def test_quality_polisher_uses_external_elite_seeds_without_db_parents() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        generation_config = load_config("config/dev.yaml").generation
        generation_config.allowed_operators = [
            "rank",
            "zscore",
            "quantile",
            "ts_mean",
            "ts_scale",
            "ts_std_dev",
        ]
        generation_config.allowed_fields = ["close", "volume"]
        generation_config.lookbacks = [2, 3]
        generation_config.max_depth = 8
        generation_config.complexity_limit = 24
        seed = "(rank(ts_mean(close,125))+rank(ts_scale(volume,145)))"
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(
                min_completed_parent_count=5,
                max_polish_candidates_per_round=6,
                enabled_transforms=["wrap_rank", "wrap_zscore"],
            ),
            adaptive_config=AdaptiveGenerationConfig(
                elite_motifs=EliteMotifConfig(
                    enabled=True,
                    lookbacks=[125, 145, 150],
                    seed_expressions=[seed],
                    max_quality_polish_seeds_per_round=1,
                    max_seed_variants_per_seed=6,
                )
            ),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized={seed},
            run_id="run-quality",
            round_index=10,
            count=6,
        )
    finally:
        repository.close()

    assert result.candidates
    assert result.stats.external_elite_seed_count == 1
    assert result.stats.external_elite_generated == len(result.candidates)
    assert all(candidate.generation_metadata["elite_seed_id"] == "elite_seed_1" for candidate in result.candidates)
    assert all(candidate.generation_metadata["elite_seed_variant"] for candidate in result.candidates)
    assert all(candidate.normalized_expression != seed for candidate in result.candidates)
    assert any("150" in candidate.expression or "125" in candidate.expression for candidate in result.candidates)
    assert any("zscore(" in candidate.expression or "quantile(" in candidate.expression for candidate in result.candidates)
    assert "redundant_expression" not in result.stats.failure_counts


def test_quality_polisher_drops_parent_fields_outside_active_registry() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(
            repository,
            run_id="run-quality",
            alpha_id="parent-stale",
            expression="rank(ts_mean(stale_field,2))",
            fields_used=("stale_field",),
        )
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-stale",
            job_id="job-stale",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )

        generation_config = load_config("config/dev.yaml").generation
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(min_completed_parent_count=1, max_polish_candidates_per_round=4),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-quality",
            round_index=2,
            count=4,
            allowed_fields={"close", "volume"},
            lane_operator_allowlist=None,
        )
    finally:
        repository.close()

    assert result.candidates == []
    assert result.stats.parent_count == 1
    assert result.stats.eligible_parent_count == 0


def test_quality_polisher_lane_operator_filter_ignores_structural_and_stale_operators() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(
            repository,
            run_id="run-quality",
            alpha_id="parent-stale-ops",
            expression="ts_mean(close,2)",
            operators_used=("ts_mean", "binary:*", "ts_sum"),
        )
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-stale-ops",
            job_id="job-stale-ops",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )

        generation_config = load_config("config/dev.yaml").generation
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(
                min_completed_parent_count=1,
                max_polish_candidates_per_round=2,
                enabled_transforms=["wrap_rank"],
                disabled_transforms=[
                    "cleanup_redundant_wrapper",
                    "smooth_ts_mean",
                    "smooth_ts_decay_linear",
                    "smooth_ts_rank",
                    "window_perturb",
                    "wrap_zscore",
                ],
            ),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-quality",
            round_index=2,
            count=2,
            allowed_fields={"close", "volume"},
            lane_operator_allowlist={"rank", "zscore", "ts_mean"},
        )
    finally:
        repository.close()

    assert result.stats.eligible_parent_count == 1
    assert result.candidates
    assert all(candidate.generation_metadata["polish_parent_alpha_id"] == "parent-stale-ops" for candidate in result.candidates)


def test_quality_polisher_smooths_safe_subterms_when_parent_contains_cross_sectional_nesting() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(
            repository,
            run_id="run-quality",
            alpha_id="parent-cross-sectional",
            expression="rank((zscore(close)+ts_mean(volume,2)))",
            fields_used=("close", "volume"),
            operators_used=("rank", "zscore", "ts_mean", "binary:+"),
        )
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-cross-sectional",
            job_id="job-cross-sectional",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )

        generation_config = load_config("config/dev.yaml").generation
        generation_config.lookbacks = [2, 5, 10]
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(
                min_completed_parent_count=1,
                max_polish_candidates_per_round=4,
                variants_per_parent=4,
                enabled_transforms=["smooth_ts_mean", "smooth_ts_decay_linear"],
                disabled_transforms=[
                    "cleanup_redundant_wrapper",
                    "smooth_ts_rank",
                    "window_perturb",
                    "wrap_rank",
                    "wrap_zscore",
                ],
                cooldown_exempt_transform_groups=["smooth_ts_mean", "smooth_ts_decay_linear"],
                max_variants_per_parent_by_transform={
                    "smooth_ts_mean": 2,
                    "smooth_ts_decay_linear": 2,
                },
            ),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-quality",
            round_index=2,
            count=4,
            lane_operator_allowlist={"rank", "zscore", "ts_mean", "ts_decay_linear"},
        )
    finally:
        repository.close()

    assert result.candidates
    assert all("zscore(close)+ts_mean(volume,2)" not in candidate.expression for candidate in result.candidates)
    assert result.stats.failure_counts["validation_invalid_nesting"] == 0


def test_quality_polisher_skips_external_elite_seed_outside_lane_operator_profile() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        generation_config = load_config("config/dev.yaml").generation
        generation_config.allowed_operators = ["rank", "zscore", "quantile", "ts_arg_max"]
        adaptive_config = AdaptiveGenerationConfig(
            elite_motifs=EliteMotifConfig(
                enabled=True,
                seed_expressions=["rank(ts_arg_max(close,125))"],
                lookbacks=[125, 145, 150],
                max_quality_polish_seeds_per_round=1,
                max_seed_variants_per_seed=4,
            )
        )

        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(min_completed_parent_count=5, max_polish_candidates_per_round=4),
            adaptive_config=adaptive_config,
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-quality",
            round_index=2,
            count=4,
            allowed_fields={"close", "volume"},
            lane_operator_allowlist={"rank"},
        )
    finally:
        repository.close()

    assert result.candidates == []
    assert result.stats.external_elite_seed_count == 1
    assert result.stats.external_elite_generated == 0
    assert result.stats.search_space_filter_blocked > 0
    assert result.stats.failure_counts["search_space_filter_blocked"] > 0


def test_quality_polisher_reserves_external_elite_slot_when_parents_fill_quota() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-1", expression="rank(ts_mean(close,2))")
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-1",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )

        generation_config = load_config("config/dev.yaml").generation
        generation_config.allowed_operators = [
            "rank",
            "zscore",
            "quantile",
            "ts_mean",
            "ts_scale",
            "ts_std_dev",
        ]
        generation_config.allowed_fields = ["close", "volume"]
        generation_config.lookbacks = [2, 3]
        generation_config.max_depth = 8
        generation_config.complexity_limit = 24
        seed = "(rank(ts_mean(close,125))+rank(ts_scale(volume,145)))"
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(
                min_completed_parent_count=1,
                max_polish_candidates_per_round=4,
                max_polish_parents_per_round=1,
                variants_per_parent=8,
                enabled_transforms=["wrap_rank", "wrap_zscore", "window_perturb"],
            ),
            adaptive_config=AdaptiveGenerationConfig(
                elite_motifs=EliteMotifConfig(
                    enabled=True,
                    lookbacks=[125, 145, 150],
                    seed_expressions=[seed],
                    max_quality_polish_seeds_per_round=1,
                    max_seed_variants_per_seed=6,
                )
            ),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized={seed},
            run_id="run-quality",
            round_index=10,
            count=4,
        )
    finally:
        repository.close()

    assert result.stats.external_elite_seed_count == 1
    assert result.stats.external_elite_generated >= 1
    assert len(result.candidates) <= 4
    assert any(candidate.generation_metadata.get("elite_seed_id") == "elite_seed_1" for candidate in result.candidates)


def test_quality_polisher_can_emit_four_window_perturb_variants() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, run_id="run-quality")
        _seed_parent(repository, run_id="run-quality", alpha_id="parent-1", expression="ts_mean(close,2)")
        _seed_result(
            repository,
            run_id="run-quality",
            alpha_id="parent-1",
            job_id="job-1",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )

        generation_config = load_config("config/dev.yaml").generation
        generation_config.lookbacks = [2, 3, 5, 10, 20]
        result = QualityPolisher(repository).generate(
            config=QualityOptimizationConfig(min_completed_parent_count=1, max_polish_candidates_per_round=8),
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(
                region="USA",
                regime_key="regime",
                global_regime_key="global",
            ),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-quality",
            round_index=2,
            count=8,
        )
    finally:
        repository.close()

    assert result.stats.transform_counts["window_perturb"] == 4
    assert sum(1 for candidate in result.candidates if candidate.generation_metadata["polish_transform_group"] == "window_perturb") == 4


def test_brain_batch_service_includes_quality_polish_metrics(tmp_path, monkeypatch) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        config.runtime.progress_log_dir = str(tmp_path / "progress")
        quality_config = config.adaptive_generation.quality_optimization
        quality_config.enabled = True
        quality_config.min_completed_parent_count = 1
        quality_config.max_polish_candidates_per_round = 4
        quality_config.polish_budget_fraction = 0.50
        environment = CommandEnvironment(
            config_path="config/dev.yaml",
            command_name="quality-polish-test",
            context=type("Context", (), {"run_id": "run-quality", "seed": 42})(),
        )
        _seed_run(repository, run_id=environment.context.run_id)
        _seed_parent(
            repository,
            run_id=environment.context.run_id,
            alpha_id="parent-1",
            expression="rank(ts_mean(close,2))",
        )
        _seed_result(
            repository,
            run_id=environment.context.run_id,
            alpha_id="parent-1",
            job_id="job-1",
            fitness=0.30,
            sharpe=0.40,
            simulated_at="2026-04-22T01:00:00+00:00",
        )
        service = BrainBatchService(repository)
        monkeypatch.setattr(service, "_generate_mutation_candidates", lambda **kwargs: ([], GenerationSessionStats()))
        monkeypatch.setattr(service, "_generate_fresh_candidates", lambda **kwargs: ([], GenerationSessionStats()))

        result = service.prepare_service_batch(
            config=config,
            environment=environment,
            count=4,
            mutation_parent_ids=None,
            round_index=2,
        )
        metrics = result.generation_stage_metrics
    finally:
        repository.close()

    assert any(candidate.generation_mode == "quality_polish" for candidate in result.candidates)
    assert metrics["quality_polish_enabled"] is True
    assert metrics["quality_polish_parent_count"] >= 1
    assert metrics["quality_polish_generated"] >= 1
    assert metrics["quality_polish_attempt_count"] >= metrics["quality_polish_success_count"]
    assert "quality_polish_transform_counts" in metrics
    assert "quality_polish_blocked_by_signature" in metrics
    assert "quality_polish_blocked_by_recent_parent_transform" in metrics
    assert "quality_polish_transform_cooldown_counts" in metrics


def _quality_generation_config():
    generation_config = load_config("config/brain_full.yaml").generation
    generation_config.lookbacks = [10, 20, 60, 100, 250]
    generation_config.max_depth = 7
    generation_config.complexity_limit = 30
    return generation_config


def _quality_registry():
    generation_config = _quality_generation_config()
    return build_registry(
        generation_config.allowed_operators,
        operator_catalog_paths=generation_config.operator_catalog_paths,
    )


def _structural_quality_config(*, limit: int = 20) -> QualityOptimizationConfig:
    return QualityOptimizationConfig(
        min_completed_parent_count=1,
        max_polish_candidates_per_round=limit,
        variants_per_parent=limit,
        enabled_transforms=[
            "wrap_rank",
            "wrap_zscore",
            "window_perturb",
            "smooth_ts_mean",
            "smooth_ts_decay_linear",
            "operator_substitution",
            "neutralization",
            "cross_section",
            "composite",
            "field_substitution",
        ],
        disabled_transforms=["cleanup_redundant_wrapper", "smooth_ts_rank"],
        max_variants_per_parent_by_transform={
            "wrap_rank": 1,
            "wrap_zscore": 1,
            "window_perturb": 4,
            "smooth_ts_mean": 4,
            "smooth_ts_decay_linear": 4,
            "operator_substitution": 4,
            "neutralization": 4,
            "cross_section": 4,
            "composite": 4,
            "field_substitution": 4,
        },
        window_perturb_neighbor_count=4,
    )


def _structural_variant_batch(*, limit: int) -> list:
    generation_config = _quality_generation_config()
    field_registry = _field_registry()
    return QualityPolisher(None)._variant_expressions(
        "ts_mean(anl69_eps_best_eeps_nxt_yr,20)",
        registry=_quality_registry(),
        field_registry=field_registry,
        generation_config=generation_config,
        lookbacks=generation_config.lookbacks,
        limit=limit,
        config=_structural_quality_config(limit=limit),
        stats=QualityPolishStats(),
        transform_scores={},
        cooldown_groups=set(),
        existing_normalized=set(),
        allowed_fields=set(field_registry.fields),
        lane_operator_allowlist=set(generation_config.allowed_operators),
    )


def _budget_group_counts(variants: list) -> dict[str, int]:
    counts: dict[str, int] = {
        "surface": 0,
        "operator_substitution": 0,
        "neutralization": 0,
        "cross_section": 0,
        "composite": 0,
        "field_substitution": 0,
    }
    for variant in variants:
        group = variant.transform_group
        if group in {"window_perturb", "smooth_ts_mean", "smooth_ts_decay_linear", "smooth_ts_rank"}:
            counts["surface"] += 1
        elif group in {"wrap_rank", "wrap_zscore", "cross_section"}:
            counts["cross_section"] += 1
        else:
            counts[group] = counts.get(group, 0) + 1
    return counts


def _assert_valid_expression(
    expression: str,
    *,
    registry,
    field_registry: FieldRegistry,
    max_depth: int,
) -> None:
    allowed_fields = set(field_registry.fields)
    result = validate_expression(
        parse_expression(expression),
        registry=registry,
        allowed_fields=allowed_fields,
        max_depth=max_depth,
        group_fields={"subindustry", "sector"},
        field_types=field_registry.field_types(allowed=allowed_fields),
        field_categories={name: spec.category for name, spec in field_registry.fields.items()},
        complexity_limit=30,
        exact_field_types=field_registry.exact_field_types(allowed=allowed_fields),
    )
    assert result.is_valid, result.errors


def _seed_run(repository: SQLiteRepository, *, run_id: str) -> None:
    repository.upsert_run(
        run_id=run_id,
        seed=42,
        config_path="config/dev.yaml",
        config_snapshot="{}",
        status="running",
        started_at="2026-04-22T00:00:00+00:00",
        profile_name="dev",
        selected_timeframe="1d",
        global_regime_key="global",
        region="USA",
        entry_command="test",
    )


def _seed_parent(
    repository: SQLiteRepository,
    *,
    run_id: str,
    alpha_id: str,
    expression: str,
    fields_used: tuple[str, ...] = ("close",),
    operators_used: tuple[str, ...] = ("rank", "ts_mean"),
) -> None:
    repository.save_alpha_candidates(
        run_id,
        [
            AlphaCandidate(
                alpha_id=alpha_id,
                expression=expression,
                normalized_expression=expression,
                generation_mode="guided_exploit",
                parent_ids=(),
                complexity=4,
                created_at="2026-04-22T00:00:00+00:00",
                template_name="seed",
                fields_used=fields_used,
                operators_used=operators_used,
                depth=3,
                generation_metadata={"family_signature": "seed-family"},
            )
        ],
    )


def _seed_result(
    repository: SQLiteRepository,
    *,
    run_id: str,
    alpha_id: str,
    job_id: str,
    fitness: float,
    sharpe: float,
    simulated_at: str,
    turnover: float = 0.30,
) -> None:
    batch_id = f"batch-{job_id}"
    repository.submissions.upsert_batch(
        SubmissionBatchRecord(
            batch_id=batch_id,
            run_id=run_id,
            round_index=1,
            backend="api",
            status="completed",
            candidate_count=1,
            sim_config_snapshot="{}",
            export_path=None,
            notes_json="{}",
            created_at=simulated_at,
            updated_at=simulated_at,
        )
    )
    repository.submissions.upsert_submissions(
        [
            SubmissionRecord(
                job_id=job_id,
                batch_id=batch_id,
                run_id=run_id,
                round_index=1,
                candidate_id=alpha_id,
                expression="rank(ts_mean(close,2))",
                backend="api",
                status="completed",
                sim_config_snapshot="{}",
                submitted_at="2026-04-22T00:00:00+00:00",
                updated_at=simulated_at,
                completed_at=simulated_at,
                export_path=None,
                raw_submission_json=json.dumps({"job_id": job_id}),
                error_message=None,
            )
        ]
    )
    repository.brain_results.save_results(
        [
            BrainResultRecord(
                job_id=job_id,
                run_id=run_id,
                round_index=1,
                batch_id=batch_id,
                candidate_id=alpha_id,
                expression="rank(ts_mean(close,2))",
                status="completed",
                region="USA",
                universe="TOP3000",
                delay=1,
                neutralization="SUBINDUSTRY",
                decay=5,
                sharpe=sharpe,
                fitness=fitness,
                turnover=turnover,
                drawdown=0.10,
                returns=0.08,
                margin=0.05,
                submission_eligible=True,
                rejection_reason=None,
                raw_result_json="{}",
                metric_source="external_brain",
                simulated_at=simulated_at,
                created_at=simulated_at,
            )
        ]
    )


def _field_registry() -> FieldRegistry:
    def matrix_field(name: str, *, category: str = "analyst", subcategory: str = "") -> FieldSpec:
        return FieldSpec(
            name=name,
            dataset="test",
            field_type="matrix",
            coverage=1.0,
            alpha_usage_count=0,
            category=category,
            runtime_available=True,
            category_weight=1.0,
            field_score=1.0,
            subcategory=subcategory,
        )

    def group_field(name: str) -> FieldSpec:
        return FieldSpec(
            name=name,
            dataset="test",
            field_type="vector",
            coverage=1.0,
            alpha_usage_count=0,
            category="group",
            runtime_available=True,
            category_weight=1.0,
            field_score=1.0,
        )

    return FieldRegistry(
        fields={
            "close": matrix_field("close", category="price"),
            "volume": matrix_field("volume", category="volume"),
            "anl69_eps_best_eeps_nxt_yr": matrix_field("anl69_eps_best_eeps_nxt_yr", subcategory="eps_next_year"),
            "anl69_epss_best_eeps_nxt_yr": matrix_field("anl69_epss_best_eeps_nxt_yr", subcategory="eps_next_year"),
            "anl69_eps_best_cur_fiscal_year_period": matrix_field(
                "anl69_eps_best_cur_fiscal_year_period",
                subcategory="eps_next_year",
            ),
            "subindustry": group_field("subindustry"),
            "sector": group_field("sector"),
        }
    )
