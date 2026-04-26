from __future__ import annotations

import json

import pytest

from core.config import AdaptiveGenerationConfig, EliteMotifConfig, QualityOptimizationConfig, load_config
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import build_registry
from generator.engine import AlphaCandidate, GenerationSessionStats
from generator.guardrails import GenerationGuardrails
from memory.pattern_memory import RegionLearningContext
from services.brain_batch_service import BrainBatchService
from services.models import CommandEnvironment
from services.quality_polisher import QualityPolisher, quality_parent_score
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


def _seed_parent(repository: SQLiteRepository, *, run_id: str, alpha_id: str, expression: str) -> None:
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
                fields_used=("close",),
                operators_used=("rank", "ts_mean"),
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
    return FieldRegistry(
        fields={
            "close": FieldSpec(
                name="close",
                dataset="test",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="price",
                runtime_available=True,
                category_weight=1.0,
                field_score=1.0,
            ),
            "volume": FieldSpec(
                name="volume",
                dataset="test",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="volume",
                runtime_available=True,
                category_weight=1.0,
                field_score=1.0,
            ),
        }
    )
