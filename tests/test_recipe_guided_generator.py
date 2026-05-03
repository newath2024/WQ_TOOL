from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from alpha.parser import parse_expression
from alpha.validator import validate_expression
from core.config import AdaptiveGenerationConfig, RecipeGenerationConfig, load_config
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import build_registry
from generator.engine import AlphaCandidate, AlphaGenerationEngine, CandidateBuildResult
from generator.guardrails import GenerationGuardrails
from memory.pattern_memory import RegionLearningContext
from services.recipe_guided_generator import RecipeGuidedGenerator, _active_buckets
from storage.models import BrainResultRecord, StageMetricRecord, SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository


def test_recipe_bucket_scheduler_rotates_four_buckets_per_round() -> None:
    config = RecipeGenerationConfig()

    round_0 = [bucket.search_bucket_id for bucket in _active_buckets(config, round_index=0)]
    round_1 = [bucket.search_bucket_id for bucket in _active_buckets(config, round_index=1)]
    round_2 = [bucket.search_bucket_id for bucket in _active_buckets(config, round_index=2)]

    assert len(round_0) == 4
    assert len(round_1) == 4
    assert len(round_2) == 4
    assert len(set(round_0 + round_1 + round_2)) == 12
    assert round_0[0] == "fundamental_quality|fundamental|balanced"
    assert round_1[0] == "accrual_vs_cashflow|fundamental|quality"
    assert round_2[0] == "value_vs_growth|fundamental|low_turnover"


def test_recipe_guided_generator_uses_parentless_low_turnover_smoothed_variants() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, "run-recipe")
        generation_config = replace(
            load_config("config/dev.yaml").generation,
            allowed_fields=["gross_margin", "operating_cash_flow", "eps_revision"],
        )
        recipe_config = RecipeGenerationConfig(
            enabled_recipe_families=["fundamental_quality"],
            objective_profiles=["low_turnover"],
            active_bucket_count=1,
            max_recipe_candidates_per_round=4,
            max_candidates_per_bucket=4,
        )

        result = RecipeGuidedGenerator(repository).generate(
            config=recipe_config,
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(region="USA", regime_key="local", global_regime_key="global"),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-recipe",
            round_index=1,
            count=4,
        )
    finally:
        repository.close()

    assert result.candidates
    assert all(candidate.generation_mode == "recipe_guided" for candidate in result.candidates)
    assert all("ts_mean(" in candidate.expression for candidate in result.candidates)
    assert all(not candidate.generation_metadata.get("recipe_parent_alpha_id") for candidate in result.candidates)
    assert result.stats.parentless_count == len(result.candidates)


def test_recipe_guided_generator_obeys_filtered_lane_field_pool() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, "run-recipe")
        generation_config = replace(
            load_config("config/dev.yaml").generation,
            allowed_fields=["eps_estimate"],
            allowed_operators=["rank", "days_from_last_change", "ts_av_diff", "ts_arg_max"],
        )
        recipe_config = RecipeGenerationConfig(
            enabled_recipe_families=["analyst_estimate_recency"],
            objective_profiles=["balanced"],
            active_bucket_count=1,
            max_recipe_candidates_per_round=4,
            max_candidates_per_bucket=4,
        )

        result = RecipeGuidedGenerator(repository).generate(
            config=recipe_config,
            adaptive_config=AdaptiveGenerationConfig(
                elite_motifs=replace(AdaptiveGenerationConfig().elite_motifs, lookbacks=[125, 145, 150])
            ),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(region="USA", regime_key="local", global_regime_key="global"),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-recipe",
            round_index=1,
            count=4,
        )
    finally:
        repository.close()

    assert result.candidates
    assert all(candidate.fields_used == ("eps_estimate",) for candidate in result.candidates)
    assert all("eps_estimate" in candidate.expression for candidate in result.candidates)
    assert all("sales_estimate" not in candidate.expression for candidate in result.candidates)


def test_recipe_guided_generator_can_use_recent_completed_parent() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, "run-recipe")
        repository.save_alpha_candidates(
            "run-recipe",
            [
                AlphaCandidate(
                    alpha_id="parent-1",
                    expression="rank(ts_mean(gross_margin,20))",
                    normalized_expression="rank(ts_mean(gross_margin,20))",
                    generation_mode="guided_explore",
                    parent_ids=(),
                    complexity=4,
                    created_at="2026-04-23T00:00:00+00:00",
                    template_name="quality_score",
                    fields_used=("gross_margin",),
                    operators_used=("rank", "ts_mean"),
                    depth=3,
                    generation_metadata={"family_signature": "family-parent"},
                )
            ],
        )
        _seed_submission(repository, run_id="run-recipe", batch_id="batch-1", job_id="job-parent", alpha_id="parent-1")
        repository.brain_results.save_results(
            [
                BrainResultRecord(
                    job_id="job-parent",
                    run_id="run-recipe",
                    round_index=1,
                    batch_id="batch-1",
                    candidate_id="parent-1",
                    expression="rank(ts_mean(gross_margin,20))",
                    status="completed",
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                    neutralization="SECTOR",
                    decay=0,
                    sharpe=0.20,
                    fitness=0.10,
                    turnover=0.40,
                    drawdown=0.20,
                    returns=0.03,
                    margin=0.02,
                    submission_eligible=True,
                    rejection_reason=None,
                    raw_result_json="{}",
                    metric_source="external_brain",
                    simulated_at="2026-04-23T00:10:00+00:00",
                    created_at="2026-04-23T00:10:00+00:00",
                )
            ]
        )
        generation_config = replace(
            load_config("config/dev.yaml").generation,
            allowed_fields=["gross_margin", "operating_cash_flow", "eps_revision"],
        )
        recipe_config = RecipeGenerationConfig(
            enabled_recipe_families=["fundamental_quality"],
            objective_profiles=["quality"],
            active_bucket_count=1,
            max_recipe_candidates_per_round=4,
            max_candidates_per_bucket=4,
        )

        result = RecipeGuidedGenerator(repository).generate(
            config=recipe_config,
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(region="USA", regime_key="local", global_regime_key="global"),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-recipe",
            round_index=2,
            count=4,
        )
    finally:
        repository.close()

    assert result.candidates
    assert any(candidate.generation_metadata.get("recipe_parent_alpha_id") == "parent-1" for candidate in result.candidates)
    assert result.stats.parented_count >= 1


def test_recipe_guided_generator_applies_positive_bucket_prior_from_recent_recipe_results() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, "run-recipe")
        repository.save_alpha_candidates(
            "run-recipe",
            [
                AlphaCandidate(
                    alpha_id="recipe-old",
                    expression="rank(gross_margin)",
                    normalized_expression="rank(gross_margin)",
                    generation_mode="recipe_guided",
                    parent_ids=(),
                    complexity=2,
                    created_at="2026-04-22T00:00:00+00:00",
                    template_name="recipe_fundamental_quality",
                    fields_used=("gross_margin",),
                    operators_used=("rank",),
                    depth=2,
                    generation_metadata={
                        "search_bucket_id": "fundamental_quality|fundamental|balanced",
                        "recipe_family": "fundamental_quality",
                        "objective_profile": "balanced",
                    },
                )
            ],
        )
        _seed_submission(repository, run_id="run-recipe", batch_id="batch-1", job_id="job-old", alpha_id="recipe-old")
        repository.brain_results.save_results(
            [
                BrainResultRecord(
                    job_id="job-old",
                    run_id="run-recipe",
                    round_index=1,
                    batch_id="batch-1",
                    candidate_id="recipe-old",
                    expression="rank(gross_margin)",
                    status="completed",
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                    neutralization="SECTOR",
                    decay=0,
                    sharpe=0.30,
                    fitness=0.20,
                    turnover=0.30,
                    drawdown=0.10,
                    returns=0.04,
                    margin=0.03,
                    submission_eligible=True,
                    rejection_reason=None,
                    raw_result_json="{}",
                    metric_source="external_brain",
                    simulated_at="2026-04-22T00:10:00+00:00",
                    created_at="2026-04-22T00:10:00+00:00",
                )
            ]
        )
        generation_config = replace(
            load_config("config/dev.yaml").generation,
            allowed_fields=["gross_margin", "operating_cash_flow", "eps_revision"],
        )
        recipe_config = RecipeGenerationConfig(
            enabled_recipe_families=["fundamental_quality"],
            objective_profiles=["balanced"],
            active_bucket_count=1,
            max_recipe_candidates_per_round=2,
            max_candidates_per_bucket=2,
            yield_lookback_rounds=5,
            min_bucket_support_for_penalty=1,
            selection_prior_weight=0.08,
        )

        result = RecipeGuidedGenerator(repository).generate(
            config=recipe_config,
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(region="USA", regime_key="local", global_regime_key="global"),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized={"rank(gross_margin)"},
            run_id="run-recipe",
            round_index=3,
            count=2,
        )
    finally:
        repository.close()

    assert result.candidates
    assert result.candidates[0].generation_metadata["recipe_bucket_prior"] == pytest.approx(0.096)


def test_recipe_guided_generator_applies_dynamic_bucket_budget_with_floor() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, "run-recipe")
        repository.save_alpha_candidates(
            "run-recipe",
            [
                AlphaCandidate(
                    alpha_id="recipe-hot",
                    expression="rank(gross_margin)",
                    normalized_expression="rank(gross_margin)",
                    generation_mode="recipe_guided",
                    parent_ids=(),
                    complexity=2,
                    created_at="2026-04-22T00:00:00+00:00",
                    template_name="recipe_fundamental_quality",
                    fields_used=("gross_margin",),
                    operators_used=("rank",),
                    depth=2,
                    generation_metadata={
                        "search_bucket_id": "fundamental_quality|fundamental|balanced",
                        "recipe_family": "fundamental_quality",
                        "objective_profile": "balanced",
                    },
                ),
                AlphaCandidate(
                    alpha_id="recipe-cold",
                    expression="rank(accrual_ratio)",
                    normalized_expression="rank(accrual_ratio)",
                    generation_mode="recipe_guided",
                    parent_ids=(),
                    complexity=2,
                    created_at="2026-04-22T00:01:00+00:00",
                    template_name="recipe_accrual_vs_cashflow",
                    fields_used=("accrual_ratio",),
                    operators_used=("rank",),
                    depth=2,
                    generation_metadata={
                        "search_bucket_id": "accrual_vs_cashflow|fundamental|balanced",
                        "recipe_family": "accrual_vs_cashflow",
                        "objective_profile": "balanced",
                    },
                ),
            ],
        )
        _seed_submission(repository, run_id="run-recipe", batch_id="batch-hot", job_id="job-hot", alpha_id="recipe-hot")
        _seed_submission(repository, run_id="run-recipe", batch_id="batch-cold", job_id="job-cold", alpha_id="recipe-cold")
        repository.brain_results.save_results(
            [
                BrainResultRecord(
                    job_id="job-hot",
                    run_id="run-recipe",
                    round_index=1,
                    batch_id="batch-hot",
                    candidate_id="recipe-hot",
                    expression="rank(gross_margin)",
                    status="completed",
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                    neutralization="SECTOR",
                    decay=0,
                    sharpe=0.40,
                    fitness=0.20,
                    turnover=0.30,
                    drawdown=0.10,
                    returns=0.05,
                    margin=0.03,
                    submission_eligible=True,
                    rejection_reason=None,
                    raw_result_json="{}",
                    metric_source="external_brain",
                    simulated_at="2026-04-22T00:10:00+00:00",
                    created_at="2026-04-22T00:10:00+00:00",
                ),
                BrainResultRecord(
                    job_id="job-cold",
                    run_id="run-recipe",
                    round_index=1,
                    batch_id="batch-cold",
                    candidate_id="recipe-cold",
                    expression="rank(accrual_ratio)",
                    status="completed",
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                    neutralization="SECTOR",
                    decay=0,
                    sharpe=-0.30,
                    fitness=-0.20,
                    turnover=0.70,
                    drawdown=0.30,
                    returns=-0.04,
                    margin=0.01,
                    submission_eligible=False,
                    rejection_reason=None,
                    raw_result_json="{}",
                    metric_source="external_brain",
                    simulated_at="2026-04-22T00:10:00+00:00",
                    created_at="2026-04-22T00:10:00+00:00",
                ),
            ]
        )
        repository.save_stage_metrics(
            [
                StageMetricRecord(
                    run_id="run-recipe",
                    round_index=1,
                    stage="generation",
                    metrics_json='{"recipe_guided_bucket_counts":{"fundamental_quality|fundamental|balanced":30,"accrual_vs_cashflow|fundamental|balanced":30},"recipe_guided_selected_by_bucket":{"fundamental_quality|fundamental|balanced":12,"accrual_vs_cashflow|fundamental|balanced":1}}',
                    created_at="2026-04-22T00:11:00+00:00",
                )
            ]
        )
        generation_config = replace(
            load_config("config/dev.yaml").generation,
            allowed_fields=["gross_margin", "operating_cash_flow", "accrual_ratio", "eps_revision"],
        )
        recipe_config = RecipeGenerationConfig(
            enabled_recipe_families=["fundamental_quality", "accrual_vs_cashflow"],
            objective_profiles=["balanced"],
            active_bucket_count=2,
            max_recipe_candidates_per_round=4,
            max_candidates_per_bucket=4,
            yield_lookback_rounds=5,
            min_bucket_support_for_penalty=1,
            dynamic_budget_min_generated_support=1,
            dynamic_budget_min_completed_support=1,
            bucket_exploration_floor=1,
            bucket_reallocation_strength=0.60,
            bucket_suppression_enabled=True,
            bucket_suppression_min_support=1,
            bucket_suppression_sharpe_floor=0.30,
            bucket_suppression_fitness_floor=0.10,
            bucket_suppression_max_candidates=1,
        )

        result = RecipeGuidedGenerator(repository).generate(
            config=recipe_config,
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(region="USA", regime_key="local", global_regime_key="global"),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-recipe",
            round_index=3,
            count=4,
        )
    finally:
        repository.close()

    hot_budget = result.stats.budget_allocations["fundamental_quality|fundamental|balanced"]
    cold_budget = result.stats.budget_allocations["accrual_vs_cashflow|fundamental|balanced"]
    assert hot_budget >= cold_budget
    assert cold_budget == 1
    assert result.stats.suppressed_bucket_caps == {"accrual_vs_cashflow|fundamental|balanced": 1}
    assert result.stats.yield_scores["fundamental_quality|fundamental|balanced"] > result.stats.yield_scores["accrual_vs_cashflow|fundamental|balanced"]


def test_recipe_guided_generator_retries_alternate_fields_when_top_drafts_duplicate() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, "run-recipe")
        generation_config = replace(
            load_config("config/dev.yaml").generation,
            allowed_fields=[
                "gross_margin",
                "operating_cash_flow",
                "free_cash_flow",
                "eps_revision",
            ],
        )
        recipe_config = RecipeGenerationConfig(
            enabled_recipe_families=["fundamental_quality"],
            objective_profiles=["balanced"],
            active_bucket_count=1,
            max_recipe_candidates_per_round=3,
            max_candidates_per_bucket=3,
            max_field_candidates_per_side=4,
            duplicate_retry_multiplier=4,
            enable_field_rotation=False,
        )

        result = RecipeGuidedGenerator(repository).generate(
            config=recipe_config,
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(region="USA", regime_key="local", global_regime_key="global"),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized={
                "rank(operating_cash_flow)",
                "rank(ts_mean(operating_cash_flow,2))",
                "zscore(ts_mean(operating_cash_flow,2))",
            },
            run_id="run-recipe",
            round_index=0,
            count=3,
        )
    finally:
        repository.close()

    assert result.candidates
    assert result.stats.duplicate_retry_count > 0
    assert result.stats.unique_draft_count > len(result.candidates)
    assert all(candidate.normalized_expression not in {"rank(operating_cash_flow)"} for candidate in result.candidates)


def test_recipe_guided_generator_tries_multiple_pairs_after_duplicate_top_pair() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, "run-recipe")
        generation_config = replace(
            load_config("config/dev.yaml").generation,
            allowed_fields=[
                "cash_top",
                "cash_alt",
                "accrual_top",
                "accrual_alt",
            ],
        )
        recipe_config = RecipeGenerationConfig(
            enabled_recipe_families=["accrual_vs_cashflow"],
            objective_profiles=["balanced"],
            active_bucket_count=1,
            max_recipe_candidates_per_round=3,
            max_candidates_per_bucket=3,
            max_field_candidates_per_side=4,
            max_pair_candidates_per_bucket=4,
            duplicate_retry_multiplier=4,
            enable_field_rotation=False,
        )

        result = RecipeGuidedGenerator(repository).generate(
            config=recipe_config,
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_pair_field_registry(),
            region_learning_context=RegionLearningContext(region="USA", regime_key="local", global_regime_key="global"),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized={
                "(rank(cash_top)-rank(accrual_top))",
                "(rank(ts_mean(cash_top,2))-rank(ts_mean(accrual_top,2)))",
                "(rank(ts_mean(cash_top,3))-rank(ts_mean(accrual_top,3)))",
            },
            run_id="run-recipe",
            round_index=0,
            count=3,
        )
    finally:
        repository.close()

    assert result.candidates
    assert result.stats.duplicate_retry_count > 0
    assert any(pair_key != "cash_top:accrual_top" for pair_key in result.stats.pair_usage_counts)


def test_recipe_guided_field_rotation_changes_first_draft_field() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, "run-recipe")
        generation_config = replace(
            load_config("config/dev.yaml").generation,
            allowed_fields=["gross_margin", "operating_cash_flow", "free_cash_flow"],
        )
        recipe_config = RecipeGenerationConfig(
            enabled_recipe_families=["fundamental_quality"],
            objective_profiles=["balanced"],
            active_bucket_count=1,
            max_recipe_candidates_per_round=1,
            max_candidates_per_bucket=1,
            max_field_candidates_per_side=3,
            enable_field_rotation=True,
        )
        common_kwargs = {
            "config": recipe_config,
            "adaptive_config": AdaptiveGenerationConfig(),
            "generation_config": generation_config,
            "registry": build_registry(generation_config.allowed_operators),
            "field_registry": _field_registry(),
            "region_learning_context": RegionLearningContext(region="USA", regime_key="local", global_regime_key="global"),
            "generation_guardrails": GenerationGuardrails(),
            "field_penalty_multipliers": {},
            "blocked_fields": set(),
            "existing_normalized": set(),
            "run_id": "run-recipe",
            "count": 1,
        }

        round_0 = RecipeGuidedGenerator(repository).generate(round_index=0, **common_kwargs)
        round_1 = RecipeGuidedGenerator(repository).generate(round_index=1, **common_kwargs)
    finally:
        repository.close()

    assert round_0.candidates
    assert round_1.candidates
    assert round_0.candidates[0].fields_used != round_1.candidates[0].fields_used


def test_group_recipes_yield_valid_expressions() -> None:
    result, generation_config, field_registry = _run_group_recipe_generation(count=40)

    group_candidates = _group_relative_candidates(result.candidates)

    assert group_candidates
    assert {candidate.generation_metadata["group_recipe_group"] for candidate in group_candidates} >= {
        "A",
        "B",
        "C",
        "D",
    }
    registry = build_registry(generation_config.allowed_operators)
    for candidate in group_candidates:
        _assert_expression_valid(
            candidate.expression,
            generation_config=generation_config,
            field_registry=field_registry,
            registry=registry,
        )


def test_group_key_from_registry_not_hardcoded() -> None:
    result, _, field_registry = _run_group_recipe_generation(count=24, group_names=("sector",))

    group_candidates = _group_relative_candidates(result.candidates)

    assert group_candidates
    assert all(candidate.generation_metadata["group_recipe_group_key"] == "sector" for candidate in group_candidates)
    assert all("subindustry" not in candidate.expression for candidate in group_candidates)
    assert field_registry.get("sector").operator_type == "group"


def test_group_recipe_budget_cap() -> None:
    result, _, _ = _run_group_recipe_generation(count=100)

    group_count = len(_group_relative_candidates(result.candidates))

    assert group_count <= 25


def test_group_field_diversity_cap() -> None:
    result, _, _ = _run_group_recipe_generation(count=100)

    group_candidates = _group_relative_candidates(result.candidates)
    primary_field_counts = Counter(
        candidate.generation_metadata["group_recipe_primary_field"]
        for candidate in group_candidates
    )

    assert group_candidates
    assert max(primary_field_counts.values()) <= 3


def test_invalid_group_recipe_skipped_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    original_build = AlphaGenerationEngine._build_candidate_result
    forced = {"count": 0}

    def flaky_build(
        self,
        expression,
        mode,
        parent_ids,
        generation_metadata=None,
        validation_ctx=None,
    ):
        if (
            generation_metadata
            and generation_metadata.get("generation_source") == "group_relative"
            and forced["count"] == 0
        ):
            forced["count"] += 1
            return CandidateBuildResult(candidate=None, failure_reason="validation_forced_group_failure")
        return original_build(
            self,
            expression,
            mode,
            parent_ids,
            generation_metadata=generation_metadata,
            validation_ctx=validation_ctx,
        )

    monkeypatch.setattr(AlphaGenerationEngine, "_build_candidate_result", flaky_build)

    result, _, _ = _run_group_recipe_generation(count=40)

    assert forced["count"] == 1
    assert _group_relative_candidates(result.candidates)
    assert result.stats.group_relative_skipped_count >= 1
    assert result.stats.group_relative_skip_reason_counts["validation_forced_group_failure"] == 1


@pytest.mark.parametrize(
    ("family", "dataset_family"),
    [
        ("analyst_estimate_recency", "analyst"),
        ("analyst_estimate_stability", "analyst"),
        ("analyst_profitability_spread", "analyst"),
        ("returns_term_structure", "returns"),
    ],
)
def test_elite_motif_recipe_families_generate_with_elite_lookbacks(family: str, dataset_family: str) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, "run-recipe")
        generation_config = replace(
            load_config("config/dev.yaml").generation,
            allowed_fields=[
                "eps_estimate",
                "ebit_estimate",
                "netprofit_estimate",
                "sales_estimate",
                "dividend_estimate",
                "return_5d",
                "return_250d",
                "sigma_60m",
            ],
            allowed_operators=[
                "rank",
                "zscore",
                "ts_mean",
                "ts_std_dev",
                "days_from_last_change",
                "ts_av_diff",
                "ts_scale",
                "ts_arg_max",
                "ts_arg_min",
                "ts_count_nans",
                "reverse",
            ],
            lookbacks=[2, 3],
        )
        recipe_config = RecipeGenerationConfig(
            enabled_recipe_families=[family],
            objective_profiles=["balanced"],
            active_bucket_count=1,
            max_recipe_candidates_per_round=6,
            max_candidates_per_bucket=6,
            max_field_candidates_per_side=4,
            max_pair_candidates_per_bucket=4,
        )

        result = RecipeGuidedGenerator(repository).generate(
            config=recipe_config,
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=_field_registry(),
            region_learning_context=RegionLearningContext(region="USA", regime_key="local", global_regime_key="global"),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-recipe",
            round_index=0,
            count=6,
        )
    finally:
        repository.close()

    assert result.candidates
    assert all(candidate.generation_mode == "recipe_guided" for candidate in result.candidates)
    assert all(candidate.generation_metadata["recipe_family"] == family for candidate in result.candidates)
    assert all(candidate.generation_metadata["dataset_family"] == dataset_family for candidate in result.candidates)
    assert all(candidate.generation_metadata["search_bucket_id"] == f"{family}|{dataset_family}|balanced" for candidate in result.candidates)
    joined = " ".join(candidate.expression for candidate in result.candidates)
    assert any(f",{lookback})" in joined for lookback in (125, 145, 150))
    assert ",2)" not in joined
    assert ",3)" not in joined


def _run_group_recipe_generation(
    *,
    count: int,
    group_names: tuple[str, ...] = ("subindustry", "sector", "industry"),
):
    field_registry = _group_recipe_field_registry(group_names=group_names)
    generation_config = _group_recipe_generation_config(field_registry)
    recipe_config = RecipeGenerationConfig(
        enabled_recipe_families=["fundamental_quality"],
        objective_profiles=["balanced"],
        active_bucket_count=1,
        max_recipe_candidates_per_round=count,
        max_candidates_per_bucket=count,
        enable_field_rotation=False,
        dynamic_budget_enabled=False,
    )
    repository = SQLiteRepository(":memory:")
    try:
        _seed_run(repository, "run-recipe")
        result = RecipeGuidedGenerator(repository).generate(
            config=recipe_config,
            adaptive_config=AdaptiveGenerationConfig(),
            generation_config=generation_config,
            registry=build_registry(generation_config.allowed_operators),
            field_registry=field_registry,
            region_learning_context=RegionLearningContext(region="USA", regime_key="local", global_regime_key="global"),
            generation_guardrails=GenerationGuardrails(),
            field_penalty_multipliers={},
            blocked_fields=set(),
            existing_normalized=set(),
            run_id="run-recipe",
            round_index=0,
            count=count,
        )
    finally:
        repository.close()
    return result, generation_config, field_registry


def _group_relative_candidates(candidates: list[AlphaCandidate]) -> list[AlphaCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.generation_metadata.get("generation_source") == "group_relative"
    ]


def _assert_expression_valid(
    expression: str,
    *,
    generation_config,
    field_registry: FieldRegistry,
    registry,
) -> None:
    allowed_fields = field_registry.generation_allowed_fields(
        generation_config.allowed_fields,
        include_catalog_fields=generation_config.allow_catalog_fields_without_runtime,
    )
    validation = validate_expression(
        node=parse_expression(expression),
        registry=registry,
        allowed_fields=allowed_fields,
        max_depth=generation_config.max_depth,
        group_fields={
            spec.name
            for spec in field_registry.generation_group_key_fields(
                include_catalog_fields=generation_config.allow_catalog_fields_without_runtime,
            )
        },
        field_types=field_registry.field_types(allowed=allowed_fields),
        field_categories={name: spec.category for name, spec in field_registry.fields.items()},
        complexity_limit=generation_config.complexity_limit,
        exact_field_types=field_registry.exact_field_types() if hasattr(field_registry, "exact_field_types") else None,
    )
    assert validation.is_valid, validation.errors


def _group_recipe_generation_config(field_registry: FieldRegistry):
    return replace(
        load_config("config/dev.yaml").generation,
        allowed_fields=[
            name
            for name, spec in field_registry.fields.items()
            if spec.operator_type == "matrix"
        ],
        allowed_operators=[
            "rank",
            "zscore",
            "quantile",
            "ts_mean",
            "ts_delta",
            "ts_decay_linear",
            "ts_std_dev",
            "ts_rank",
            "ts_scale",
            "ts_av_diff",
            "ts_arg_max",
            "ts_arg_min",
            "ts_count_nans",
            "min",
            "max",
            "inverse",
            "reverse",
            "days_from_last_change",
            "group_rank",
            "group_zscore",
            "group_neutralize",
        ],
        lookbacks=[5, 10, 20],
        max_depth=8,
        complexity_limit=30,
    )


def _group_recipe_field_registry(
    *,
    group_names: tuple[str, ...],
) -> FieldRegistry:
    numeric_fields = {
        "anl39_agrosmgn": _field_spec("anl39_agrosmgn", category="analyst", description="gross margin"),
        "anl39_agrosmgn2": _field_spec("anl39_agrosmgn2", category="analyst", description="gross margin"),
        "anl39_epschngin": _field_spec("anl39_epschngin", category="analyst", description="eps change"),
        "anl39_ttmepsincx": _field_spec("anl39_ttmepsincx", category="analyst", description="trailing eps"),
        "anl39_qepsinclxo": _field_spec("anl39_qepsinclxo", category="analyst", description="quarterly eps"),
        "anl39_rasv2_atotd2eq": _field_spec("anl39_rasv2_atotd2eq", category="analyst", description="leverage"),
        "anl46_sentiment": _field_spec("anl46_sentiment", category="analyst", description="sentiment"),
        "anl46_performancepercentile": _field_spec(
            "anl46_performancepercentile",
            category="analyst",
            description="performance percentile",
        ),
        "anl69_roe_best_cur_fiscal_year_period": _field_spec(
            "anl69_roe_best_cur_fiscal_year_period",
            category="analyst",
            description="roe estimate",
        ),
        "anl69_roa_best_cur_fiscal_year_period": _field_spec(
            "anl69_roa_best_cur_fiscal_year_period",
            category="analyst",
            description="roa estimate",
        ),
        "anl69_eps_best_eeps_nxt_yr": _field_spec(
            "anl69_eps_best_eeps_nxt_yr",
            category="analyst",
            description="next year eps estimate",
        ),
        "anl69_ebit_best_eeps_nxt_yr": _field_spec(
            "anl69_ebit_best_eeps_nxt_yr",
            category="analyst",
            description="next year ebit estimate",
        ),
        "anl69_eps_best_eeps_cur_yr": _field_spec(
            "anl69_eps_best_eeps_cur_yr",
            category="analyst",
            description="current year eps estimate",
        ),
        "anl69_roe_best_eeps_nxt_yr": _field_spec(
            "anl69_roe_best_eeps_nxt_yr",
            category="analyst",
            description="next year roe estimate",
        ),
        "anl69_roa_best_eeps_nxt_yr": _field_spec(
            "anl69_roa_best_eeps_nxt_yr",
            category="analyst",
            description="next year roa estimate",
        ),
        "returns": _field_spec("returns", category="price", description="returns"),
        "close": _field_spec("close", category="price", description="close"),
        "volume": _field_spec("volume", category="volume", description="volume"),
    }
    for name in group_names:
        numeric_fields[name] = _group_key_spec(name)
    return FieldRegistry(fields=numeric_fields)


def _field_registry() -> FieldRegistry:
    return FieldRegistry(
        fields={
            "gross_margin": _field_spec("gross_margin", category="fundamental", description="profitability margin quality"),
            "operating_cash_flow": _field_spec("operating_cash_flow", category="fundamental", description="cash cfo fcf"),
            "free_cash_flow": _field_spec("free_cash_flow", category="fundamental", description="cash free_cash_flow fcf"),
            "accrual_ratio": _field_spec("accrual_ratio", category="fundamental", description="accrual reserve inventory"),
            "receivable_accrual": _field_spec("receivable_accrual", category="fundamental", description="accrual receivable reserve"),
            "book_value_yield": _field_spec("book_value_yield", category="fundamental", description="book value earnings_yield"),
            "sales_growth": _field_spec("sales_growth", category="fundamental", description="growth forecast expected"),
            "eps_revision": _field_spec("eps_revision", category="analyst", description="estimate revision analyst forecast surprise"),
            "eps_estimate": _field_spec("eps_estimate", category="analyst", description="eps earnings estimate analyst forecast"),
            "ebit_estimate": _field_spec("ebit_estimate", category="analyst", description="ebit profit estimate analyst"),
            "netprofit_estimate": _field_spec("netprofit_estimate", category="analyst", description="net profit income estimate"),
            "sales_estimate": _field_spec("sales_estimate", category="analyst", description="sales revenue estimate forecast"),
            "dividend_estimate": _field_spec("dividend_estimate", category="analyst", description="dividend dps estimate"),
            "return_5d": _field_spec("return_5d", category="price", description="short daily return 5d ret"),
            "return_250d": _field_spec("return_250d", category="price", description="long return 250d ret"),
            "sigma_60m": _field_spec("sigma_60m", category="model", description="monthly 60m sigma volatility returns"),
        }
    )


def _pair_field_registry() -> FieldRegistry:
    return FieldRegistry(
        fields={
            "cash_top": _field_spec(
                "cash_top",
                category="fundamental",
                description="cash cfo fcf operating_cash",
                field_score=3.0,
            ),
            "cash_alt": _field_spec(
                "cash_alt",
                category="fundamental",
                description="cash cfo fcf operating_cash",
                field_score=2.0,
            ),
            "accrual_top": _field_spec(
                "accrual_top",
                category="fundamental",
                description="accrual receivable working_capital inventory",
                field_score=3.0,
            ),
            "accrual_alt": _field_spec(
                "accrual_alt",
                category="fundamental",
                description="accrual receivable working_capital inventory",
                field_score=2.0,
            ),
        }
    )


def _field_spec(name: str, *, category: str, description: str, field_score: float = 1.0) -> FieldSpec:
    return FieldSpec(
        name=name,
        dataset="catalog",
        field_type="matrix",
        coverage=1.0,
        alpha_usage_count=0,
        category=category,
        runtime_available=True,
        description=description,
        category_weight=1.0,
        field_score=field_score,
    )


def _group_key_spec(name: str) -> FieldSpec:
    return FieldSpec(
        name=name,
        dataset="runtime",
        field_type="vector",
        coverage=1.0,
        alpha_usage_count=0,
        category="group",
        runtime_available=True,
        description=f"{name} group key",
        category_weight=1.0,
        field_score=1.0,
    )


def _seed_run(repository: SQLiteRepository, run_id: str) -> None:
    repository.upsert_run(
        run_id=run_id,
        seed=7,
        config_path="config/dev.yaml",
        config_snapshot="{}",
        status="running",
        started_at="2026-04-23T00:00:00+00:00",
    )


def _seed_submission(repository: SQLiteRepository, *, run_id: str, batch_id: str, job_id: str, alpha_id: str) -> None:
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
            created_at="2026-04-23T00:00:00+00:00",
            updated_at="2026-04-23T00:00:00+00:00",
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
                expression="{}",
                backend="api",
                status="completed",
                sim_config_snapshot="{}",
                submitted_at="2026-04-23T00:00:00+00:00",
                updated_at="2026-04-23T00:00:10+00:00",
                completed_at="2026-04-23T00:00:10+00:00",
                export_path=None,
                raw_submission_json="{}",
                error_message=None,
            )
        ]
    )
