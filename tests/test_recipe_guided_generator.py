from __future__ import annotations

from dataclasses import replace

import pytest

from core.config import AdaptiveGenerationConfig, RecipeGenerationConfig, load_config
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import build_registry
from generator.engine import AlphaCandidate
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
