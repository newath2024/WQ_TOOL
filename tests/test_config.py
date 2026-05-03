from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.config import (
    BrainConfig,
    EliteMotifConfig,
    OperatorDiversityBoostConfig,
    PreSimSelectionWeightsConfig,
    QualityOptimizationConfig,
    RecipeGenerationConfig,
    SearchSpaceFilterConfig,
    load_config,
)


def test_grouped_evaluation_config_builds_submission_thresholds(tmp_path: Path) -> None:
    config_path = tmp_path / "grouped.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "examples/sample_data/daily_ohlcv.csv"},
                "splits": {
                    "train": {"start": "2021-01-01", "end": "2021-02-01"},
                    "validation": {"start": "2021-02-02", "end": "2021-03-01"},
                    "test": {"start": "2021-03-02", "end": "2021-03-31"},
                },
                "generation": {
                    "allowed_fields": ["close", "volume", "returns"],
                    "allowed_operators": ["rank", "delta"],
                    "lookbacks": [2, 5],
                    "max_depth": 4,
                    "complexity_limit": 10,
                    "template_count": 2,
                    "grammar_count": 2,
                    "mutation_count": 1,
                    "normalization_wrappers": ["rank"],
                },
                "backtest": {"timeframe": "1d"},
                "evaluation": {
                    "hard_filters": {
                        "min_validation_sharpe": 0.25,
                        "max_validation_turnover": 3.0,
                        "max_validation_drawdown": 0.5,
                    },
                    "data_requirements": {
                        "min_validation_observations": 15,
                        "min_stability": 0.2,
                    },
                    "diversity": {
                        "signal_correlation_threshold": 0.9,
                        "returns_correlation_threshold": 0.9,
                    },
                    "ranking": {"top_k": 7},
                    "robustness": {
                        "enable_subuniverse_test": True,
                        "enable_ladder_test": True,
                        "enable_robustness_test": True,
                        "ladder_buckets": 3,
                        "ladder_min_sharpe": 0.0,
                        "ladder_min_passes": 2,
                        "subuniverse_min_sharpe": -0.1,
                        "subuniverse_min_pass_fraction": 0.6,
                        "robustness_min_fitness_ratio": 0.1,
                    },
                },
                "storage": {"path": str(tmp_path / "out.sqlite3")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.evaluation.min_sharpe == 0.25
    assert config.evaluation.max_turnover == 3.0
    assert config.evaluation.top_k == 7
    assert config.submission_tests.subuniverse_min_pass_fraction == 0.6
    assert config.runtime.profile_name == "grouped"


def test_default_and_research_profiles_normalize_to_same_thresholds() -> None:
    default_config = load_config("config/default.yaml")
    research_config = load_config("config/research.yaml")

    assert default_config.runtime.profile_name == "research"
    assert default_config.runtime.progress_log_enabled is True
    assert default_config.runtime.progress_log_dir == ""
    assert research_config.runtime.profile_name == "research"
    assert default_config.evaluation.min_sharpe == research_config.evaluation.min_sharpe
    assert default_config.evaluation.max_turnover == research_config.evaluation.max_turnover
    assert default_config.evaluation.min_observations == research_config.evaluation.min_observations
    assert (
        default_config.submission_tests.robustness_min_fitness_ratio
        == research_config.submission_tests.robustness_min_fitness_ratio
    )
    assert default_config.brain.backend == "manual"
    assert default_config.brain.nan_handling == "OFF"
    assert default_config.brain.unit_handling == "VERIFY"
    assert default_config.brain.neutralization == "SECTOR"
    assert default_config.brain.simulation_profiles == []
    assert default_config.loop.simulation_batch_size == default_config.brain.batch_size
    assert default_config.service.poll_interval_seconds == default_config.brain.poll_interval_seconds
    assert default_config.service.max_pending_jobs == default_config.brain.batch_size
    assert default_config.service.lock_name == "brain-service"
    assert default_config.service.ambiguous_submission_policy == "fail"
    assert default_config.brain.credentials_file == "secrets/brain_credentials.json"
    assert default_config.brain.session_path == "outputs/brain_api_session.json"
    assert default_config.brain.persona_poll_interval_seconds == 15
    assert default_config.brain.persona_timeout_seconds == 1800
    assert "ts_delta" in default_config.generation.allowed_operators
    assert "ts_corr" in default_config.generation.allowed_operators
    assert "ts_covariance" in default_config.generation.allowed_operators
    assert "ts_decay_linear" in default_config.generation.allowed_operators
    assert "ts_std_dev" in default_config.generation.allowed_operators
    assert "delta" not in default_config.generation.allowed_operators
    assert "correlation" not in default_config.generation.allowed_operators
    assert default_config.generation.operator_catalog_paths
    assert default_config.generation.sim_neutralization == "SECTOR"
    assert default_config.generation.sim_decay == 0
    assert default_config.adaptive_generation.region_learning.enabled is True
    assert default_config.adaptive_generation.region_learning.local_scope == "region_regime"
    assert default_config.adaptive_generation.region_learning.global_prior_scope == "match_non_region_regime"
    assert default_config.adaptive_generation.region_learning.blend_mode == "linear_ramp"
    assert default_config.adaptive_generation.meta_model.enabled is True
    assert default_config.adaptive_generation.meta_model.rollout_mode == "blend"
    assert default_config.adaptive_generation.meta_model.target == "positive_outcome"
    assert default_config.adaptive_generation.learned_regime.enabled is True
    assert default_config.adaptive_generation.learned_regime.model_type == "minibatch_kmeans"
    assert default_config.adaptive_generation.learned_regime.tsfresh_profile == "minimal"
    assert default_config.generation.engine_validation_cache_enabled is True
    assert default_config.adaptive_generation.max_generation_seconds == 20.0
    assert default_config.adaptive_generation.max_attempt_multiplier == 12
    assert default_config.adaptive_generation.exploit_budget_ratio == 0.60
    assert default_config.adaptive_generation.explore_budget_ratio == 0.40
    assert default_config.adaptive_generation.min_explore_attempts == 150
    assert default_config.adaptive_generation.min_explore_seconds == 2.0
    assert default_config.adaptive_generation.max_consecutive_failures == 400
    assert default_config.adaptive_generation.explore_max_consecutive_failures is None
    assert default_config.adaptive_generation.min_candidates_before_early_exit == 5
    assert default_config.service.research_context_cache_enabled is True
    assert default_config.service.research_context_cache_ttl_seconds == 0


def test_brain_config_code_defaults_include_weighted_simulation_profiles() -> None:
    brain = BrainConfig()

    assert brain.neutralization == "SUBINDUSTRY"
    assert brain.decay == 3
    assert brain.truncation == 0.01
    assert [profile.name for profile in brain.simulation_profiles] == ["stable", "aggressive_short"]
    assert brain.simulation_profiles[0].universe == "TOP1000"
    assert brain.simulation_profiles[0].weight == 0.6
    assert brain.simulation_profiles[1].universe == "TOP500"
    assert brain.simulation_profiles[1].weight == 0.4


def test_recipe_generation_config_defaults_and_validation() -> None:
    config = RecipeGenerationConfig()

    assert config.recipe_budget_fraction == 0.20
    assert config.max_recipe_candidates_per_round == 24
    assert config.active_bucket_count == 4
    assert config.max_field_candidates_per_side == 8
    assert config.max_pair_candidates_per_bucket == 12
    assert config.max_drafts_per_bucket == 64
    assert config.duplicate_retry_multiplier == 4
    assert config.enable_field_rotation is True
    assert config.field_rotation_lookback_rounds == 12
    assert config.dynamic_budget_enabled is True
    assert config.dynamic_budget_min_generated_support == 10
    assert config.dynamic_budget_min_completed_support == 3
    assert config.bucket_suppression_enabled is False
    assert config.bucket_suppression_min_support == 5
    assert config.bucket_suppression_sharpe_floor == 0.30
    assert config.bucket_suppression_fitness_floor == 0.10
    assert config.bucket_suppression_max_candidates == 1
    assert config.source_exploration_floor_fractions == {
        "quality_polish": 0.10,
        "recipe_guided": 0.10,
        "fresh": 0.30,
    }
    assert config.source_reallocation_strength == 0.65
    assert config.max_fresh_budget_fraction == 1.0
    assert config.fresh_spillover_fraction == 1.0
    assert config.bucket_exploration_floor == 1
    assert config.bucket_reallocation_strength == 0.75
    assert config.bucket_biases["revision_surprise|fundamental|balanced"] == 1.35
    assert config.bucket_biases["fundamental_quality|fundamental|quality"] == 0.70
    assert config.enabled_recipe_families == [
        "fundamental_quality",
        "accrual_vs_cashflow",
        "value_vs_growth",
        "revision_surprise",
    ]
    assert config.objective_profiles == ["balanced", "quality", "low_turnover"]

    with pytest.raises(ValueError):
        RecipeGenerationConfig(recipe_budget_fraction=1.5)
    with pytest.raises(ValueError):
        RecipeGenerationConfig(active_bucket_count=0)
    with pytest.raises(ValueError):
        RecipeGenerationConfig(source_reallocation_strength=1.5)
    with pytest.raises(ValueError):
        RecipeGenerationConfig(bucket_suppression_min_support=0)
    with pytest.raises(ValueError):
        RecipeGenerationConfig(bucket_suppression_max_candidates=-1)
    with pytest.raises(ValueError):
        RecipeGenerationConfig(max_pair_candidates_per_bucket=0)
    with pytest.raises(ValueError):
        RecipeGenerationConfig(bucket_biases={"x": 0.0})
    with pytest.raises(ValueError):
        RecipeGenerationConfig(enabled_recipe_families=[])


def test_search_space_filter_config_defaults_and_validation() -> None:
    config = SearchSpaceFilterConfig()

    assert config.enabled is False
    assert config.profile_mismatch_multiplier == 0.25
    assert config.unknown_profile_multiplier == 0.75
    assert config.validation_field_multiplier == 0.35
    assert config.validation_field_min_count == 2
    assert config.completed_lookback_rounds == 20
    assert config.min_completed_support == 3
    assert config.sharpe_floor == 0.30
    assert config.fitness_floor == 0.10
    assert config.field_result_multiplier == 0.50
    assert config.operator_result_multiplier == 0.60
    assert config.field_floor_ratio == 0.30
    assert config.field_floor_absolute_min == 0.10
    assert config.operator_floor_absolute_min == 0.05
    assert config.exploration_budget_pct == 0.15
    assert config.winner_prior_enabled is False
    assert config.winner_prior_lookback_rounds == 50
    assert config.winner_prior_min_support == 2
    assert config.winner_prior_min_completed == 15
    assert config.winner_prior_min_winners_for_boost == 3
    assert config.winner_prior_min_losers_for_penalty == 3
    assert config.winner_prior_laplace_k == 1.0
    assert config.winner_prior_multiplier_max == 1.5
    assert config.winner_prior_multiplier_min == 0.5
    assert config.winner_prior_alltime_dampen == 0.5
    assert config.winner_prior_cache_ttl_seconds == 300
    assert config.winner_prior_min_sharpe == 0.50
    assert config.winner_prior_min_fitness == 0.0
    assert config.winner_prior_sharpe_floor == 0.30
    assert config.winner_prior_fitness_floor == 0.10
    assert config.winner_prior_strong_sharpe_floor == 0.50
    assert config.winner_prior_strong_fitness_floor == 0.30
    assert config.winner_field_multiplier == 1.35
    assert config.strong_winner_field_multiplier == 1.80
    assert config.weak_field_multiplier == 0.65
    assert config.winner_operator_multiplier == 1.35
    assert config.strong_winner_operator_multiplier == 1.80
    assert config.weak_operator_multiplier == 0.65
    assert config.lane_field_caps == {}
    assert config.lane_field_min_count == 0
    assert config.lane_operator_allowlists == {}

    custom = SearchSpaceFilterConfig(
        lane_field_caps={"fresh": "12"},
        lane_operator_allowlists={"quality_polish": ["rank", "", "zscore"]},
    )
    assert custom.lane_field_caps == {"fresh": 12}
    assert custom.lane_operator_allowlists == {"quality_polish": ["rank", "zscore"]}

    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(profile_mismatch_multiplier=1.5)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(validation_field_min_count=0)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(completed_lookback_rounds=0)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(min_completed_support=0)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(winner_prior_lookback_rounds=0)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(winner_prior_min_completed=0)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(winner_prior_min_winners_for_boost=0)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(winner_prior_min_losers_for_penalty=0)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(winner_prior_laplace_k=0.0)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(winner_prior_multiplier_min=1.5)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(winner_prior_multiplier_max=0.5)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(winner_prior_alltime_dampen=1.5)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(winner_prior_cache_ttl_seconds=-1)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(winner_field_multiplier=0.0)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(field_floor_ratio=1.5)
    with pytest.raises(ValueError):
        SearchSpaceFilterConfig(exploration_budget_pct=1.5)


def test_operator_diversity_boost_config_defaults_and_validation() -> None:
    config = OperatorDiversityBoostConfig()

    assert config.enabled is False
    assert config.dominant_decay_rate == 2.0
    assert config.dominant_min_multiplier == 0.30
    assert config.underused_boost == 3.0
    assert config.underused_decay == 0.5
    assert config.seed_corr_pair_probability == 0.20
    assert config.corr_min_lookback == 10
    assert config.invalid_retry_limit == 3
    assert "ts_mean" in config.dominant_operators
    assert "ts_corr" in config.underused_operators
    assert "days_from_last_change" in config.underused_operators
    assert "anl69_eps_expected_report_dt" in config.expected_report_date_fields
    assert ("returns", "anl69_eps_best_eeps_nxt_yr") in config.seed_corr_pairs

    with pytest.raises(ValueError):
        OperatorDiversityBoostConfig(dominant_min_multiplier=1.5)
    with pytest.raises(ValueError):
        OperatorDiversityBoostConfig(underused_boost=0.0)
    with pytest.raises(ValueError):
        OperatorDiversityBoostConfig(seed_corr_pair_probability=1.5)
    with pytest.raises(ValueError):
        OperatorDiversityBoostConfig(corr_min_lookback=0)


def test_quality_optimization_variant_budget_defaults_and_validation() -> None:
    config = QualityOptimizationConfig()

    assert config.variant_budget_percentages == {
        "surface": 0.30,
        "operator_substitution": 0.20,
        "neutralization": 0.15,
        "cross_section": 0.15,
        "composite": 0.10,
        "field_substitution": 0.10,
    }
    assert QualityOptimizationConfig(variant_budget_percentages={}).variant_budget_percentages == config.variant_budget_percentages
    with pytest.raises(ValueError):
        QualityOptimizationConfig(variant_budget_percentages={"surface": -0.1})
    with pytest.raises(ValueError):
        QualityOptimizationConfig(
            variant_budget_percentages={
                "surface": 0.0,
                "operator_substitution": 0.0,
                "neutralization": 0.0,
                "cross_section": 0.0,
                "composite": 0.0,
                "field_substitution": 0.0,
            }
        )


def test_elite_motif_config_defaults_and_validation() -> None:
    config = EliteMotifConfig()

    assert config.enabled is False
    assert config.lookbacks == [125, 145, 150]
    assert config.seed_expressions == []
    assert config.clone_similarity_threshold == 0.70
    assert config.max_quality_polish_seeds_per_round == 6
    assert config.max_seed_variants_per_seed == 4
    assert PreSimSelectionWeightsConfig().elite_motif_bonus == 0.0
    assert PreSimSelectionWeightsConfig().elite_seed_similarity_penalty == 0.0

    with pytest.raises(ValueError):
        EliteMotifConfig(clone_similarity_threshold=1.5)
    with pytest.raises(ValueError):
        EliteMotifConfig(max_quality_polish_seeds_per_round=-1)
    with pytest.raises(ValueError):
        EliteMotifConfig(max_seed_variants_per_seed=-1)


def test_region_learning_config_loads_and_legacy_yaml_keeps_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "region_learning.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "examples/sample_data/daily_ohlcv.csv"},
                "splits": {
                    "train": {"start": "2021-01-01", "end": "2021-02-01"},
                    "validation": {"start": "2021-02-02", "end": "2021-03-01"},
                    "test": {"start": "2021-03-02", "end": "2021-03-31"},
                },
                "generation": {
                    "allowed_fields": ["close", "volume", "returns"],
                    "allowed_operators": ["rank", "ts_delta"],
                    "lookbacks": [2, 5],
                    "max_depth": 4,
                    "complexity_limit": 10,
                    "template_count": 2,
                    "grammar_count": 2,
                    "mutation_count": 1,
                    "normalization_wrappers": ["rank"],
                },
                "adaptive_generation": {
                    "region_learning": {
                        "enabled": True,
                        "local_scope": "region_regime",
                        "global_prior_scope": "match_non_region_regime",
                        "blend_mode": "linear_ramp",
                        "min_local_pattern_samples": 5,
                        "full_local_pattern_samples": 25,
                        "min_local_case_samples": 3,
                        "full_local_case_samples": 12,
                        "allow_global_parent_fallback": False,
                    }
                },
                "backtest": {"timeframe": "1d"},
                "evaluation": {
                    "hard_filters": {},
                    "data_requirements": {},
                    "diversity": {},
                    "ranking": {},
                    "robustness": {},
                },
                "storage": {"path": str(tmp_path / "out.sqlite3")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.adaptive_generation.region_learning.min_local_pattern_samples == 5
    assert config.adaptive_generation.region_learning.full_local_pattern_samples == 25
    assert config.adaptive_generation.region_learning.min_local_case_samples == 3
    assert config.adaptive_generation.region_learning.full_local_case_samples == 12
    assert config.adaptive_generation.region_learning.allow_global_parent_fallback is False


def test_generation_config_can_override_explore_failure_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "explore_failure_limit.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "examples/sample_data/daily_ohlcv.csv"},
                "splits": {
                    "train": {"start": "2021-01-01", "end": "2021-02-01"},
                    "validation": {"start": "2021-02-02", "end": "2021-03-01"},
                    "test": {"start": "2021-03-02", "end": "2021-03-31"},
                },
                "generation": {
                    "allowed_fields": ["close", "volume", "returns"],
                    "allowed_operators": ["rank", "ts_delta"],
                    "lookbacks": [2, 5],
                    "max_depth": 4,
                    "complexity_limit": 10,
                    "template_count": 2,
                    "grammar_count": 2,
                    "mutation_count": 1,
                    "normalization_wrappers": ["rank"],
                },
                "adaptive_generation": {
                    "explore_max_consecutive_failures": 12,
                },
                "backtest": {"timeframe": "1d"},
                "evaluation": {
                    "hard_filters": {},
                    "data_requirements": {},
                    "diversity": {},
                    "ranking": {},
                    "robustness": {},
                },
                "storage": {"path": str(tmp_path / "out.sqlite3")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.adaptive_generation.explore_max_consecutive_failures == 12


def test_generation_config_can_enable_catalog_only_brain_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "brain_full_like.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "examples/sample_data/daily_ohlcv.csv"},
                "splits": {
                    "train": {"start": "2021-01-01", "end": "2021-02-01"},
                    "validation": {"start": "2021-02-02", "end": "2021-03-01"},
                    "test": {"start": "2021-03-02", "end": "2021-03-31"},
                },
                "generation": {
                    "allowed_fields": [],
                    "allowed_operators": ["rank", "ts_delta"],
                    "lookbacks": [2, 5],
                    "max_depth": 4,
                    "complexity_limit": 10,
                    "template_count": 2,
                    "grammar_count": 2,
                    "mutation_count": 1,
                    "normalization_wrappers": ["rank"],
                    "field_catalog_paths": ["inputs/wq_snapshots/2026-03-29"],
                    "allow_catalog_fields_without_runtime": True,
                },
                "backtest": {"timeframe": "1d"},
                "evaluation": {
                    "hard_filters": {},
                    "data_requirements": {},
                    "diversity": {},
                    "ranking": {},
                    "robustness": {},
                },
                "storage": {"path": str(tmp_path / "out.sqlite3")},
                "brain": {"region": "USA", "universe": "TOP3000", "delay": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.generation.allow_catalog_fields_without_runtime is True
    assert config.generation.field_catalog_paths == ["inputs/wq_snapshots/2026-03-29"]
    assert config.generation.allowed_fields == []


def test_generation_config_can_override_simulation_awareness_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "sim_awareness.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "examples/sample_data/daily_ohlcv.csv"},
                "splits": {
                    "train": {"start": "2021-01-01", "end": "2021-02-01"},
                    "validation": {"start": "2021-02-02", "end": "2021-03-01"},
                    "test": {"start": "2021-03-02", "end": "2021-03-31"},
                },
                "generation": {
                    "allowed_fields": ["close", "volume", "returns"],
                    "allowed_operators": ["rank", "ts_delta"],
                    "lookbacks": [2, 5],
                    "max_depth": 4,
                    "complexity_limit": 10,
                    "template_count": 2,
                    "grammar_count": 2,
                    "mutation_count": 1,
                    "normalization_wrappers": ["rank"],
                    "sim_neutralization": "sector",
                    "sim_decay": 5,
                },
                "backtest": {"timeframe": "1d"},
                "evaluation": {
                    "hard_filters": {},
                    "data_requirements": {},
                    "diversity": {},
                    "ranking": {},
                    "robustness": {},
                },
                "storage": {"path": str(tmp_path / "out.sqlite3")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.generation.sim_neutralization == "sector"
    assert config.generation.sim_decay == 5


def test_brain_full_profile_loads_simulation_profiles_and_propagates_generation_awareness() -> None:
    config = load_config("config/brain_full.yaml")

    assert config.brain.neutralization == "SUBINDUSTRY"
    assert config.brain.decay == 5
    assert config.brain.truncation == 0.01
    assert [profile.name for profile in config.brain.simulation_profiles] == ["fixed_laboratory_usa"]
    assert config.brain.simulation_profiles[0].universe == "TOP3000"
    assert config.service.ambiguous_submission_policy == "resubmit"
    assert config.generation.sim_neutralization == "SUBINDUSTRY"
    assert config.generation.sim_decay == 5
    assert config.quality_score.check_penalty_weight == 1.0
    assert config.quality_score.check_warning_weight == 0.5
    assert config.quality_score.rejection_penalty_weight == 1.0
    assert config.quality_score.base_rejection_penalty == 0.25
    penalty = config.adaptive_generation.local_validation_field_penalty
    assert penalty.enabled is True
    assert penalty.lookback_rounds == 20
    assert penalty.min_count == 2
    assert penalty.max_fields == 200
    assert penalty.penalty_strength == 1.0
    assert penalty.min_multiplier == 0.10
    assert penalty.sample_limit == 10
    search_filter = config.adaptive_generation.search_space_filter
    assert search_filter.enabled is True
    assert search_filter.lane_field_caps == {"recipe_guided": 60, "fresh": 60}
    assert search_filter.lane_field_min_count == 30
    assert search_filter.min_completed_support == 3
    assert search_filter.sharpe_floor == 0.30
    assert search_filter.fitness_floor == 0.10
    assert search_filter.field_result_multiplier == 1.0
    assert search_filter.operator_result_multiplier == 1.0
    assert search_filter.winner_prior_enabled is True
    assert search_filter.winner_prior_lookback_rounds == 50
    assert search_filter.winner_prior_min_support == 3
    assert search_filter.winner_prior_min_completed == 15
    assert search_filter.winner_prior_min_winners_for_boost == 3
    assert search_filter.winner_prior_min_losers_for_penalty == 3
    assert search_filter.winner_prior_laplace_k == 1.0
    assert search_filter.winner_prior_multiplier_max == 1.5
    assert search_filter.winner_prior_multiplier_min == 0.5
    assert search_filter.winner_prior_alltime_dampen == 0.5
    assert search_filter.winner_prior_cache_ttl_seconds == 300
    assert search_filter.winner_prior_min_sharpe == 0.50
    assert search_filter.winner_prior_min_fitness == 0.0
    assert search_filter.winner_field_multiplier == 1.35
    assert search_filter.strong_winner_field_multiplier == 1.80
    assert search_filter.weak_field_multiplier == 0.65
    assert search_filter.winner_operator_multiplier == 1.35
    assert search_filter.strong_winner_operator_multiplier == 1.80
    assert search_filter.weak_operator_multiplier == 0.65
    assert search_filter.check_penalty_min_support == 3
    assert search_filter.hard_fail_field_multiplier == 0.35
    assert search_filter.hard_fail_operator_multiplier == 1.0
    assert search_filter.blocking_warning_field_multiplier == 0.55
    assert search_filter.blocking_warning_operator_multiplier == 1.0
    assert search_filter.field_floor_ratio == 0.30
    assert search_filter.field_floor_absolute_min == 0.10
    assert search_filter.operator_floor_absolute_min == 0.05
    assert search_filter.exploration_budget_pct == 0.15
    assert search_filter.winner_prior_sharpe_floor == 0.50
    assert search_filter.winner_prior_fitness_floor == 0.0
    assert search_filter.winner_prior_strong_sharpe_floor == 1.00
    assert search_filter.winner_prior_strong_fitness_floor == 0.70
    assert search_filter.lane_operator_allowlists["quality_polish"] == [
        "rank",
        "zscore",
        "quantile",
        "ts_mean",
        "ts_decay_linear",
        "ts_std_dev",
        "ts_rank",
        "ts_sum",
        "ts_scale",
        "ts_arg_max",
        "ts_arg_min",
        "days_from_last_change",
        "ts_av_diff",
        "ts_count_nans",
        "inverse",
        "reverse",
        "sign",
        "abs",
        "min",
        "max",
        "group_neutralize",
    ]
    assert "quantile" in search_filter.lane_operator_allowlists["recipe_guided"]
    assert "ts_arg_max" in search_filter.lane_operator_allowlists["recipe_guided"]
    for operator in [
        "ts_delta",
        "ts_decay_linear",
        "ts_rank",
        "group_rank",
        "group_zscore",
        "group_neutralize",
    ]:
        assert operator in search_filter.lane_operator_allowlists["recipe_guided"]
    assert search_filter.lane_operator_allowlists["fresh"] == [
        "rank",
        "zscore",
        "ts_mean",
        "ts_decay_linear",
        "ts_std_dev",
        "ts_rank",
        "ts_sum",
        "ts_scale",
        "days_from_last_change",
        "ts_av_diff",
        "ts_count_nans",
        "sign",
        "abs",
        "min",
        "max",
        "group_neutralize",
    ]
    assert "ts_corr" not in search_filter.lane_operator_allowlists["fresh"]
    operator_boost = config.adaptive_generation.operator_diversity_boost
    assert operator_boost.enabled is True
    assert operator_boost.dominant_decay_rate == 2.0
    assert operator_boost.dominant_min_multiplier == 0.30
    assert operator_boost.underused_boost == 3.0
    assert operator_boost.underused_decay == 0.5
    assert operator_boost.seed_corr_pair_probability == 0.20
    assert operator_boost.corr_min_lookback == 10
    assert operator_boost.invalid_retry_limit == 3
    assert operator_boost.dominant_operators == [
        "ts_decay_linear",
        "ts_mean",
        "ts_sum",
        "ts_std_dev",
        "rank",
        "ts_rank",
        "zscore",
    ]
    assert "ts_corr" in operator_boost.underused_operators
    assert "ts_covariance" in operator_boost.underused_operators
    assert "days_from_last_change" in operator_boost.underused_operators
    assert "anl69_eps_expected_report_dt" in operator_boost.expected_report_date_fields
    assert ("ts_delta(close,5)", "anl39_epschngin") in operator_boost.seed_corr_pairs
    recipe = config.adaptive_generation.recipe_generation
    assert recipe.recipe_budget_fraction == 0.18
    assert recipe.max_recipe_candidates_per_round == 20
    assert recipe.active_bucket_count == 2
    assert recipe.max_candidates_per_bucket == 4
    assert recipe.source_exploration_floor_fractions == {
        "quality_polish": 0.24,
        "recipe_guided": 0.10,
        "fresh": 0.08,
    }
    assert recipe.bucket_suppression_enabled is True
    assert recipe.bucket_suppression_min_support == 3
    assert recipe.bucket_suppression_sharpe_floor == 0.30
    assert recipe.bucket_suppression_fitness_floor == 0.10
    assert recipe.bucket_suppression_max_candidates == 1
    assert recipe.max_fresh_budget_fraction == 0.30
    assert recipe.fresh_spillover_fraction == 0.20
    assert recipe.bucket_biases["fundamental_quality|fundamental|balanced"] == 1.05
    assert recipe.bucket_biases["value_vs_growth|fundamental|quality"] == 0.80
    assert recipe.bucket_biases["accrual_vs_cashflow|fundamental|balanced"] == 0.25
    assert "analyst_estimate_recency" in recipe.enabled_recipe_families
    assert "analyst_estimate_stability" in recipe.enabled_recipe_families
    assert "analyst_profitability_spread" in recipe.enabled_recipe_families
    assert "returns_term_structure" in recipe.enabled_recipe_families
    assert recipe.bucket_biases["analyst_estimate_recency|analyst|balanced"] == 1.45
    assert recipe.bucket_biases["analyst_estimate_stability|analyst|quality"] == 1.60
    assert recipe.bucket_biases["analyst_profitability_spread|analyst|balanced"] == 0.90
    assert recipe.bucket_biases["returns_term_structure|returns|balanced"] == 0.25
    selection = config.adaptive_generation.selection
    assert selection.pre_sim.brain_robustness_proxy_penalty == 0.35
    assert selection.pre_sim.elite_motif_bonus == 0.04
    assert selection.pre_sim.elite_seed_similarity_penalty == 0.20
    assert selection.brain_robustness_proxy.enabled is True
    assert selection.brain_robustness_proxy.lookback_rounds == 12
    assert selection.brain_robustness_proxy.min_support == 3
    assert selection.brain_robustness_proxy.sharpe_floor == 0.30
    assert selection.brain_robustness_proxy.fitness_floor == 0.10
    quality = config.adaptive_generation.quality_optimization
    assert quality.enabled is True
    assert quality.lookback_completed_results == 800
    assert quality.polish_budget_fraction == 0.22
    assert quality.max_polish_candidates_per_round == 24
    assert quality.max_polish_parents_per_round == 12
    assert quality.variants_per_parent == 10
    assert quality.min_parent_fitness == 0.20
    assert quality.min_parent_sharpe == 0.50
    assert quality.min_parent_turnover == 0.01
    assert quality.max_parent_turnover == 0.60
    assert quality.max_parent_drawdown == 0.75
    assert quality.min_completed_parent_count == 5
    assert quality.selection_prior_weight == 0.14
    assert quality.parent_scan_multiplier == 6
    assert quality.enabled_transforms == [
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
    ]
    assert quality.primary_transform == "wrap_rank"
    assert quality.max_variants_per_parent_by_transform == {
        "wrap_rank": 1,
        "wrap_zscore": 1,
        "window_perturb": 4,
        "smooth_ts_mean": 3,
        "smooth_ts_decay_linear": 3,
    }
    assert quality.parent_transform_recent_rounds == 2
    assert quality.max_parent_transform_uses_per_recent_window == 1
    assert quality.transform_score_lookback_rounds == 4
    assert quality.transform_cooldown_min_attempts == 3
    assert quality.transform_cooldown_success_rate_floor == 0.20
    assert quality.cooldown_exempt_transform_groups == ["smooth_ts_mean", "smooth_ts_decay_linear"]
    assert quality.window_perturb_neighbor_count == 4
    assert quality.variant_budget_percentages == {
        "surface": 0.30,
        "operator_substitution": 0.20,
        "neutralization": 0.15,
        "cross_section": 0.15,
        "composite": 0.10,
        "field_substitution": 0.10,
    }
    assert "smooth_ts_mean" not in quality.disabled_transforms
    assert "smooth_ts_decay_linear" not in quality.disabled_transforms
    assert "smooth_ts_rank" in quality.disabled_transforms
    elite = config.adaptive_generation.elite_motifs
    assert elite.enabled is True
    assert elite.lookbacks == [125, 145, 150]
    assert elite.clone_similarity_threshold == 0.70
    assert elite.max_quality_polish_seeds_per_round == 6
    assert elite.max_seed_variants_per_seed == 4
    assert len(elite.seed_expressions) == 10
    for operator in [
        "days_from_last_change",
        "ts_av_diff",
        "ts_scale",
        "ts_arg_max",
        "ts_arg_min",
        "quantile",
        "inverse",
        "reverse",
        "ts_count_nans",
        "min",
        "max",
    ]:
        assert operator in config.generation.allowed_operators
    for operator in [
        "power",
        "signed_power",
        "ts_product",
        "ts_regression",
        "ts_backfill",
        "ts_quantile",
        "ts_target_tvr_decay",
    ]:
        assert operator not in config.generation.allowed_operators


def test_legacy_yaml_without_generation_optimization_keys_keeps_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy_generation_defaults.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "examples/sample_data/daily_ohlcv.csv"},
                "splits": {
                    "train": {"start": "2021-01-01", "end": "2021-02-01"},
                    "validation": {"start": "2021-02-02", "end": "2021-03-01"},
                    "test": {"start": "2021-03-02", "end": "2021-03-31"},
                },
                "generation": {
                    "allowed_fields": ["close", "volume", "returns"],
                    "allowed_operators": ["rank", "ts_delta"],
                    "lookbacks": [2, 5],
                    "max_depth": 4,
                    "complexity_limit": 10,
                    "template_count": 2,
                    "grammar_count": 2,
                    "mutation_count": 1,
                    "normalization_wrappers": ["rank"],
                },
                "backtest": {"timeframe": "1d"},
                "evaluation": {
                    "hard_filters": {},
                    "data_requirements": {},
                    "diversity": {},
                    "ranking": {},
                    "robustness": {},
                },
                "storage": {"path": str(tmp_path / "out.sqlite3")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.generation.engine_validation_cache_enabled is True
    assert config.adaptive_generation.max_generation_seconds == 20.0
    assert config.adaptive_generation.max_attempt_multiplier == 12
    assert config.adaptive_generation.exploit_budget_ratio == 0.60
    assert config.adaptive_generation.explore_budget_ratio == 0.40
    assert config.adaptive_generation.min_explore_attempts == 150
    assert config.adaptive_generation.min_explore_seconds == 2.0
    assert config.adaptive_generation.max_consecutive_failures == 400
    assert config.adaptive_generation.explore_max_consecutive_failures is None
    assert config.adaptive_generation.min_candidates_before_early_exit == 5
    assert config.adaptive_generation.local_validation_field_penalty.enabled is True
    assert config.adaptive_generation.local_validation_field_penalty.lookback_rounds == 20
    assert config.adaptive_generation.search_space_filter.enabled is False
    assert config.adaptive_generation.search_space_filter.lane_field_caps == {}
    assert config.adaptive_generation.search_space_filter.lane_field_min_count == 0
    assert config.adaptive_generation.search_space_filter.lane_operator_allowlists == {}
    assert config.adaptive_generation.search_space_filter.winner_prior_enabled is False
    assert config.adaptive_generation.operator_diversity_boost.enabled is False
    assert config.adaptive_generation.quality_optimization.enabled is True
    assert config.adaptive_generation.quality_optimization.polish_budget_fraction == 0.35
    assert config.adaptive_generation.quality_optimization.enabled_transforms == [
        "wrap_rank",
        "wrap_zscore",
        "window_perturb",
    ]
    assert config.adaptive_generation.quality_optimization.max_variants_per_parent_by_transform["window_perturb"] == 4
    assert config.adaptive_generation.quality_optimization.parent_scan_multiplier == 1
    assert config.adaptive_generation.quality_optimization.cooldown_exempt_transform_groups == []
    assert config.adaptive_generation.quality_optimization.parent_transform_recent_rounds == 2
    assert config.adaptive_generation.quality_optimization.window_perturb_neighbor_count == 4
    assert config.adaptive_generation.quality_optimization.variant_budget_percentages["surface"] == 0.30
    assert config.adaptive_generation.quality_optimization.variant_budget_percentages["operator_substitution"] == 0.20
    assert config.adaptive_generation.recipe_generation.max_fresh_budget_fraction == 1.0
    assert config.adaptive_generation.recipe_generation.fresh_spillover_fraction == 1.0
    assert config.adaptive_generation.recipe_generation.bucket_suppression_enabled is False
    assert config.adaptive_generation.selection.pre_sim.brain_robustness_proxy_penalty == 0.0
    assert config.adaptive_generation.selection.pre_sim.elite_motif_bonus == 0.0
    assert config.adaptive_generation.selection.pre_sim.elite_seed_similarity_penalty == 0.0
    assert config.adaptive_generation.selection.brain_robustness_proxy.enabled is False
    assert config.adaptive_generation.elite_motifs.enabled is False
    assert config.adaptive_generation.elite_motifs.lookbacks == [125, 145, 150]
    assert config.adaptive_generation.elite_motifs.seed_expressions == []
    assert config.service.research_context_cache_enabled is True
    assert config.service.research_context_cache_ttl_seconds == 0
    assert config.runtime.progress_log_enabled is True
    assert config.runtime.progress_log_dir == ""
