from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.config import (
    BrainConfig,
    EliteMotifConfig,
    PreSimSelectionWeightsConfig,
    RecipeGenerationConfig,
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
    penalty = config.adaptive_generation.local_validation_field_penalty
    assert penalty.enabled is True
    assert penalty.lookback_rounds == 20
    assert penalty.min_count == 2
    assert penalty.max_fields == 200
    assert penalty.penalty_strength == 1.0
    assert penalty.min_multiplier == 0.10
    assert penalty.sample_limit == 10
    recipe = config.adaptive_generation.recipe_generation
    assert recipe.recipe_budget_fraction == 0.32
    assert recipe.max_recipe_candidates_per_round == 40
    assert recipe.active_bucket_count == 4
    assert recipe.max_candidates_per_bucket == 6
    assert recipe.source_exploration_floor_fractions == {
        "quality_polish": 0.14,
        "recipe_guided": 0.22,
        "fresh": 0.14,
    }
    assert recipe.bucket_suppression_enabled is True
    assert recipe.bucket_suppression_min_support == 3
    assert recipe.bucket_suppression_sharpe_floor == 0.30
    assert recipe.bucket_suppression_fitness_floor == 0.10
    assert recipe.bucket_suppression_max_candidates == 1
    assert recipe.max_fresh_budget_fraction == 0.34
    assert recipe.fresh_spillover_fraction == 0.06
    assert recipe.bucket_biases["fundamental_quality|fundamental|balanced"] == 1.45
    assert recipe.bucket_biases["value_vs_growth|fundamental|quality"] == 1.25
    assert recipe.bucket_biases["accrual_vs_cashflow|fundamental|balanced"] == 0.35
    assert "analyst_estimate_recency" in recipe.enabled_recipe_families
    assert "analyst_estimate_stability" in recipe.enabled_recipe_families
    assert "analyst_profitability_spread" in recipe.enabled_recipe_families
    assert "returns_term_structure" in recipe.enabled_recipe_families
    assert recipe.bucket_biases["analyst_profitability_spread|analyst|balanced"] == 1.30
    assert recipe.bucket_biases["returns_term_structure|returns|balanced"] == 1.05
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
    assert quality.polish_budget_fraction == 0.12
    assert quality.max_polish_candidates_per_round == 16
    assert quality.max_polish_parents_per_round == 8
    assert quality.variants_per_parent == 8
    assert quality.min_parent_fitness == 0.02
    assert quality.min_parent_sharpe == 0.03
    assert quality.max_parent_turnover == 1.00
    assert quality.max_parent_drawdown == 0.75
    assert quality.min_completed_parent_count == 5
    assert quality.selection_prior_weight == 0.10
    assert quality.parent_scan_multiplier == 4
    assert quality.enabled_transforms == [
        "wrap_rank",
        "wrap_zscore",
        "window_perturb",
        "smooth_ts_mean",
        "smooth_ts_decay_linear",
    ]
    assert quality.primary_transform == "wrap_rank"
    assert quality.max_variants_per_parent_by_transform == {
        "wrap_rank": 1,
        "wrap_zscore": 1,
        "window_perturb": 4,
        "smooth_ts_mean": 2,
        "smooth_ts_decay_linear": 2,
    }
    assert quality.parent_transform_recent_rounds == 2
    assert quality.max_parent_transform_uses_per_recent_window == 1
    assert quality.transform_score_lookback_rounds == 4
    assert quality.transform_cooldown_min_attempts == 3
    assert quality.transform_cooldown_success_rate_floor == 0.20
    assert quality.cooldown_exempt_transform_groups == ["smooth_ts_mean", "smooth_ts_decay_linear"]
    assert quality.window_perturb_neighbor_count == 4
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
