from __future__ import annotations

from pathlib import Path

import yaml

from core.config import load_config


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
    assert default_config.loop.simulation_batch_size == default_config.brain.batch_size
    assert default_config.service.poll_interval_seconds == default_config.brain.poll_interval_seconds
    assert default_config.service.max_pending_jobs == default_config.brain.batch_size
    assert default_config.service.lock_name == "brain-service"
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
    assert default_config.adaptive_generation.region_learning.enabled is True
    assert default_config.adaptive_generation.region_learning.local_scope == "region_regime"
    assert default_config.adaptive_generation.region_learning.global_prior_scope == "match_non_region_regime"
    assert default_config.adaptive_generation.region_learning.blend_mode == "linear_ramp"


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
