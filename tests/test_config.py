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
    assert default_config.loop.simulation_batch_size == default_config.brain.batch_size
    assert default_config.brain.credentials_file == "secrets/brain_credentials.json"
    assert default_config.brain.session_path == "outputs/brain_api_session.json"
    assert default_config.brain.persona_poll_interval_seconds == 15
    assert default_config.brain.persona_timeout_seconds == 1800
