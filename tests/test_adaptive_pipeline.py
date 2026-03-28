from __future__ import annotations

import sqlite3
from pathlib import Path

import yaml

from main import main
from tests.conftest import write_sample_csv, write_sample_metadata_csv


def write_adaptive_config(config_path: Path, data_path: Path, metadata_path: Path, storage_path: Path) -> None:
    config_payload = {
        "data": {
            "path": str(data_path),
            "format": "csv",
            "input_layout": "canonical",
            "default_timeframe": "1d",
            "selected_timeframe": "1d",
            "filename_pattern": "{symbol}_{timeframe}.csv",
            "column_mapping": {},
            "universe": [],
        },
        "aux_data": {
            "group_path": str(metadata_path),
            "factor_path": str(metadata_path),
            "mask_path": str(metadata_path),
            "format": "csv",
            "timestamp_column": "timestamp",
            "symbol_column": "symbol",
            "group_columns": ["sector", "industry", "country", "subindustry"],
            "factor_columns": ["beta", "size", "volatility", "liquidity"],
            "mask_columns": ["core_mask", "liquid_mask"],
            "column_mapping": {},
        },
        "splits": {
            "train": {"start": "2021-01-01", "end": "2021-02-12"},
            "validation": {"start": "2021-02-15", "end": "2021-03-05"},
            "test": {"start": "2021-03-08", "end": "2021-03-31"},
        },
        "generation": {
            "allowed_fields": ["open", "high", "low", "close", "volume", "returns"],
            "allowed_operators": [
                "delay",
                "delta",
                "returns",
                "rolling_mean",
                "rolling_std",
                "rolling_min",
                "rolling_max",
                "rank",
                "zscore",
                "correlation",
                "covariance",
                "decay_linear",
                "ts_rank",
                "ts_sum",
                "ts_mean",
                "ts_std",
                "sign",
                "abs",
                "log",
                "clip",
                "group_rank",
                "group_zscore",
                "group_neutralize",
            ],
            "lookbacks": [2, 3, 5, 10],
            "max_depth": 5,
            "complexity_limit": 20,
            "template_count": 12,
            "grammar_count": 12,
            "mutation_count": 8,
            "normalization_wrappers": ["rank", "zscore", "sign"],
            "random_seed": 11,
        },
        "adaptive_generation": {
            "enabled": True,
            "memory_scope": "regime",
            "success_rule": "validation_first",
            "strategy_mix": {
                "guided_mutation": 0.40,
                "memory_templates": 0.30,
                "random_exploration": 0.20,
                "novelty_behavior": 0.10,
            },
            "exploration_epsilon": 0.10,
            "sampling_temperature": 0.75,
            "family_cap_fraction": 1.0,
            "parent_pool_size": 12,
            "novelty_reference_top_k": 10,
            "min_pattern_support": 1,
            "pattern_decay": 0.98,
            "critic_thresholds": {
                "turnover_warning_fraction": 0.85,
                "overfit_gap_threshold": 0.35,
                "complexity_warning_fraction": 0.80,
                "noisy_short_horizon_max_lookback": 3,
                "novelty_success_threshold": 0.60,
                "score_prior_weight": 3.0,
            },
        },
        "simulation": {
            "delay_mode": "d1",
            "neutralization": "sector",
            "secondary_neutralization": "country",
            "pasteurize": True,
            "signal_clip": 5.0,
            "weight_clip": 0.35,
            "factor_columns": ["beta", "size", "volatility", "liquidity"],
            "subuniverses": [{"name": "liquid", "mask_field": "liquid_mask"}],
            "robustness_windows": [21, 42],
            "cache_enabled": True,
        },
        "backtest": {
            "timeframe": "1d",
            "mode": "cross_sectional",
            "portfolio_construction": "long_short",
            "selection_fraction": 0.34,
            "signal_delay": 1,
            "holding_period": 2,
            "volatility_scaling": True,
            "volatility_lookback": 5,
            "transaction_cost_bps": 5.0,
            "annualization_factor": 252,
            "symbol_rank_window": 10,
            "upper_quantile": 0.8,
            "lower_quantile": 0.2,
            "turnover_penalty": 0.1,
            "drawdown_penalty": 0.5,
        },
        "evaluation": {
            "min_sharpe": -10.0,
            "max_turnover": 10.0,
            "min_observations": 3,
            "max_drawdown": 1.0,
            "min_stability": 0.0,
            "signal_correlation_threshold": 0.99,
            "returns_correlation_threshold": 0.99,
            "top_k": 5,
        },
        "submission_tests": {
            "enable_subuniverse_test": False,
            "enable_ladder_test": False,
            "enable_robustness_test": False,
            "ladder_buckets": 3,
            "ladder_min_sharpe": -1.0,
            "ladder_min_passes": 1,
            "subuniverse_min_sharpe": -1.0,
            "subuniverse_min_pass_fraction": 0.5,
            "robustness_min_fitness_ratio": -5.0,
        },
        "storage": {"path": str(storage_path)},
        "runtime": {"log_level": "WARNING", "fail_fast": False},
    }
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")


def test_adaptive_memory_pipeline_learns_across_rounds(tmp_path: Path) -> None:
    data_path = tmp_path / "daily_ohlcv.csv"
    metadata_path = tmp_path / "symbol_metadata.csv"
    config_path = tmp_path / "adaptive.yaml"
    storage_path = tmp_path / "adaptive.sqlite3"
    write_sample_csv(data_path)
    write_sample_metadata_csv(metadata_path)
    write_adaptive_config(config_path, data_path, metadata_path, storage_path)

    assert main(["--config", str(config_path), "run-full-pipeline"]) == 0
    assert main(["--config", str(config_path), "run-full-pipeline"]) == 0

    connection = sqlite3.connect(storage_path)
    try:
        run_ids = [row[0] for row in connection.execute("SELECT run_id FROM runs ORDER BY started_at ASC").fetchall()]
    finally:
        connection.close()

    assert len(run_ids) == 2
    second_run = run_ids[-1]

    assert main(["--config", str(config_path), "--run-id", second_run, "mutate", "--from-top", "5", "--count", "8"]) == 0
    assert main(["--config", str(config_path), "--run-id", second_run, "evaluate"]) == 0
    assert main(["--config", str(config_path), "--run-id", second_run, "report", "--limit", "3"]) == 0
    assert main(["--config", str(config_path), "--run-id", second_run, "memory-top-patterns", "--limit", "3"]) == 0
    assert main(["--config", str(config_path), "--run-id", second_run, "memory-failed-patterns", "--limit", "3"]) == 0
    assert main(["--config", str(config_path), "--run-id", second_run, "memory-top-genes", "--limit", "3"]) == 0

    connection = sqlite3.connect(storage_path)
    try:
        history_count = connection.execute("SELECT COUNT(*) FROM alpha_history").fetchone()[0]
        diagnosis_count = connection.execute("SELECT COUNT(*) FROM alpha_diagnoses").fetchone()[0]
        pattern_count = connection.execute("SELECT COUNT(*) FROM alpha_patterns").fetchone()[0]
        membership_count = connection.execute("SELECT COUNT(*) FROM alpha_pattern_membership").fetchone()[0]
        adaptive_modes = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT generation_mode FROM alphas WHERE run_id = ?",
                (second_run,),
            ).fetchall()
        }
        guided_child = connection.execute(
            """
            SELECT alpha_id
            FROM alphas
            WHERE run_id = ? AND generation_mode = 'guided_mutation'
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (second_run,),
        ).fetchone()
        parent_links = connection.execute(
            "SELECT COUNT(*) FROM alpha_parents WHERE run_id = ?",
            (second_run,),
        ).fetchone()[0]
    finally:
        connection.close()

    assert history_count > 0
    assert diagnosis_count > 0
    assert pattern_count > 0
    assert membership_count > 0
    assert adaptive_modes & {"guided_mutation", "memory_template", "novelty_search"}
    assert parent_links > 0
    assert guided_child is not None
    assert main(["--config", str(config_path), "--run-id", second_run, "lineage", "--alpha-id", guided_child[0]]) == 0
