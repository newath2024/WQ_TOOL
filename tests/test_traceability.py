from __future__ import annotations

import sqlite3
from pathlib import Path

import yaml

from main import main
from tests.conftest import write_sample_csv, write_sample_metadata_csv


def test_pipeline_persists_run_traceability_metadata(tmp_path: Path) -> None:
    data_path = tmp_path / "daily_ohlcv.csv"
    metadata_path = tmp_path / "symbol_metadata.csv"
    storage_path = tmp_path / "trace.sqlite3"
    config_path = tmp_path / "research_nested.yaml"
    write_sample_csv(data_path)
    write_sample_metadata_csv(metadata_path)

    config_path.write_text(
        yaml.safe_dump(
            {
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
                    "allowed_operators": ["rank", "delta", "zscore", "ts_mean", "correlation", "covariance"],
                    "lookbacks": [2, 3, 5],
                    "max_depth": 4,
                    "complexity_limit": 20,
                    "template_count": 4,
                    "grammar_count": 4,
                    "mutation_count": 2,
                    "normalization_wrappers": ["rank", "zscore"],
                    "random_seed": 11,
                },
                "adaptive_generation": {"enabled": False},
                "simulation": {
                    "delay_mode": "d1",
                    "neutralization": "sector",
                    "subuniverses": [{"name": "liquid", "mask_field": "liquid_mask"}],
                    "cache_enabled": True,
                },
                "backtest": {
                    "timeframe": "1d",
                    "mode": "cross_sectional",
                    "portfolio_construction": "long_short",
                    "selection_fraction": 0.34,
                    "signal_delay": 1,
                    "holding_period": 2,
                    "volatility_scaling": False,
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
                    "hard_filters": {
                        "min_validation_sharpe": -10.0,
                        "max_validation_turnover": 10.0,
                        "max_validation_drawdown": 1.0,
                    },
                    "data_requirements": {
                        "min_validation_observations": 3,
                        "min_stability": 0.0,
                    },
                    "diversity": {
                        "signal_correlation_threshold": 0.99,
                        "returns_correlation_threshold": 0.99,
                    },
                    "ranking": {"top_k": 5},
                    "robustness": {
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
                },
                "storage": {"path": str(storage_path)},
                "runtime": {"log_level": "WARNING", "fail_fast": False, "profile_name": "research"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path), "run-full-pipeline", "--count", "8"]) == 0

    connection = sqlite3.connect(storage_path)
    try:
        run_row = connection.execute(
            """
            SELECT profile_name, dataset_fingerprint, selected_timeframe, regime_key, entry_command
            FROM runs
            LIMIT 1
            """
        ).fetchone()
        selection_row = connection.execute(
            "SELECT ranking_rationale_json FROM selections LIMIT 1"
        ).fetchone()
        history_row = connection.execute(
            "SELECT rejection_reasons_json FROM alpha_history LIMIT 1"
        ).fetchone()
    finally:
        connection.close()

    assert run_row is not None
    assert run_row[0] == "research"
    assert run_row[1]
    assert run_row[2] == "1d"
    assert run_row[3]
    assert run_row[4] == "run-full-pipeline"
    assert selection_row is not None and selection_row[0]
    assert history_row is not None and history_row[0]
