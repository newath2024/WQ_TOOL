from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
import yaml

from core.config import AuxDataConfig, DataConfig
from core.logging import configure_logging
from data.loader import load_market_data
from generator.engine import AlphaCandidate
from main import build_alpha_simulation_signature, main
from tests.conftest import build_static_metadata_frame, write_sample_csv, write_sample_metadata_csv


def test_loader_aligns_auxiliary_metadata(tmp_path: Path) -> None:
    data_path = tmp_path / "daily_ohlcv.csv"
    metadata_path = tmp_path / "symbol_metadata.csv"
    write_sample_csv(data_path)
    write_sample_metadata_csv(metadata_path)
    configure_logging("WARNING")

    bundle = load_market_data(
        DataConfig(path=str(data_path)),
        logger=__import__("logging").getLogger("test-loader"),
        aux_config=AuxDataConfig(
            group_path=str(metadata_path),
            factor_path=str(metadata_path),
            mask_path=str(metadata_path),
            group_columns=["sector", "industry", "country", "subindustry"],
            factor_columns=["beta", "size", "volatility", "liquidity"],
            mask_columns=["core_mask", "liquid_mask"],
        ),
    )

    timeframe = bundle.get_timeframe_data("1d")
    assert len(timeframe.groups) == len(timeframe.prices)
    assert len(timeframe.factors) == len(timeframe.prices)
    assert len(timeframe.masks) == len(timeframe.prices)
    assert {"sector", "industry", "country", "subindustry"} <= set(timeframe.groups.columns)
    assert bundle.fingerprint


def test_loader_rejects_missing_auxiliary_symbol(tmp_path: Path) -> None:
    data_path = tmp_path / "daily_ohlcv.csv"
    metadata_path = tmp_path / "symbol_metadata.csv"
    write_sample_csv(data_path)
    broken = build_static_metadata_frame().query("symbol != 'DDD'")
    broken.to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="missing"):
        load_market_data(
            DataConfig(path=str(data_path)),
            logger=__import__("logging").getLogger("test-loader-fail"),
            aux_config=AuxDataConfig(
                group_path=str(metadata_path),
                group_columns=["sector", "industry", "country", "subindustry"],
            ),
        )


def test_simulation_signature_changes_with_submission_config(tmp_path: Path) -> None:
    data_path = tmp_path / "daily_ohlcv.csv"
    metadata_path = tmp_path / "symbol_metadata.csv"
    config_path = tmp_path / "config.yaml"
    write_sample_csv(data_path)
    write_sample_metadata_csv(metadata_path)

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
            "group_columns": ["sector", "industry", "country", "subindustry"],
            "factor_columns": ["beta", "size", "volatility", "liquidity"],
            "mask_columns": ["core_mask", "liquid_mask"],
        },
        "splits": {
            "train": {"start": "2021-01-01", "end": "2021-02-12"},
            "validation": {"start": "2021-02-15", "end": "2021-03-05"},
            "test": {"start": "2021-03-08", "end": "2021-03-31"},
        },
        "generation": {
            "allowed_fields": ["open", "high", "low", "close", "volume", "returns"],
            "allowed_operators": ["rank", "delta"],
            "lookbacks": [2, 3, 5],
            "max_depth": 4,
            "complexity_limit": 20,
            "template_count": 2,
            "grammar_count": 2,
            "mutation_count": 1,
            "normalization_wrappers": ["rank"],
            "random_seed": 11,
        },
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
            "enable_subuniverse_test": True,
            "enable_ladder_test": True,
            "enable_robustness_test": True,
            "ladder_buckets": 3,
            "ladder_min_sharpe": -1.0,
            "ladder_min_passes": 1,
            "subuniverse_min_sharpe": -1.0,
            "subuniverse_min_pass_fraction": 0.5,
            "robustness_min_fitness_ratio": -5.0,
        },
        "storage": {"path": str(tmp_path / "results.sqlite3")},
        "runtime": {"log_level": "WARNING", "fail_fast": False},
    }
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    from core.config import load_config

    config_a = load_config(config_path)
    config_b = load_config(config_path)
    config_b.submission_tests.subuniverse_min_sharpe = -3.0
    bundle = load_market_data(
        config_a.data,
        logger=__import__("logging").getLogger("test-signature"),
        aux_config=config_a.aux_data,
    )
    candidate = AlphaCandidate(
        alpha_id="alpha-1",
        expression="rank(delta(close, 2))",
        normalized_expression="rank(delta(close, 2))",
        generation_mode="template",
        parent_ids=(),
        complexity=3,
        created_at="2021-01-01T00:00:00+00:00",
    )

    assert build_alpha_simulation_signature(candidate, bundle, config_a) != build_alpha_simulation_signature(
        candidate,
        bundle,
        config_b,
    )


def test_run_full_pipeline_and_resume(tmp_path: Path) -> None:
    data_path = tmp_path / "daily_ohlcv.csv"
    metadata_path = tmp_path / "symbol_metadata.csv"
    config_path = tmp_path / "config.yaml"
    storage_path = tmp_path / "results.sqlite3"
    write_sample_csv(data_path)
    write_sample_metadata_csv(metadata_path)

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
            "lookbacks": [2, 3, 5],
            "max_depth": 4,
            "complexity_limit": 20,
            "template_count": 8,
            "grammar_count": 8,
            "mutation_count": 4,
            "normalization_wrappers": ["rank", "zscore", "sign"],
            "random_seed": 11,
        },
        "simulation": {
            "delay_mode": "fast_d1",
            "neutralization": "sector",
            "secondary_neutralization": "country",
            "pasteurize": True,
            "signal_clip": 5.0,
            "weight_clip": 0.35,
            "factor_columns": ["beta", "size", "volatility", "liquidity"],
            "subuniverses": [
                {"name": "liquid", "mask_field": "liquid_mask"},
                {"name": "large_cap", "top_n_by": "size", "top_n": 2},
            ],
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
            "enable_subuniverse_test": True,
            "enable_ladder_test": True,
            "enable_robustness_test": True,
            "ladder_buckets": 3,
            "ladder_min_sharpe": -1.0,
            "ladder_min_passes": 1,
            "subuniverse_min_sharpe": -1.0,
            "subuniverse_min_pass_fraction": 0.5,
            "robustness_min_fitness_ratio": -5.0,
        },
        "storage": {"path": str(storage_path)},
        "runtime": {"log_level": "INFO", "fail_fast": False},
    }
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    assert main(["--config", str(config_path), "run-full-pipeline"]) == 0
    assert main(["--config", str(config_path), "report", "--limit", "3"]) == 0
    assert main(["--config", str(config_path), "--resume", "run-full-pipeline"]) == 0
    assert main(["--config", str(config_path), "top", "--limit", "3"]) == 0

    connection = sqlite3.connect(storage_path)
    try:
        alpha_count = connection.execute("SELECT COUNT(*) FROM alphas").fetchone()[0]
        metric_count = connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        selection_count = connection.execute("SELECT COUNT(*) FROM selections").fetchone()[0]
        submission_count = connection.execute("SELECT COUNT(*) FROM submission_tests").fetchone()[0]
        validation = pd.read_sql_query(
            "SELECT delay_mode, neutralization, cache_hit, submission_pass_count FROM metrics WHERE split = 'validation'",
            connection,
        )
    finally:
        connection.close()

    assert alpha_count > 0
    assert metric_count >= alpha_count * 3
    assert selection_count > 0
    assert submission_count > 0
    assert (validation["delay_mode"] == "fast_d1").all()
    assert (validation["neutralization"] == "sector").all()
    assert validation["submission_pass_count"].ge(0).all()
    assert validation["cache_hit"].max() == 1
