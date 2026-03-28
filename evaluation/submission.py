from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtest.metrics import PerformanceMetrics, compute_performance_metrics
from core.config import AppConfig


@dataclass(frozen=True, slots=True)
class TestResult:
    name: str
    passed: bool
    details: dict[str, Any]


def build_submission_tests(
    validation_metrics: PerformanceMetrics,
    combined_returns: pd.Series,
    combined_turnover: pd.Series,
    subuniverse_metrics: dict[str, PerformanceMetrics],
    config: AppConfig,
) -> dict[str, TestResult]:
    tests: dict[str, TestResult] = {}
    submission = config.submission_tests
    if submission.enable_subuniverse_test:
        tests["subuniverse"] = run_subuniverse_test(subuniverse_metrics, config)
    if submission.enable_ladder_test:
        tests["ladder"] = run_ladder_test(combined_returns, combined_turnover, config)
    if submission.enable_robustness_test:
        tests["robustness"] = run_robustness_test(validation_metrics, subuniverse_metrics, config)
    return tests


def run_subuniverse_test(
    subuniverse_metrics: dict[str, PerformanceMetrics],
    config: AppConfig,
) -> TestResult:
    if not subuniverse_metrics:
        return TestResult(name="subuniverse", passed=True, details={"reason": "no_subuniverses_configured"})
    minimum = config.submission_tests.subuniverse_min_sharpe
    passes = {
        name: metric.sharpe >= minimum and metric.observation_count >= config.evaluation.min_observations
        for name, metric in subuniverse_metrics.items()
    }
    pass_fraction = sum(passes.values()) / max(len(passes), 1)
    passed = pass_fraction >= config.submission_tests.subuniverse_min_pass_fraction
    return TestResult(
        name="subuniverse",
        passed=passed,
        details={
            "min_sharpe": minimum,
            "pass_fraction": pass_fraction,
            "required_pass_fraction": config.submission_tests.subuniverse_min_pass_fraction,
            "per_subuniverse": {
                name: {
                    "sharpe": metric.sharpe,
                    "fitness": metric.fitness,
                    "passed": passes[name],
                }
                for name, metric in subuniverse_metrics.items()
            },
        },
    )


def run_ladder_test(
    combined_returns: pd.Series,
    combined_turnover: pd.Series,
    config: AppConfig,
) -> TestResult:
    if combined_returns.dropna().empty:
        return TestResult(name="ladder", passed=False, details={"reason": "empty_combined_returns"})
    index_buckets = [bucket for bucket in np.array_split(combined_returns.dropna().index.to_numpy(), config.submission_tests.ladder_buckets) if len(bucket) > 0]
    bucket_metrics: list[dict[str, Any]] = []
    passes = 0
    for bucket_index, bucket in enumerate(index_buckets, start=1):
        bucket_returns = combined_returns.loc[list(bucket)]
        bucket_turnover = combined_turnover.reindex(bucket_returns.index).fillna(0.0)
        metrics = compute_performance_metrics(
            daily_returns=bucket_returns,
            turnover=bucket_turnover,
            annualization_factor=config.backtest.annualization_factor,
            turnover_penalty=config.backtest.turnover_penalty,
            drawdown_penalty=config.backtest.drawdown_penalty,
        )
        bucket_pass = metrics.sharpe >= config.submission_tests.ladder_min_sharpe
        passes += int(bucket_pass)
        bucket_metrics.append(
            {
                "bucket": bucket_index,
                "sharpe": metrics.sharpe,
                "fitness": metrics.fitness,
                "passed": bucket_pass,
            }
        )
    passed = passes >= config.submission_tests.ladder_min_passes
    return TestResult(
        name="ladder",
        passed=passed,
        details={
            "passes": passes,
            "required_passes": config.submission_tests.ladder_min_passes,
            "min_sharpe": config.submission_tests.ladder_min_sharpe,
            "buckets": bucket_metrics,
        },
    )


def run_robustness_test(
    validation_metrics: PerformanceMetrics,
    subuniverse_metrics: dict[str, PerformanceMetrics],
    config: AppConfig,
) -> TestResult:
    if not subuniverse_metrics:
        return TestResult(name="robustness", passed=True, details={"reason": "no_subuniverses_configured"})
    denominator = max(abs(validation_metrics.fitness), 1e-9)
    ratios = {name: metric.fitness / denominator for name, metric in subuniverse_metrics.items()}
    min_ratio = min(ratios.values())
    passed = min_ratio >= config.submission_tests.robustness_min_fitness_ratio
    return TestResult(
        name="robustness",
        passed=passed,
        details={
            "main_validation_fitness": validation_metrics.fitness,
            "required_min_ratio": config.submission_tests.robustness_min_fitness_ratio,
            "observed_min_ratio": min_ratio,
            "per_subuniverse_ratio": ratios,
        },
    )


def submission_pass_count(test_results: dict[str, TestResult]) -> int:
    return sum(int(result.passed) for result in test_results.values())
