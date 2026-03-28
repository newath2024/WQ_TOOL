from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtest.metrics import PerformanceMetrics
from core.config import EvaluationConfig
from evaluation.critic import AlphaDiagnosis
from evaluation.stability import compute_stability_score
from evaluation.submission import TestResult, submission_pass_count
from generator.engine import AlphaCandidate
from memory.pattern_memory import StructuralSignature


@dataclass(slots=True)
class EvaluatedAlpha:
    candidate: AlphaCandidate
    split_metrics: dict[str, PerformanceMetrics]
    stability_score: float
    validation_signal: pd.DataFrame
    validation_returns: pd.Series
    simulation_signature: str
    regime_key: str = ""
    submission_tests: dict[str, TestResult] = field(default_factory=dict)
    subuniverse_metrics: dict[str, PerformanceMetrics] = field(default_factory=dict)
    cache_hit: bool = False
    simulation_profile: dict = field(default_factory=dict)
    diagnosis: AlphaDiagnosis = field(default_factory=AlphaDiagnosis)
    behavioral_novelty_score: float = 0.5
    structural_signature: StructuralSignature | None = None
    gene_ids: list[str] = field(default_factory=list)
    outcome_score: float = 0.0
    passed_filters: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    submission_passes: int = 0


def build_evaluated_alpha(
    candidate: AlphaCandidate,
    split_metrics: dict[str, PerformanceMetrics],
    validation_signal: pd.DataFrame,
    validation_returns: pd.Series,
    simulation_signature: str = "",
    regime_key: str = "",
    submission_tests: dict[str, TestResult] | None = None,
    subuniverse_metrics: dict[str, PerformanceMetrics] | None = None,
    cache_hit: bool = False,
    simulation_profile: dict | None = None,
    diagnosis: AlphaDiagnosis | None = None,
    behavioral_novelty_score: float = 0.5,
    structural_signature: StructuralSignature | None = None,
    gene_ids: list[str] | None = None,
) -> EvaluatedAlpha:
    tests = submission_tests or {}
    stability_score = compute_stability_score(split_metrics["train"], split_metrics["validation"])
    return EvaluatedAlpha(
        candidate=candidate,
        split_metrics=split_metrics,
        stability_score=stability_score,
        validation_signal=validation_signal,
        validation_returns=validation_returns,
        simulation_signature=simulation_signature,
        regime_key=regime_key,
        submission_tests=tests,
        subuniverse_metrics=subuniverse_metrics or {},
        cache_hit=cache_hit,
        simulation_profile=simulation_profile or {},
        diagnosis=diagnosis or AlphaDiagnosis(),
        behavioral_novelty_score=behavioral_novelty_score,
        structural_signature=structural_signature,
        gene_ids=gene_ids or [],
        submission_passes=submission_pass_count(tests),
    )


def apply_quality_filters(
    evaluations: list[EvaluatedAlpha],
    config: EvaluationConfig,
) -> tuple[list[EvaluatedAlpha], list[EvaluatedAlpha]]:
    passed: list[EvaluatedAlpha] = []
    rejected: list[EvaluatedAlpha] = []
    for evaluation in evaluations:
        validation = evaluation.split_metrics["validation"]
        reasons: list[str] = []
        if validation.sharpe < config.min_sharpe:
            reasons.append(f"validation sharpe {validation.sharpe:.4f} below minimum {config.min_sharpe:.4f}")
        if validation.turnover > config.max_turnover:
            reasons.append(f"validation turnover {validation.turnover:.4f} above maximum {config.max_turnover:.4f}")
        if validation.observation_count < config.min_observations:
            reasons.append(
                f"validation observations {validation.observation_count} below minimum {config.min_observations}"
            )
        if abs(validation.max_drawdown) > config.max_drawdown:
            reasons.append(
                f"validation max drawdown {abs(validation.max_drawdown):.4f} above maximum {config.max_drawdown:.4f}"
            )
        if evaluation.stability_score < config.min_stability:
            reasons.append(
                f"train/validation stability {evaluation.stability_score:.4f} below minimum {config.min_stability:.4f}"
            )
        for test_name, result in evaluation.submission_tests.items():
            if not result.passed:
                reasons.append(f"submission test '{test_name}' failed")

        evaluation.rejection_reasons = reasons
        evaluation.passed_filters = not reasons
        if evaluation.passed_filters:
            passed.append(evaluation)
        else:
            rejected.append(evaluation)
    return passed, rejected
