from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from backtest.metrics import PerformanceMetrics
from core.config import EvaluationConfig
from evaluation.dedup import deduplicate_evaluations
from evaluation.filtering import apply_quality_filters, build_evaluated_alpha
from evaluation.ranking import rank_evaluations
from evaluation.submission import TestResult as SubmissionTestResult
from generator.engine import AlphaCandidate


def make_candidate(alpha_id: str, expression: str, complexity: int = 3) -> AlphaCandidate:
    now = datetime.now(timezone.utc).isoformat()
    return AlphaCandidate(
        alpha_id=alpha_id,
        expression=expression,
        normalized_expression=expression,
        generation_mode="template",
        parent_ids=(),
        complexity=complexity,
        created_at=now,
    )


def make_metrics(sharpe: float, turnover: float = 0.2, obs: int = 20, drawdown: float = -0.1) -> PerformanceMetrics:
    return PerformanceMetrics(
        sharpe=sharpe,
        max_drawdown=drawdown,
        win_rate=0.55,
        average_return=0.01,
        turnover=turnover,
        observation_count=obs,
        cumulative_return=0.2,
        fitness=sharpe - turnover - abs(drawdown),
    )


def test_quality_filters_reject_unstable_alpha() -> None:
    config = EvaluationConfig(
        min_sharpe=0.0,
        max_turnover=1.0,
        min_observations=10,
        max_drawdown=0.5,
        min_stability=0.5,
        signal_correlation_threshold=0.95,
        returns_correlation_threshold=0.95,
        top_k=10,
    )
    validation_signal = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]])
    validation_returns = pd.Series([0.01, -0.01])

    good = build_evaluated_alpha(
        candidate=make_candidate("good", "rank(delta(close,1))"),
        split_metrics={"train": make_metrics(1.2), "validation": make_metrics(1.0), "test": make_metrics(0.9)},
        validation_signal=validation_signal,
        validation_returns=validation_returns,
    )
    bad = build_evaluated_alpha(
        candidate=make_candidate("bad", "rank(delta(volume,1))"),
        split_metrics={"train": make_metrics(1.2), "validation": make_metrics(-0.2), "test": make_metrics(0.1)},
        validation_signal=validation_signal,
        validation_returns=validation_returns,
    )

    passed, rejected = apply_quality_filters([good, bad], config)

    assert len(passed) == 1
    assert len(rejected) == 1
    assert rejected[0].candidate.alpha_id == "bad"


def test_deduplicate_evaluations_keeps_stronger_candidate() -> None:
    validation_signal = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]])
    validation_returns = pd.Series([0.01, -0.01])
    stronger = build_evaluated_alpha(
        candidate=make_candidate("strong", "expr_a", complexity=2),
        split_metrics={"train": make_metrics(1.4), "validation": make_metrics(1.3), "test": make_metrics(1.0)},
        validation_signal=validation_signal,
        validation_returns=validation_returns,
    )
    weaker = build_evaluated_alpha(
        candidate=make_candidate("weak", "expr_b", complexity=5),
        split_metrics={"train": make_metrics(0.9), "validation": make_metrics(0.8), "test": make_metrics(0.7)},
        validation_signal=validation_signal.copy(),
        validation_returns=validation_returns.copy(),
    )

    deduped = deduplicate_evaluations([stronger, weaker], signal_threshold=0.9, returns_threshold=0.9)

    assert len(deduped) == 1
    assert deduped[0].candidate.alpha_id == "strong"


def test_quality_filters_reject_failed_submission_tests() -> None:
    config = EvaluationConfig(
        min_sharpe=-1.0,
        max_turnover=1.0,
        min_observations=5,
        max_drawdown=1.0,
        min_stability=0.0,
        signal_correlation_threshold=0.95,
        returns_correlation_threshold=0.95,
        top_k=10,
    )
    evaluation = build_evaluated_alpha(
        candidate=make_candidate("subfail", "rank(delta(close,1))"),
        split_metrics={"train": make_metrics(1.0), "validation": make_metrics(0.8), "test": make_metrics(0.7)},
        validation_signal=pd.DataFrame([[1.0, 0.0], [0.0, 1.0]]),
        validation_returns=pd.Series([0.01, -0.01]),
        submission_tests={"ladder": SubmissionTestResult(name="ladder", passed=False, details={"passes": 0})},
    )

    passed, rejected = apply_quality_filters([evaluation], config)

    assert not passed
    assert rejected[0].candidate.alpha_id == "subfail"
    assert "submission test 'ladder' failed" in rejected[0].rejection_reasons


def test_ranking_prefers_submission_passes_when_fitness_matches() -> None:
    validation_signal = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]])
    validation_returns = pd.Series([0.01, -0.01])
    common_metrics = {"train": make_metrics(1.1), "validation": make_metrics(1.0), "test": make_metrics(0.9)}
    stronger = build_evaluated_alpha(
        candidate=make_candidate("passes_more", "expr_a", complexity=4),
        split_metrics=common_metrics,
        validation_signal=validation_signal,
        validation_returns=validation_returns,
        submission_tests={
            "subuniverse": SubmissionTestResult(name="subuniverse", passed=True, details={}),
            "ladder": SubmissionTestResult(name="ladder", passed=True, details={}),
        },
    )
    weaker = build_evaluated_alpha(
        candidate=make_candidate("passes_less", "expr_b", complexity=2),
        split_metrics=common_metrics,
        validation_signal=validation_signal,
        validation_returns=validation_returns,
        submission_tests={"subuniverse": SubmissionTestResult(name="subuniverse", passed=True, details={})},
    )

    ranked = rank_evaluations([weaker, stronger])

    assert ranked[0].candidate.alpha_id == "passes_more"
