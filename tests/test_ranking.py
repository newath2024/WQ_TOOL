from __future__ import annotations

import pandas as pd

from backtest.metrics import PerformanceMetrics
from evaluation.filtering import build_evaluated_alpha
from evaluation.ranking import rank_evaluations
from generator.engine import AlphaCandidate


def make_metrics(fitness: float, sharpe: float) -> PerformanceMetrics:
    return PerformanceMetrics(
        sharpe=sharpe,
        max_drawdown=-0.1,
        win_rate=0.55,
        average_return=0.01,
        turnover=0.5,
        observation_count=20,
        cumulative_return=0.15,
        fitness=fitness,
    )


def make_alpha(alpha_id: str, complexity: int) -> AlphaCandidate:
    return AlphaCandidate(
        alpha_id=alpha_id,
        expression=f"rank(delta(close, {complexity}))",
        normalized_expression=f"rank(delta(close,{complexity}))",
        generation_mode="template",
        parent_ids=(),
        complexity=complexity,
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_ranking_prefers_behavioral_novelty_before_complexity() -> None:
    base_metrics = {
        "train": make_metrics(1.0, 1.0),
        "validation": make_metrics(1.0, 1.0),
        "test": make_metrics(1.0, 1.0),
    }
    first = build_evaluated_alpha(
        candidate=make_alpha("alpha-a", complexity=8),
        split_metrics=base_metrics,
        validation_signal=pd.DataFrame({"AAA": [1.0, 2.0]}),
        validation_returns=pd.Series([0.01, 0.02]),
    )
    first.submission_passes = 2
    first.behavioral_novelty_score = 0.25

    second = build_evaluated_alpha(
        candidate=make_alpha("alpha-b", complexity=10),
        split_metrics=base_metrics,
        validation_signal=pd.DataFrame({"AAA": [1.0, 2.0]}),
        validation_returns=pd.Series([0.01, 0.02]),
    )
    second.submission_passes = 2
    second.behavioral_novelty_score = 0.75

    ranked = rank_evaluations([first, second])

    assert ranked[0].candidate.alpha_id == "alpha-b"
