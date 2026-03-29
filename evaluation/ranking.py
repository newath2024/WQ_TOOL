from __future__ import annotations

from typing import Iterable


def evaluation_sort_key(evaluation) -> tuple[float, int, float, float, int, str]:
    validation = evaluation.split_metrics["validation"]
    return (
        -validation.fitness,
        -evaluation.submission_passes,
        -validation.sharpe,
        -evaluation.behavioral_novelty_score,
        evaluation.candidate.complexity,
        evaluation.candidate.normalized_expression,
    )


def rank_evaluations(evaluations: Iterable) -> list:
    return sorted(evaluations, key=evaluation_sort_key)
