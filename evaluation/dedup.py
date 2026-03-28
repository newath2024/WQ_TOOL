from __future__ import annotations

import pandas as pd

from evaluation.ranking import rank_evaluations


def deduplicate_with_diagnostics(
    evaluations: list,
    signal_threshold: float,
    returns_threshold: float,
) -> tuple[list, dict[str, dict[str, float | str]]]:
    best_by_expression = {}
    for evaluation in rank_evaluations(evaluations):
        best_by_expression.setdefault(evaluation.candidate.normalized_expression, evaluation)

    selected: list = []
    duplicate_map: dict[str, dict[str, float | str]] = {}
    for evaluation in rank_evaluations(best_by_expression.values()):
        duplicate = False
        for incumbent in selected:
            signal_corr = signal_correlation(evaluation.validation_signal, incumbent.validation_signal)
            returns_corr = returns_correlation(evaluation.validation_returns, incumbent.validation_returns)
            if abs(signal_corr) >= signal_threshold or abs(returns_corr) >= returns_threshold:
                duplicate = True
                duplicate_map[evaluation.candidate.alpha_id] = {
                    "incumbent_alpha_id": incumbent.candidate.alpha_id,
                    "signal_correlation": float(signal_corr),
                    "returns_correlation": float(returns_corr),
                }
                break
        if not duplicate:
            selected.append(evaluation)
    return selected, duplicate_map


def deduplicate_evaluations(
    evaluations: list,
    signal_threshold: float,
    returns_threshold: float,
) -> list:
    selected, _ = deduplicate_with_diagnostics(
        evaluations=evaluations,
        signal_threshold=signal_threshold,
        returns_threshold=returns_threshold,
    )
    return selected


def signal_correlation(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left_vector = left.stack().rename("left").dropna()
    right_vector = right.stack().rename("right").dropna()
    aligned = pd.concat([left_vector, right_vector], axis=1, join="inner").dropna()
    if aligned.shape[0] < 2:
        return 0.0
    if aligned["left"].std(ddof=0) == 0 or aligned["right"].std(ddof=0) == 0:
        return 0.0
    return float(aligned["left"].corr(aligned["right"]))


def returns_correlation(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner").dropna()
    if aligned.shape[0] < 2:
        return 0.0
    if aligned["left"].std(ddof=0) == 0 or aligned["right"].std(ddof=0) == 0:
        return 0.0
    return float(aligned["left"].corr(aligned["right"]))
