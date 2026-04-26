from __future__ import annotations

import math

from domain.metrics import PerformanceMetrics


def compute_stability_score(train: PerformanceMetrics, validation: PerformanceMetrics) -> float:
    if math.isnan(train.sharpe) or math.isnan(validation.sharpe):
        return 0.0
    if train.sharpe == 0 or validation.sharpe == 0:
        return 0.0
    if train.sharpe * validation.sharpe <= 0:
        return 0.0
    denominator = max(abs(train.sharpe), abs(validation.sharpe), 1e-9)
    gap = abs(train.sharpe - validation.sharpe) / denominator
    return max(0.0, 1.0 - gap)
