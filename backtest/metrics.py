from __future__ import annotations

import numpy as np
import pandas as pd

from domain.metrics import PerformanceMetrics


def compute_performance_metrics(
    daily_returns: pd.Series,
    turnover: pd.Series,
    annualization_factor: int,
    turnover_penalty: float,
    drawdown_penalty: float,
) -> PerformanceMetrics:
    clean_returns = daily_returns.dropna()
    clean_turnover = turnover.reindex(clean_returns.index).fillna(0.0)
    observation_count = int(clean_returns.shape[0])

    average_return = float(clean_returns.mean()) if observation_count else 0.0
    volatility = float(clean_returns.std(ddof=0)) if observation_count else 0.0
    sharpe = (
        float(np.sqrt(annualization_factor) * average_return / volatility)
        if observation_count and volatility > 0
        else 0.0
    )

    equity = (1.0 + clean_returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity.div(running_max).sub(1.0)
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    win_rate = float((clean_returns > 0).mean()) if observation_count else 0.0
    turnover_value = float(clean_turnover.mean()) if not clean_turnover.empty else 0.0
    cumulative_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
    fitness = sharpe - turnover_penalty * turnover_value - drawdown_penalty * abs(max_drawdown)

    return PerformanceMetrics(
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        average_return=average_return,
        turnover=turnover_value,
        observation_count=observation_count,
        cumulative_return=cumulative_return,
        fitness=fitness,
    )
