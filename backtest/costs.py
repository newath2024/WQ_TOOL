from __future__ import annotations

import pandas as pd


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    shifted = weights.shift(1).fillna(0.0)
    turnover = weights.fillna(0.0).sub(shifted, fill_value=0.0).abs().sum(axis=1)
    return turnover


def transaction_costs(turnover: pd.Series, cost_bps: float) -> pd.Series:
    return turnover * (cost_bps / 10000.0)
