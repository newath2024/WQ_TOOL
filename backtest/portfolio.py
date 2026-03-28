from __future__ import annotations

import pandas as pd

from features.operators import ts_rank


def cross_sectional_bucket_weights(
    scores: pd.DataFrame,
    selection_fraction: float,
    construction: str,
) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    bucket_fraction = min(max(selection_fraction, 0.01), 0.5)

    for timestamp, row in scores.iterrows():
        valid = row.dropna().sort_values()
        if valid.empty:
            continue
        bucket_size = max(1, int(len(valid) * bucket_fraction))
        longs = valid.iloc[-bucket_size:].index
        if construction == "long_only":
            weights.loc[timestamp, longs] = 1.0 / bucket_size
            continue
        shorts = valid.iloc[:bucket_size].index
        weights.loc[timestamp, longs] = 0.5 / bucket_size
        weights.loc[timestamp, shorts] = -0.5 / bucket_size

    return weights


def symbol_level_weights(
    scores: pd.DataFrame,
    construction: str,
    rank_window: int,
    upper_quantile: float,
    lower_quantile: float,
) -> pd.DataFrame:
    rolling_ranks = ts_rank(scores, rank_window)
    signals = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    signals = signals.mask(rolling_ranks >= upper_quantile, 1.0)
    if construction == "long_short":
        signals = signals.mask(rolling_ranks <= lower_quantile, -1.0)

    if construction == "long_only":
        denominators = signals.where(signals > 0).sum(axis=1).replace(0, pd.NA)
        return signals.where(signals > 0, 0.0).div(denominators, axis=0).fillna(0.0)

    denominators = signals.abs().sum(axis=1).replace(0, pd.NA)
    return signals.div(denominators, axis=0).fillna(0.0)


def apply_holding_period(weights: pd.DataFrame, holding_period: int) -> pd.DataFrame:
    if holding_period <= 1:
        return weights.fillna(0.0)
    vintages = [weights.shift(offset).fillna(0.0) for offset in range(holding_period)]
    return sum(vintages) / float(holding_period)
