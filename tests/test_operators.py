from __future__ import annotations

import numpy as np
import pandas as pd

from features.operators import delay, delta, group_neutralize, group_rank, group_zscore, rank, returns, safe_divide, ts_mean, zscore


def test_delay_delta_and_returns(sample_wide_fields) -> None:
    close = sample_wide_fields["close"]
    delayed = delay(close, 1)
    diff = delta(close, 1)
    pct = returns(close, 1)

    assert delayed.iloc[1, 0] == close.iloc[0, 0]
    assert diff.iloc[1, 0] == close.iloc[1, 0] - close.iloc[0, 0]
    assert np.isclose(pct.iloc[1, 0], (close.iloc[1, 0] / close.iloc[0, 0]) - 1.0)


def test_safe_divide_rank_zscore_and_ts_mean(sample_wide_fields) -> None:
    close = sample_wide_fields["close"]
    divided = safe_divide(close, 0.0)
    ranked = rank(close)
    normalized = zscore(close)
    averaged = ts_mean(close, 3)

    assert divided.isna().all().all()
    assert np.isclose(ranked.iloc[0].max(), 1.0)
    assert np.isclose(normalized.iloc[10].mean(skipna=True), 0.0)
    assert averaged.iloc[:2].isna().all().all()


def test_group_operators_respect_group_boundaries() -> None:
    index = pd.date_range("2021-01-01", periods=2, freq="B")
    values = pd.DataFrame(
        [[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]],
        index=index,
        columns=["AAA", "BBB", "CCC", "DDD"],
    )
    groups = pd.DataFrame(
        [["tech", "tech", "fin", "fin"], ["tech", "tech", "fin", "fin"]],
        index=index,
        columns=values.columns,
    )

    ranked = group_rank(values, groups)
    normalized = group_zscore(values, groups)
    neutralized = group_neutralize(values, groups)

    assert np.isclose(ranked.loc[index[0], "BBB"], 1.0)
    assert np.isclose(ranked.loc[index[0], "AAA"], 0.5)
    assert np.isclose(normalized.loc[index[0], ["AAA", "BBB"]].mean(), 0.0)
    assert np.isclose(normalized.loc[index[0], ["CCC", "DDD"]].mean(), 0.0)
    assert np.isclose(neutralized.loc[index[1], ["AAA", "BBB"]].mean(), 0.0)
    assert np.isclose(neutralized.loc[index[1], ["CCC", "DDD"]].mean(), 0.0)
