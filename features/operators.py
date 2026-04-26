from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd


FrameLike: TypeAlias = pd.DataFrame | pd.Series
NumericLike: TypeAlias = FrameLike | float | int
EPSILON = 1e-12


def _coerce_window(window: int | float) -> int:
    return int(window)


def safe_divide(left: NumericLike, right: NumericLike) -> NumericLike:
    if isinstance(right, (int, float)):
        if abs(float(right)) <= EPSILON:
            if isinstance(left, (pd.DataFrame, pd.Series)):
                return left * np.nan
            return np.nan
        return left / right
    cleaned = right.where(right.abs() > EPSILON)
    return left / cleaned


def delay(values: NumericLike, window: int) -> NumericLike:
    normalized = _coerce_window(window)
    return values.shift(normalized) if isinstance(values, (pd.DataFrame, pd.Series)) else values


def delta(values: NumericLike, window: int) -> NumericLike:
    return values - delay(values, window)


def returns(values: NumericLike, window: int = 1) -> NumericLike:
    return safe_divide(values, delay(values, window)) - 1.0


def rolling_mean(values: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    return values.rolling(window=normalized, min_periods=normalized).mean()


def rolling_std(values: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    return values.rolling(window=normalized, min_periods=normalized).std(ddof=0)


def rolling_min(values: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    return values.rolling(window=normalized, min_periods=normalized).min()


def rolling_max(values: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    return values.rolling(window=normalized, min_periods=normalized).max()


def rank(values: FrameLike) -> FrameLike:
    if isinstance(values, pd.Series):
        return values.rank(pct=True)
    return values.rank(axis=1, pct=True)


def zscore(values: FrameLike) -> FrameLike:
    if isinstance(values, pd.Series):
        std = values.std(ddof=0)
        if std == 0 or np.isnan(std):
            return values * np.nan
        return (values - values.mean()) / std
    means = values.mean(axis=1)
    stds = values.std(axis=1, ddof=0).replace(0, np.nan)
    return values.sub(means, axis=0).div(stds, axis=0)


def correlation(left: FrameLike, right: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    return left.rolling(window=normalized, min_periods=normalized).corr(right)


def covariance(left: FrameLike, right: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    return left.rolling(window=normalized, min_periods=normalized).cov(right)


def decay_linear(values: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    weights = np.arange(1, normalized + 1, dtype=float)
    weights /= weights.sum()
    return values.rolling(window=normalized, min_periods=normalized).apply(
        lambda row: float(np.dot(row, weights)) if not np.isnan(row).all() else np.nan,
        raw=True,
    )


def ts_rank(values: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    return values.rolling(window=normalized, min_periods=normalized).apply(
        lambda row: pd.Series(row).rank(pct=True).iloc[-1] if not np.isnan(row).all() else np.nan,
        raw=True,
    )


def ts_sum(values: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    return values.rolling(window=normalized, min_periods=normalized).sum()


def ts_mean(values: FrameLike, window: int) -> FrameLike:
    return rolling_mean(values, window)


def ts_std(values: FrameLike, window: int) -> FrameLike:
    return rolling_std(values, window)


def days_from_last_change(values: FrameLike) -> FrameLike:
    def transform(series: pd.Series) -> pd.Series:
        result: list[float] = []
        last_change_index: int | None = None
        previous = np.nan
        for index, value in enumerate(series):
            if pd.isna(value):
                result.append(np.nan)
                continue
            if index == 0 or pd.isna(previous) or value != previous:
                last_change_index = index
                result.append(0.0)
            else:
                result.append(float(index - (last_change_index if last_change_index is not None else index)))
            previous = value
        return pd.Series(result, index=series.index, dtype=float)

    if isinstance(values, pd.Series):
        return transform(values)
    return values.apply(transform, axis=0)


def ts_av_diff(values: FrameLike, window: int) -> FrameLike:
    return values - rolling_mean(values, window)


def ts_scale(values: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    rolling_low = values.rolling(window=normalized, min_periods=normalized).min()
    rolling_high = values.rolling(window=normalized, min_periods=normalized).max()
    return safe_divide(values - rolling_low, rolling_high - rolling_low)


def ts_arg_max(values: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    return values.rolling(window=normalized, min_periods=normalized).apply(
        lambda row: float(np.nanargmax(row)) if not np.isnan(row).all() else np.nan,
        raw=True,
    )


def ts_arg_min(values: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    return values.rolling(window=normalized, min_periods=normalized).apply(
        lambda row: float(np.nanargmin(row)) if not np.isnan(row).all() else np.nan,
        raw=True,
    )


def quantile(values: FrameLike) -> FrameLike:
    return rank(values)


def inverse(values: NumericLike) -> NumericLike:
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return safe_divide(1.0, values)
    return np.nan if abs(float(values)) <= EPSILON else 1.0 / float(values)


def reverse(values: NumericLike) -> NumericLike:
    return -values


def ts_count_nans(values: FrameLike, window: int) -> FrameLike:
    normalized = _coerce_window(window)
    return values.isna().astype(float).rolling(window=normalized, min_periods=normalized).sum()


def min_value(left: NumericLike, right: NumericLike) -> NumericLike:
    return np.minimum(left, right)


def max_value(left: NumericLike, right: NumericLike) -> NumericLike:
    return np.maximum(left, right)


def sign(values: NumericLike) -> NumericLike:
    return np.sign(values)


def abs_value(values: NumericLike) -> NumericLike:
    return np.abs(values)


def log(values: NumericLike) -> NumericLike:
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return np.log(values.where(values > 0))
    return float(np.log(values)) if float(values) > 0 else np.nan


def clip(values: NumericLike, lower: float, upper: float) -> NumericLike:
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.clip(lower=lower, upper=upper)
    return float(np.clip(values, lower, upper))


def group_rank(values: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    return _group_transform(values, groups, lambda frame: frame.groupby("group")["value"].rank(pct=True))


def group_zscore(values: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    def transform(frame: pd.DataFrame) -> pd.Series:
        grouped = frame.groupby("group")["value"]
        means = grouped.transform("mean")
        stds = grouped.transform(lambda series: series.std(ddof=0)).replace(0, np.nan)
        return (frame["value"] - means) / stds

    return _group_transform(values, groups, transform)


def group_neutralize(values: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    return _group_transform(values, groups, lambda frame: frame["value"] - frame.groupby("group")["value"].transform("mean"))


def _group_transform(
    values: pd.DataFrame,
    groups: pd.DataFrame,
    transform,
) -> pd.DataFrame:
    aligned_groups = groups.reindex(index=values.index, columns=values.columns)
    result = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)

    for timestamp in values.index:
        value_row = values.loc[timestamp]
        group_row = aligned_groups.loc[timestamp]
        frame = pd.DataFrame({"value": value_row, "group": group_row}).dropna(subset=["value", "group"])
        if frame.empty:
            continue
        transformed = transform(frame)
        result.loc[timestamp, transformed.index] = transformed.astype(float).to_numpy()

    return result
