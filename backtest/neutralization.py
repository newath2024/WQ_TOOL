from __future__ import annotations

import numpy as np
import pandas as pd

from features.operators import group_neutralize
from features.transforms import ResearchMatrices


def apply_neutralization(
    signal: pd.DataFrame,
    mode: str,
    matrices: ResearchMatrices,
    factor_columns: list[str] | None = None,
) -> pd.DataFrame:
    if mode == "none":
        return signal
    if mode == "market":
        return market_neutralize(signal)
    if mode in {"sector", "industry", "country", "subindustry"}:
        return group_neutralize(signal, matrices.group_fields[mode])
    if mode == "factor_model":
        columns = factor_columns or list(matrices.factor_fields)
        if not columns:
            raise ValueError("Factor model neutralization requested but no factor columns are available.")
        exposures = {
            name: matrices.factor_fields.get(name) or matrices.numeric_fields.get(name)
            for name in columns
        }
        missing = [name for name, frame in exposures.items() if frame is None]
        if missing:
            raise KeyError(f"Unknown factor neutralization fields: {missing}")
        return factor_neutralize(signal, {name: frame for name, frame in exposures.items() if frame is not None})
    if mode == "sector_then_country":
        sector_neutral = group_neutralize(signal, matrices.group_fields["sector"])
        return group_neutralize(sector_neutral, matrices.group_fields["country"])
    raise ValueError(f"Unsupported neutralization mode '{mode}'.")


def market_neutralize(signal: pd.DataFrame) -> pd.DataFrame:
    return signal.sub(signal.mean(axis=1), axis=0)


def factor_neutralize(
    signal: pd.DataFrame,
    exposures: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    residuals = pd.DataFrame(index=signal.index, columns=signal.columns, dtype=float)
    exposure_names = list(exposures)

    for timestamp in signal.index:
        y = signal.loc[timestamp]
        x_frames = [exposures[name].reindex(index=signal.index, columns=signal.columns).loc[timestamp] for name in exposure_names]
        frame = pd.DataFrame({"y": y})
        for name, exposure_row in zip(exposure_names, x_frames, strict=True):
            frame[name] = exposure_row
        frame = frame.dropna()
        if frame.shape[0] <= len(exposure_names):
            continue
        x = frame[exposure_names].to_numpy(dtype=float)
        x = np.column_stack([np.ones(frame.shape[0], dtype=float), x])
        y_values = frame["y"].to_numpy(dtype=float)
        betas, _, _, _ = np.linalg.lstsq(x, y_values, rcond=None)
        fitted = x @ betas
        frame["residual"] = y_values - fitted
        residuals.loc[timestamp, frame.index] = frame["residual"].to_numpy()

    return residuals
