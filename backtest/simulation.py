from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from core.config import BacktestConfig, SimulationConfig, SubuniverseConfig
from features.transforms import ResearchMatrices


@dataclass(frozen=True, slots=True)
class SimulationProfile:
    delay_mode: str
    effective_signal_delay: int
    effective_holding_period: int
    neutralization: str
    secondary_neutralization: str | None
    pasteurize: bool
    signal_clip: float | None
    weight_clip: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_simulation_profile(
    simulation_config: SimulationConfig,
    backtest_config: BacktestConfig,
) -> SimulationProfile:
    if simulation_config.delay_mode == "d0":
        signal_delay = 0
        holding_period = backtest_config.holding_period
    elif simulation_config.delay_mode == "fast_d1":
        signal_delay = 1
        holding_period = 1
    else:
        signal_delay = 1 if backtest_config.signal_delay >= 1 else 0
        holding_period = backtest_config.holding_period

    return SimulationProfile(
        delay_mode=simulation_config.delay_mode,
        effective_signal_delay=signal_delay,
        effective_holding_period=holding_period,
        neutralization=simulation_config.neutralization,
        secondary_neutralization=simulation_config.secondary_neutralization,
        pasteurize=simulation_config.pasteurize,
        signal_clip=simulation_config.signal_clip,
        weight_clip=simulation_config.weight_clip,
    )


def apply_signal_controls(
    signal: pd.DataFrame,
    close_prices: pd.DataFrame,
    profile: SimulationProfile,
) -> pd.DataFrame:
    cleaned = signal.replace([np.inf, -np.inf], np.nan)
    if profile.pasteurize:
        cleaned = cleaned.where(close_prices.notna())
    if profile.signal_clip is not None:
        cleaned = cleaned.clip(lower=-abs(profile.signal_clip), upper=abs(profile.signal_clip))
    return cleaned


def apply_weight_clip(weights: pd.DataFrame, clip_value: float | None) -> pd.DataFrame:
    if clip_value is None:
        return weights
    clipped = weights.clip(lower=-abs(clip_value), upper=abs(clip_value))
    gross = clipped.abs().sum(axis=1).replace(0.0, pd.NA)
    return clipped.div(gross, axis=0).fillna(0.0)


def build_subuniverse_masks(
    matrices: ResearchMatrices,
    subuniverses: list[SubuniverseConfig],
) -> dict[str, pd.DataFrame]:
    masks: dict[str, pd.DataFrame] = {}
    for subuniverse in subuniverses:
        if subuniverse.mask_field:
            mask_source = _lookup_field(matrices, subuniverse.mask_field)
            if mask_source is None:
                raise KeyError(f"Unknown subuniverse mask field '{subuniverse.mask_field}'.")
            masks[subuniverse.name] = mask_source.astype(float).fillna(0.0) > 0
            continue
        if subuniverse.top_n_by:
            source = _lookup_field(matrices, subuniverse.top_n_by)
            if source is None:
                raise KeyError(f"Unknown subuniverse ranking field '{subuniverse.top_n_by}'.")
            if subuniverse.top_n is None or subuniverse.top_n <= 0:
                raise ValueError(f"Subuniverse '{subuniverse.name}' requires a positive top_n value.")
            masks[subuniverse.name] = top_n_mask(source, subuniverse.top_n)
            continue
        raise ValueError(f"Subuniverse '{subuniverse.name}' requires either mask_field or top_n_by.")
    return masks


def top_n_mask(source: pd.DataFrame, top_n: int) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=source.index, columns=source.columns)
    for timestamp, row in source.iterrows():
        valid = row.dropna().sort_values(ascending=False)
        if valid.empty:
            continue
        selected = valid.head(top_n).index
        mask.loc[timestamp, selected] = True
    return mask


def _lookup_field(matrices: ResearchMatrices, name: str) -> pd.DataFrame | None:
    for mapping in (matrices.mask_fields, matrices.factor_fields, matrices.numeric_fields):
        field = mapping.get(name)
        if field is not None:
            return field
    return None
