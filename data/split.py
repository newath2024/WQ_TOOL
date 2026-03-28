from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.config import PeriodConfig, SplitConfig
from data.schema import MarketDataBundle


@dataclass(slots=True)
class SplitBundle:
    train: MarketDataBundle
    validation: MarketDataBundle
    test: MarketDataBundle


def slice_frame_by_period(frame: pd.DataFrame, period: PeriodConfig) -> pd.DataFrame:
    if frame.empty or "timestamp" not in frame.columns:
        return frame.copy()
    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end)
    mask = (frame["timestamp"] >= start) & (frame["timestamp"] <= end)
    return frame.loc[mask].copy()


def split_market_data(bundle: MarketDataBundle, split_config: SplitConfig) -> SplitBundle:
    def split_bundle(period: PeriodConfig) -> MarketDataBundle:
        return MarketDataBundle(
            prices={timeframe: slice_frame_by_period(frame, period) for timeframe, frame in bundle.prices.items()},
            groups={timeframe: slice_frame_by_period(frame, period) for timeframe, frame in bundle.groups.items()},
            factors={timeframe: slice_frame_by_period(frame, period) for timeframe, frame in bundle.factors.items()},
            masks={timeframe: slice_frame_by_period(frame, period) for timeframe, frame in bundle.masks.items()},
            source_path=bundle.source_path,
            aux_source_paths=dict(bundle.aux_source_paths),
            fingerprint=bundle.fingerprint,
        )

    return SplitBundle(
        train=split_bundle(split_config.train),
        validation=split_bundle(split_config.validation),
        test=split_bundle(split_config.test),
    )
