from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data.schema import TimeframeData


@dataclass(slots=True)
class ResearchMatrices:
    numeric_fields: dict[str, pd.DataFrame]
    group_fields: dict[str, pd.DataFrame] = field(default_factory=dict)
    factor_fields: dict[str, pd.DataFrame] = field(default_factory=dict)
    mask_fields: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def fields(self) -> dict[str, pd.DataFrame]:
        return self.numeric_fields

    def __getitem__(self, field: str) -> pd.DataFrame:
        return self.numeric_fields[field]


def pivot_ohlcv_fields(
    frame: pd.DataFrame,
    groups: pd.DataFrame | None = None,
    factors: pd.DataFrame | None = None,
    masks: pd.DataFrame | None = None,
) -> ResearchMatrices:
    matrices = {
        column: frame.pivot(index="timestamp", columns="symbol", values=column).sort_index()
        for column in ("open", "high", "low", "close", "volume")
    }
    matrices["returns"] = matrices["close"].pct_change()
    return ResearchMatrices(
        numeric_fields=matrices,
        group_fields=_pivot_auxiliary_fields(groups),
        factor_fields=_pivot_auxiliary_fields(factors),
        mask_fields=_pivot_auxiliary_fields(masks),
    )


def build_research_matrices(timeframe_data: TimeframeData) -> ResearchMatrices:
    return pivot_ohlcv_fields(
        frame=timeframe_data.prices,
        groups=timeframe_data.groups if not timeframe_data.groups.empty else None,
        factors=timeframe_data.factors if not timeframe_data.factors.empty else None,
        masks=timeframe_data.masks if not timeframe_data.masks.empty else None,
    )


def _pivot_auxiliary_fields(frame: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if frame is None or frame.empty:
        return {}
    columns = [column for column in frame.columns if column not in {"timestamp", "symbol"}]
    return {
        column: frame.pivot(index="timestamp", columns="symbol", values=column).sort_index()
        for column in columns
    }
