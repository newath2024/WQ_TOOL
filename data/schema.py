from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
OPTIONAL_COLUMNS: tuple[str, ...] = ("timeframe",)
AUXILIARY_KEY_COLUMNS: tuple[str, ...] = ("timestamp", "symbol")
COLUMN_ALIASES: dict[str, str] = {
    "date": "timestamp",
    "datetime": "timestamp",
    "ticker": "symbol",
    "asset": "symbol",
}


@dataclass(slots=True)
class TimeframeData:
    prices: pd.DataFrame
    groups: pd.DataFrame = field(default_factory=pd.DataFrame)
    factors: pd.DataFrame = field(default_factory=pd.DataFrame)
    masks: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(slots=True)
class MarketDataBundle:
    prices: dict[str, pd.DataFrame]
    groups: dict[str, pd.DataFrame] = field(default_factory=dict)
    factors: dict[str, pd.DataFrame] = field(default_factory=dict)
    masks: dict[str, pd.DataFrame] = field(default_factory=dict)
    source_path: str = ""
    aux_source_paths: dict[str, str] = field(default_factory=dict)
    fingerprint: str = ""

    @property
    def frames(self) -> dict[str, pd.DataFrame]:
        return self.prices

    def get_timeframe(self, timeframe: str) -> pd.DataFrame:
        try:
            return self.prices[timeframe]
        except KeyError as exc:
            raise KeyError(f"Timeframe '{timeframe}' is not available.") from exc

    def get_timeframe_data(self, timeframe: str) -> TimeframeData:
        return TimeframeData(
            prices=self.get_timeframe(timeframe),
            groups=self.groups.get(timeframe, pd.DataFrame()),
            factors=self.factors.get(timeframe, pd.DataFrame()),
            masks=self.masks.get(timeframe, pd.DataFrame()),
        )

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "source_path": self.source_path,
            "aux_source_paths": self.aux_source_paths,
            "fingerprint": self.fingerprint,
            "timeframes": {},
        }
        for timeframe, frame in self.prices.items():
            groups = self.groups.get(timeframe, pd.DataFrame())
            factors = self.factors.get(timeframe, pd.DataFrame())
            masks = self.masks.get(timeframe, pd.DataFrame())
            summary["timeframes"][timeframe] = {
                "rows": int(len(frame)),
                "symbols": int(frame["symbol"].nunique()),
                "start": frame["timestamp"].min().isoformat() if not frame.empty else None,
                "end": frame["timestamp"].max().isoformat() if not frame.empty else None,
                "group_columns": [column for column in groups.columns if column not in AUXILIARY_KEY_COLUMNS],
                "factor_columns": [column for column in factors.columns if column not in AUXILIARY_KEY_COLUMNS],
                "mask_columns": [column for column in masks.columns if column not in AUXILIARY_KEY_COLUMNS],
            }
        return summary


def normalize_columns(
    frame: pd.DataFrame,
    column_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        normalized = str(column).strip().lower()
        normalized = COLUMN_ALIASES.get(normalized, normalized)
        renamed[column] = normalized
    normalized_frame = frame.rename(columns=renamed)
    if column_mapping:
        normalized_frame = normalized_frame.rename(columns=column_mapping)
    return normalized_frame


def attach_symbol_and_timeframe(
    frame: pd.DataFrame,
    symbol: str | None,
    timeframe: str | None,
) -> pd.DataFrame:
    result = frame.copy()
    if symbol and "symbol" not in result.columns:
        result["symbol"] = symbol
    if timeframe and "timeframe" not in result.columns:
        result["timeframe"] = timeframe
    return result


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
