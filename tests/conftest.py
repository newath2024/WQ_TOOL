from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from features.transforms import pivot_ohlcv_fields


def build_sample_market_frame() -> pd.DataFrame:
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    base = {"AAA": 100.0, "BBB": 60.0, "CCC": 140.0, "DDD": 90.0}
    slopes = {"AAA": 0.35, "BBB": -0.12, "CCC": 0.18, "DDD": 0.05}
    volumes = {"AAA": 100000, "BBB": 140000, "CCC": 90000, "DDD": 110000}
    rows: list[list[object]] = []

    current = date(2021, 1, 1)
    end = date(2021, 3, 31)
    dates: list[date] = []
    while current <= end:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)

    for i, timestamp in enumerate(dates):
        for j, symbol in enumerate(symbols):
            drift = slopes[symbol] * i
            season = ((i + 1) * (j + 2)) % 7 - 3
            close = round(base[symbol] + drift + season * 0.4 + j * 1.5, 4)
            open_ = round(close - 0.25 + ((i + j) % 3) * 0.08, 4)
            high = round(max(open_, close) + 0.45 + (j * 0.05), 4)
            low = round(min(open_, close) - 0.40 - (j * 0.04), 4)
            volume = int(volumes[symbol] + i * (250 + j * 40) + season * 500)
            rows.append([timestamp.isoformat(), symbol, open_, high, low, close, volume])

    return pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])


def write_sample_csv(path) -> None:
    build_sample_market_frame().to_csv(path, index=False)


def build_static_metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["AAA", "technology", "software", "usa", "application_software", 1.18, 0.92, 0.38, 0.95, 1, 1],
            ["BBB", "technology", "software", "usa", "application_software", 1.05, 0.88, 0.41, 0.90, 1, 1],
            ["CCC", "financials", "banks", "canada", "regional_banks", 0.82, 0.71, 0.52, 0.68, 1, 0],
            ["DDD", "financials", "banks", "canada", "regional_banks", 0.79, 0.67, 0.49, 0.62, 1, 0],
        ],
        columns=[
            "symbol",
            "sector",
            "industry",
            "country",
            "subindustry",
            "beta",
            "size",
            "volatility",
            "liquidity",
            "core_mask",
            "liquid_mask",
        ],
    )


def build_aligned_metadata_frame(market_frame: pd.DataFrame) -> pd.DataFrame:
    keys = market_frame.loc[:, ["timestamp", "symbol"]].drop_duplicates()
    return keys.merge(build_static_metadata_frame(), on="symbol", how="left", validate="many_to_one")


def write_sample_metadata_csv(path) -> None:
    build_static_metadata_frame().to_csv(path, index=False)


@pytest.fixture()
def sample_market_frame() -> pd.DataFrame:
    return build_sample_market_frame()


@pytest.fixture()
def sample_metadata_frame(sample_market_frame) -> pd.DataFrame:
    return build_aligned_metadata_frame(sample_market_frame)


@pytest.fixture()
def sample_research_matrices(sample_market_frame, sample_metadata_frame):
    return pivot_ohlcv_fields(
        sample_market_frame,
        groups=sample_metadata_frame.loc[:, ["timestamp", "symbol", "sector", "industry", "country", "subindustry"]],
        factors=sample_metadata_frame.loc[:, ["timestamp", "symbol", "beta", "size", "volatility", "liquidity"]],
        masks=sample_metadata_frame.loc[:, ["timestamp", "symbol", "core_mask", "liquid_mask"]],
    )


@pytest.fixture()
def sample_wide_fields(sample_research_matrices):
    return sample_research_matrices
