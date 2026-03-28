from __future__ import annotations

import logging

import pandas as pd

from data.schema import AUXILIARY_KEY_COLUMNS, CANONICAL_COLUMNS, PRICE_COLUMNS


def validate_market_frame(
    frame: pd.DataFrame,
    timeframe: str,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns for timeframe '{timeframe}': {missing}")

    result = frame.loc[:, list(CANONICAL_COLUMNS)].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=False, errors="raise")
    result["symbol"] = result["symbol"].astype(str)
    for column in PRICE_COLUMNS + ("volume",):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    duplicate_mask = result.duplicated(subset=["timestamp", "symbol"], keep=False)
    if duplicate_mask.any():
        duplicates = result.loc[duplicate_mask, ["timestamp", "symbol"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate timestamp/symbol rows detected for '{timeframe}': {duplicates}")

    for column in PRICE_COLUMNS:
        invalid = result[column].dropna() <= 0
        if invalid.any():
            raise ValueError(f"Nonpositive values found in column '{column}' for timeframe '{timeframe}'.")

    invalid_volume = result["volume"].dropna() < 0
    if invalid_volume.any():
        raise ValueError(f"Negative volume found for timeframe '{timeframe}'.")

    result = result.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    _log_missing_counts(result, logger, timeframe, "market")
    return result


def validate_auxiliary_frame(
    frame: pd.DataFrame,
    label: str,
    value_columns: list[str],
    numeric_columns: list[str],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    missing = [column for column in ("symbol",) + tuple(value_columns) if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns for auxiliary frame '{label}': {missing}")

    result = frame.loc[:, [column for column in frame.columns if column in {"timestamp", "symbol", *value_columns}]].copy()
    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=False, errors="raise")
    result["symbol"] = result["symbol"].astype(str)

    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    key_columns = list(AUXILIARY_KEY_COLUMNS) if "timestamp" in result.columns else ["symbol"]
    duplicate_mask = result.duplicated(subset=key_columns, keep=False)
    if duplicate_mask.any():
        duplicates = result.loc[duplicate_mask, key_columns].head(10).to_dict("records")
        raise ValueError(f"Duplicate rows detected for auxiliary frame '{label}': {duplicates}")

    result = result.sort_values(key_columns).reset_index(drop=True)
    _log_missing_counts(result, logger, label, "auxiliary")
    return result


def _log_missing_counts(
    frame: pd.DataFrame,
    logger: logging.Logger,
    label: str,
    kind: str,
) -> None:
    missing_counts = frame.isna().sum()
    if missing_counts.any():
        observed = {column: int(count) for column, count in missing_counts.items() if count > 0}
        logger.warning("NaN values retained for %s frame '%s': %s", kind, label, observed)
