from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from core.config import AuxDataConfig, DataConfig
from data.schema import MarketDataBundle, attach_symbol_and_timeframe, normalize_columns, resolve_path
from data.validation import validate_auxiliary_frame, validate_market_frame


FILENAME_REGEX = re.compile(r"(?P<symbol>[A-Za-z0-9_\-]+)(?:[_\-](?P<timeframe>[A-Za-z0-9]+))?\.csv$")


class MarketDataLoader:
    def __init__(self, config: DataConfig, aux_config: AuxDataConfig, logger) -> None:
        self.config = config
        self.aux_config = aux_config
        self.logger = logger

    def load(self) -> MarketDataBundle:
        if self.config.format.lower() != "csv":
            raise NotImplementedError("Only CSV is implemented in the MVP. Parquet is a planned extension hook.")

        source_path = resolve_path(self.config.path)
        frames, source_files = self._load_csv(source_path)
        validated_prices = {
            timeframe: validate_market_frame(frame, timeframe=timeframe, logger=self.logger)
            for timeframe, frame in frames.items()
        }

        groups, group_path = self._load_auxiliary_bundle(
            path=self.aux_config.group_path,
            value_columns=self.aux_config.group_columns,
            numeric_columns=[],
            prices=validated_prices,
            label="groups",
        )
        factors, factor_path = self._load_auxiliary_bundle(
            path=self.aux_config.factor_path,
            value_columns=self.aux_config.factor_columns,
            numeric_columns=self.aux_config.factor_columns,
            prices=validated_prices,
            label="factors",
        )
        masks, mask_path = self._load_auxiliary_bundle(
            path=self.aux_config.mask_path,
            value_columns=self.aux_config.mask_columns,
            numeric_columns=self.aux_config.mask_columns,
            prices=validated_prices,
            label="masks",
        )

        aux_paths = {
            name: str(path)
            for name, path in {"groups": group_path, "factors": factor_path, "masks": mask_path}.items()
            if path is not None
        }
        fingerprint = self._fingerprint_files(source_files + [path for path in (group_path, factor_path, mask_path) if path])
        return MarketDataBundle(
            prices=validated_prices,
            groups=groups,
            factors=factors,
            masks=masks,
            source_path=str(source_path),
            aux_source_paths=aux_paths,
            fingerprint=fingerprint,
        )

    def _load_csv(self, source_path: Path) -> tuple[dict[str, pd.DataFrame], list[Path]]:
        if source_path.is_file():
            frame = self._read_csv_file(source_path)
            return self._split_by_timeframe(frame), [source_path]
        if source_path.is_dir():
            if self.config.input_layout == "canonical":
                files = sorted(source_path.glob("*.csv"))
                if not files:
                    raise FileNotFoundError(f"No CSV files found under {source_path}")
                combined = pd.concat([self._read_csv_file(path) for path in files], ignore_index=True)
                return self._split_by_timeframe(combined), files
            if self.config.input_layout == "per_file":
                return self._load_per_file_csv(source_path)
        raise FileNotFoundError(f"Unsupported data path or layout: {source_path}")

    def _read_csv_file(
        self,
        path: Path,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> pd.DataFrame:
        self.logger.info("Reading CSV file %s", path)
        frame = pd.read_csv(path)
        frame = normalize_columns(frame, column_mapping=self.config.column_mapping)
        return attach_symbol_and_timeframe(frame, symbol=symbol, timeframe=timeframe or self.config.default_timeframe)

    def _split_by_timeframe(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if "timeframe" not in frame.columns:
            frame = frame.copy()
            frame["timeframe"] = self.config.default_timeframe
        grouped: dict[str, pd.DataFrame] = {}
        for timeframe, chunk in frame.groupby("timeframe", sort=True):
            subset = chunk.drop(columns=["timeframe"])
            if self.config.universe:
                subset = subset[subset["symbol"].isin(self.config.universe)]
            grouped[str(timeframe)] = subset
        if not grouped:
            raise ValueError("No market data remained after applying the universe filter.")
        return grouped

    def _load_per_file_csv(self, source_path: Path) -> tuple[dict[str, pd.DataFrame], list[Path]]:
        frames: list[pd.DataFrame] = []
        files = sorted(source_path.glob("*.csv"))
        for path in files:
            symbol, timeframe = self._infer_metadata_from_filename(path)
            frames.append(self._read_csv_file(path, symbol=symbol, timeframe=timeframe))
        if not frames:
            raise FileNotFoundError(f"No CSV files found under {source_path}")
        return self._split_by_timeframe(pd.concat(frames, ignore_index=True)), files

    def _infer_metadata_from_filename(self, path: Path) -> tuple[str | None, str | None]:
        match = FILENAME_REGEX.match(path.name)
        if not match:
            return None, self.config.default_timeframe
        symbol = match.group("symbol")
        timeframe = match.group("timeframe") or self.config.default_timeframe
        return symbol, timeframe

    def _load_auxiliary_bundle(
        self,
        path: str | None,
        value_columns: list[str],
        numeric_columns: list[str],
        prices: dict[str, pd.DataFrame],
        label: str,
    ) -> tuple[dict[str, pd.DataFrame], Path | None]:
        if not path or not value_columns:
            return {}, None
        if self.aux_config.format.lower() != "csv":
            raise NotImplementedError("Only CSV auxiliary data is implemented in the MVP.")

        aux_path = resolve_path(path)
        self.logger.info("Reading auxiliary %s file %s", label, aux_path)
        frame = pd.read_csv(aux_path)
        frame = normalize_columns(frame, column_mapping=self.aux_config.column_mapping)
        frame = self._rename_auxiliary_keys(frame)
        frame = validate_auxiliary_frame(
            frame=frame,
            label=label,
            value_columns=value_columns,
            numeric_columns=numeric_columns,
            logger=self.logger,
        )
        return self._align_auxiliary_to_prices(prices, frame, value_columns, label), aux_path

    def _rename_auxiliary_keys(self, frame: pd.DataFrame) -> pd.DataFrame:
        renamed = frame.copy()
        if self.aux_config.timestamp_column in renamed.columns and self.aux_config.timestamp_column != "timestamp":
            renamed = renamed.rename(columns={self.aux_config.timestamp_column: "timestamp"})
        if self.aux_config.symbol_column in renamed.columns and self.aux_config.symbol_column != "symbol":
            renamed = renamed.rename(columns={self.aux_config.symbol_column: "symbol"})
        return renamed

    def _align_auxiliary_to_prices(
        self,
        prices: dict[str, pd.DataFrame],
        aux_frame: pd.DataFrame,
        value_columns: list[str],
        label: str,
    ) -> dict[str, pd.DataFrame]:
        aligned: dict[str, pd.DataFrame] = {}
        has_timestamp = "timestamp" in aux_frame.columns
        for timeframe, price_frame in prices.items():
            keys = price_frame.loc[:, ["timestamp", "symbol"]].drop_duplicates().copy()
            if has_timestamp:
                merged = keys.merge(aux_frame, on=["timestamp", "symbol"], how="left", validate="one_to_one")
            else:
                merged = keys.merge(aux_frame, on=["symbol"], how="left", validate="many_to_one")
            if merged[value_columns].isna().any().any():
                missing_count = int(merged[value_columns].isna().any(axis=1).sum())
                raise ValueError(
                    f"Auxiliary frame '{label}' is missing {missing_count} aligned rows for timeframe '{timeframe}'."
                )
            aligned[timeframe] = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        return aligned

    def _fingerprint_files(self, paths: list[Path]) -> str:
        digest = hashlib.sha1()
        for path in sorted(paths):
            digest.update(str(path).encode("utf-8"))
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 64):
                    digest.update(chunk)
        return digest.hexdigest()


def load_market_data(config: DataConfig, logger, aux_config: AuxDataConfig | None = None) -> MarketDataBundle:
    return MarketDataLoader(config=config, aux_config=aux_config or AuxDataConfig(), logger=logger).load()
