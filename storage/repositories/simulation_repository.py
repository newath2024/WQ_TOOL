from __future__ import annotations

import sqlite3
from io import StringIO

import pandas as pd

from storage.models import (
    SimulationCacheRecord,
)


class SimulationRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def save_simulation_cache(self, record: SimulationCacheRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO simulation_cache
            (simulation_signature, normalized_expression, simulation_config_snapshot, delay_mode, neutralization,
             neutralization_profile, split_metrics_json, submission_tests_json, subuniverse_metrics_json,
             validation_signal_json, validation_returns_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(simulation_signature) DO UPDATE SET
                normalized_expression = excluded.normalized_expression,
                simulation_config_snapshot = excluded.simulation_config_snapshot,
                delay_mode = excluded.delay_mode,
                neutralization = excluded.neutralization,
                neutralization_profile = excluded.neutralization_profile,
                split_metrics_json = excluded.split_metrics_json,
                submission_tests_json = excluded.submission_tests_json,
                subuniverse_metrics_json = excluded.subuniverse_metrics_json,
                validation_signal_json = excluded.validation_signal_json,
                validation_returns_json = excluded.validation_returns_json,
                created_at = excluded.created_at
            """,
            (
                record.simulation_signature,
                record.normalized_expression,
                record.simulation_config_snapshot,
                record.delay_mode,
                record.neutralization,
                record.neutralization_profile,
                record.split_metrics_json,
                record.submission_tests_json,
                record.subuniverse_metrics_json,
                record.validation_signal_json,
                record.validation_returns_json,
                record.created_at,
            ),
        )
        self.connection.commit()

    def get_cached_simulation(self, simulation_signature: str) -> SimulationCacheRecord | None:
        row = self.connection.execute(
            "SELECT * FROM simulation_cache WHERE simulation_signature = ?",
            (simulation_signature,),
        ).fetchone()
        return SimulationCacheRecord(**dict(row)) if row else None

    @staticmethod
    def dataframe_to_json(frame: pd.DataFrame) -> str:
        return frame.to_json(orient="split", date_format="iso")

    @staticmethod
    def dataframe_from_json(payload: str) -> pd.DataFrame:
        return pd.read_json(StringIO(payload), orient="split")

    @staticmethod
    def series_to_json(series: pd.Series) -> str:
        return series.to_json(orient="split", date_format="iso")

    @staticmethod
    def series_from_json(payload: str) -> pd.Series:
        return pd.read_json(StringIO(payload), orient="split", typ="series")
