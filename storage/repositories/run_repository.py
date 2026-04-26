from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime


from storage.models import (
    RunRecord,
)


class RunRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def upsert_run(
        self,
        run_id: str,
        seed: int,
        config_path: str,
        config_snapshot: str,
        status: str,
        started_at: str,
        profile_name: str = "",
        selected_timeframe: str = "",
        global_regime_key: str = "",
        region: str = "",
        entry_command: str = "",
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO runs
            (run_id, seed, config_path, config_snapshot, status, started_at, profile_name, selected_timeframe,
             global_regime_key, region, entry_command)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                seed = excluded.seed,
                config_path = excluded.config_path,
                config_snapshot = excluded.config_snapshot,
                status = excluded.status,
                profile_name = excluded.profile_name,
                selected_timeframe = excluded.selected_timeframe,
                global_regime_key = CASE
                    WHEN excluded.global_regime_key <> '' THEN excluded.global_regime_key
                    ELSE runs.global_regime_key
                END,
                region = CASE
                    WHEN excluded.region <> '' THEN excluded.region
                    ELSE runs.region
                END,
                entry_command = excluded.entry_command
            """,
            (
                run_id,
                seed,
                config_path,
                config_snapshot,
                status,
                started_at,
                profile_name,
                selected_timeframe,
                global_regime_key,
                region,
                entry_command,
            ),
        )
        self.connection.commit()

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return RunRecord(**dict(row)) if row else None

    def get_latest_run(self) -> RunRecord | None:
        row = self.connection.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        return RunRecord(**dict(row)) if row else None

    def update_run_status(self, run_id: str, status: str, finished: bool = False) -> None:
        finished_at = datetime.now(UTC).isoformat() if finished else None
        self.connection.execute(
            "UPDATE runs SET status = ?, finished_at = COALESCE(?, finished_at) WHERE run_id = ?",
            (status, finished_at, run_id),
        )
        self.connection.commit()

    def save_dataset_summary(
        self,
        run_id: str,
        summary: dict,
        dataset_fingerprint: str | None = None,
        selected_timeframe: str | None = None,
        regime_key: str | None = None,
        global_regime_key: str | None = None,
        market_regime_key: str | None = None,
        effective_regime_key: str | None = None,
        regime_label: str | None = None,
        regime_confidence: float | None = None,
        region: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE runs
            SET dataset_summary = ?,
                dataset_fingerprint = COALESCE(?, dataset_fingerprint),
                selected_timeframe = COALESCE(?, selected_timeframe),
                regime_key = COALESCE(?, regime_key),
                global_regime_key = COALESCE(?, global_regime_key),
                market_regime_key = COALESCE(?, market_regime_key),
                effective_regime_key = COALESCE(?, effective_regime_key),
                regime_label = COALESCE(?, regime_label),
                regime_confidence = COALESCE(?, regime_confidence),
                region = COALESCE(?, region)
            WHERE run_id = ?
            """,
            (
                json.dumps(summary, sort_keys=True),
                dataset_fingerprint,
                selected_timeframe,
                regime_key,
                global_regime_key,
                market_regime_key,
                effective_regime_key,
                regime_label,
                regime_confidence,
                region,
                run_id,
            ),
        )
        self.connection.commit()
