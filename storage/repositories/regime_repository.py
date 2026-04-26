from __future__ import annotations

import sqlite3


from storage.models import (
    RegimeSnapshotRecord,
)


class RegimeRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def save_regime_snapshots(self, records: list[RegimeSnapshotRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO regime_snapshots
                (run_id, round_index, region, legacy_regime_key, global_regime_key, market_regime_key,
                 effective_regime_key, regime_label, confidence, features_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, round_index) DO UPDATE SET
                    region = excluded.region,
                    legacy_regime_key = excluded.legacy_regime_key,
                    global_regime_key = excluded.global_regime_key,
                    market_regime_key = excluded.market_regime_key,
                    effective_regime_key = excluded.effective_regime_key,
                    regime_label = excluded.regime_label,
                    confidence = excluded.confidence,
                    features_json = excluded.features_json,
                    created_at = excluded.created_at
                """,
                (
                    record.run_id,
                    record.round_index,
                    record.region,
                    record.legacy_regime_key,
                    record.global_regime_key,
                    record.market_regime_key,
                    record.effective_regime_key,
                    record.regime_label,
                    record.confidence,
                    record.features_json,
                    record.created_at,
                ),
            )
        self.connection.commit()

    def get_latest_regime_snapshot(self, run_id: str) -> dict | None:
        row = self.connection.execute(
            """
            SELECT *
            FROM regime_snapshots
            WHERE run_id = ?
            ORDER BY round_index DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        return dict(row) if row else None
