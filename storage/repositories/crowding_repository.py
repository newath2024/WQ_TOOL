from __future__ import annotations

import sqlite3


from storage.models import (
    CrowdingScoreRecord,
)


class CrowdingRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def save_crowding_scores(self, records: list[CrowdingScoreRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO alpha_crowding_scores
                (run_id, round_index, alpha_id, stage, total_penalty, family_penalty, motif_penalty,
                 operator_path_penalty, lineage_penalty, batch_penalty, historical_penalty, hard_blocked,
                 reason_codes_json, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, round_index, stage, alpha_id) DO UPDATE SET
                    total_penalty = excluded.total_penalty,
                    family_penalty = excluded.family_penalty,
                    motif_penalty = excluded.motif_penalty,
                    operator_path_penalty = excluded.operator_path_penalty,
                    lineage_penalty = excluded.lineage_penalty,
                    batch_penalty = excluded.batch_penalty,
                    historical_penalty = excluded.historical_penalty,
                    hard_blocked = excluded.hard_blocked,
                    reason_codes_json = excluded.reason_codes_json,
                    metrics_json = excluded.metrics_json,
                    created_at = excluded.created_at
                """,
                (
                    record.run_id,
                    record.round_index,
                    record.alpha_id,
                    record.stage,
                    record.total_penalty,
                    record.family_penalty,
                    record.motif_penalty,
                    record.operator_path_penalty,
                    record.lineage_penalty,
                    record.batch_penalty,
                    record.historical_penalty,
                    int(record.hard_blocked),
                    record.reason_codes_json,
                    record.metrics_json,
                    record.created_at,
                ),
            )
        self.connection.commit()

    def get_average_crowding_penalty(self, run_id: str) -> float:
        row = self.connection.execute(
            """
            SELECT AVG(total_penalty) AS avg_penalty
            FROM alpha_crowding_scores
            WHERE run_id = ? AND stage = 'pre_sim'
            """,
            (run_id,),
        ).fetchone()
        return float(row["avg_penalty"] or 0.0)
