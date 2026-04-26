from __future__ import annotations

import sqlite3


from storage.models import (
    DuplicateDecisionRecord,
)


class DuplicateRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def get_cross_run_duplicate_references(
        self,
        *,
        run_id: str,
        global_regime_key: str,
        limit: int,
    ) -> list[dict]:
        if not global_regime_key or limit <= 0:
            return []
        rows = self.connection.execute(
            """
            SELECT
                run_id,
                alpha_id,
                normalized_expression,
                structural_signature_json,
                outcome_score,
                selected,
                passed_filters,
                metric_source,
                created_at
            FROM alpha_history
            WHERE run_id <> ?
              AND global_regime_key = ?
              AND (
                    passed_filters = 1
                    OR selected = 1
                    OR (metric_source = 'external_brain' AND outcome_score > 0)
                  )
            ORDER BY outcome_score DESC, created_at DESC
            LIMIT ?
            """,
            (run_id, global_regime_key, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def save_duplicate_decisions(self, records: list[DuplicateDecisionRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO alpha_duplicate_decisions
                (run_id, round_index, alpha_id, stage, decision, reason_code, matched_run_id, matched_alpha_id,
                 matched_scope, similarity_score, normalized_match, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, round_index, stage, alpha_id, decision, reason_code) DO UPDATE SET
                    matched_run_id = excluded.matched_run_id,
                    matched_alpha_id = excluded.matched_alpha_id,
                    matched_scope = excluded.matched_scope,
                    similarity_score = excluded.similarity_score,
                    normalized_match = excluded.normalized_match,
                    metrics_json = excluded.metrics_json,
                    created_at = excluded.created_at
                """,
                (
                    record.run_id,
                    record.round_index,
                    record.alpha_id,
                    record.stage,
                    record.decision,
                    record.reason_code,
                    record.matched_run_id,
                    record.matched_alpha_id,
                    record.matched_scope,
                    record.similarity_score,
                    int(record.normalized_match),
                    record.metrics_json,
                    record.created_at,
                ),
            )
        self.connection.commit()

    def get_duplicate_decision_summary(self, run_id: str) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT stage, decision, reason_code, COUNT(*) AS total_count
            FROM alpha_duplicate_decisions
            WHERE run_id = ?
            GROUP BY stage, decision, reason_code
            ORDER BY stage ASC, total_count DESC, reason_code ASC
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]
