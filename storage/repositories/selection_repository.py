from __future__ import annotations

import sqlite3
from typing import Any


from storage.models import (
    AlphaRecord,
    SelectionRecord,
    SelectionScoreRecord,
)


class SelectionRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def list_meta_model_training_rows(
        self,
        *,
        run_id: str,
        round_index: int,
        lookback_rounds: int,
        use_cross_run_history: bool,
    ) -> list[dict[str, Any]]:
        start_round = max(0, int(round_index) - int(lookback_rounds))
        params: list[object] = []
        history_clause = (
            "(s.run_id = ? AND s.round_index < ? AND s.round_index >= ?)"
            if not use_cross_run_history
            else "((s.run_id = ? AND s.round_index < ? AND s.round_index >= ?) OR s.run_id <> ?)"
        )
        params.extend([run_id, int(round_index), start_round])
        if use_cross_run_history:
            params.append(run_id)
        rows = self.connection.execute(
            f"""
            SELECT
                s.run_id,
                s.round_index,
                s.alpha_id,
                s.selected,
                s.composite_score,
                s.reason_codes_json,
                s.breakdown_json,
                a.generation_mode,
                a.template_name,
                a.fields_used_json,
                a.depth,
                a.complexity,
                a.generation_metadata,
                c.metric_source,
                c.motif,
                c.field_families_json,
                c.operator_path_json,
                c.complexity_bucket,
                c.turnover_bucket,
                c.horizon_bucket,
                c.effective_regime_key,
                c.outcome_score,
                c.created_at
            FROM alpha_selection_scores s
            JOIN alphas a
              ON a.run_id = s.run_id AND a.alpha_id = s.alpha_id
            JOIN alpha_cases c
              ON c.run_id = s.run_id AND c.alpha_id = s.alpha_id AND c.metric_source = 'external_brain'
            WHERE s.score_stage = 'pre_sim'
              AND {history_clause}
            ORDER BY s.run_id ASC, s.round_index ASC, s.alpha_id ASC
            """,
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]

    def replace_selections(self, run_id: str, records: list[SelectionRecord]) -> None:
        self.connection.execute("DELETE FROM selections WHERE run_id = ?", (run_id,))
        for record in records:
            self.connection.execute(
                """
                INSERT INTO selections
                (run_id, alpha_id, rank, selected_at, validation_fitness, reason, ranking_rationale_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.alpha_id,
                    record.rank,
                    record.selected_at,
                    record.validation_fitness,
                    record.reason,
                    record.ranking_rationale_json,
                ),
            )
        self.connection.commit()

    def get_top_alpha_records(self, run_id: str, limit: int) -> list[AlphaRecord]:
        rows = self.connection.execute(
            """
            SELECT a.*
            FROM selections s
            JOIN alphas a
              ON a.run_id = s.run_id AND a.alpha_id = s.alpha_id
            WHERE s.run_id = ?
            ORDER BY s.rank ASC
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        return [AlphaRecord(**dict(row)) for row in rows]

    def save_selection_scores(self, records: list[SelectionScoreRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO alpha_selection_scores
                (run_id, round_index, alpha_id, score_stage, composite_score, selected, rank, reason_codes_json,
                 breakdown_json, quality_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, round_index, score_stage, alpha_id) DO UPDATE SET
                    composite_score = excluded.composite_score,
                    selected = excluded.selected,
                    rank = excluded.rank,
                    reason_codes_json = excluded.reason_codes_json,
                    breakdown_json = excluded.breakdown_json,
                    quality_score = excluded.quality_score,
                    created_at = excluded.created_at
                """,
                (
                    record.run_id,
                    record.round_index,
                    record.alpha_id,
                    record.score_stage,
                    record.composite_score,
                    int(record.selected),
                    record.rank,
                    record.reason_codes_json,
                    record.breakdown_json,
                    record.quality_score,
                    record.created_at,
                ),
            )
        self.connection.commit()

    def list_selection_scores(self, run_id: str, *, score_stage: str | None = None) -> list[dict]:
        if score_stage:
            rows = self.connection.execute(
                """
                SELECT *
                FROM alpha_selection_scores
                WHERE run_id = ? AND score_stage = ?
                ORDER BY round_index ASC, composite_score DESC, alpha_id ASC
                """,
                (run_id, score_stage),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT *
                FROM alpha_selection_scores
                WHERE run_id = ?
                ORDER BY round_index ASC, score_stage ASC, composite_score DESC, alpha_id ASC
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_top_selections(self, run_id: str, limit: int) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT s.rank, s.alpha_id, s.validation_fitness, s.reason, s.ranking_rationale_json,
                   a.expression, a.generation_mode, a.complexity,
                   m.delay_mode, m.neutralization, m.submission_pass_count, m.cache_hit
            FROM selections s
            JOIN alphas a
              ON a.run_id = s.run_id AND a.alpha_id = s.alpha_id
            LEFT JOIN metrics m
              ON m.run_id = s.run_id AND m.alpha_id = s.alpha_id AND m.split = 'validation'
            WHERE s.run_id = ?
            ORDER BY s.rank ASC
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
