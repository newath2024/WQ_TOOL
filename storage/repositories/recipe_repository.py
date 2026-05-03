from __future__ import annotations

import json
import sqlite3
from typing import Any


from storage.repositories.alpha_repository import AlphaRepository


class RecipeRepository:
    def __init__(self, connection: sqlite3.Connection, alpha_repository: AlphaRepository) -> None:
        self.connection = connection
        self.alphas = alpha_repository

    def list_recipe_parent_rows(
        self,
        *,
        run_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return self.alphas._list_recent_completed_parent_rows(run_id=run_id, limit=limit)

    def list_recent_recipe_guided_usage_rows(
        self,
        *,
        run_id: str,
        before_round_index: int,
        lookback_rounds: int,
    ) -> list[dict[str, Any]]:
        if lookback_rounds <= 0:
            return []
        rows = self.connection.execute(
            """
            SELECT alpha_id, normalized_expression, fields_used_json, generation_metadata, created_at
            FROM alphas
            WHERE run_id = ?
              AND generation_mode = 'recipe_guided'
            ORDER BY created_at DESC, alpha_id DESC
            """,
            (run_id,),
        ).fetchall()
        min_round_index = max(0, int(before_round_index) - int(lookback_rounds))
        usage_rows: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(row["generation_metadata"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            round_index = _optional_int(metadata.get("recipe_round_index"))
            if round_index is None:
                continue
            if round_index < min_round_index or round_index >= int(before_round_index):
                continue
            usage_rows.append(
                {
                    "alpha_id": str(row["alpha_id"] or "").strip(),
                    "normalized_expression": str(row["normalized_expression"] or "").strip(),
                    "fields_used_json": str(row["fields_used_json"] or "[]"),
                    "generation_metadata": metadata,
                    "recipe_round_index": round_index,
                    "created_at": str(row["created_at"] or ""),
                }
            )
        return usage_rows

    def list_recipe_bucket_result_rows(
        self,
        *,
        run_id: str,
        before_round_index: int,
        lookback_rounds: int,
    ) -> list[dict[str, Any]]:
        if lookback_rounds <= 0:
            return []
        min_round_index = max(0, int(before_round_index) - int(lookback_rounds))
        rows = self.connection.execute(
            """
            SELECT
                r.round_index,
                r.candidate_id,
                r.status,
                r.fitness,
                r.sharpe,
                r.turnover,
                r.drawdown,
                r.returns,
                r.margin,
                r.submission_eligible,
                r.rejection_reason,
                r.quality_score,
                r.check_summary_json,
                r.hard_fail_checks_json,
                r.warning_checks_json,
                r.blocking_warning_checks_json,
                r.derived_submit_ready,
                a.generation_mode,
                a.generation_metadata
            FROM brain_results r
            JOIN alphas a
              ON a.run_id = r.run_id AND a.alpha_id = r.candidate_id
            WHERE r.run_id = ?
              AND a.generation_mode = 'recipe_guided'
              AND r.round_index >= ?
              AND r.round_index < ?
            ORDER BY r.round_index DESC, r.simulated_at DESC, r.job_id DESC
            """,
            (run_id, min_round_index, int(before_round_index)),
        ).fetchall()
        return [dict(row) for row in rows]


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
