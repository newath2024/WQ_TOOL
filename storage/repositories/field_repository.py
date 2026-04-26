from __future__ import annotations

import sqlite3
from collections.abc import Iterable


from storage.models import (
    FieldCatalogRecord,
    RunFieldScoreRecord,
)


class FieldRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def delete_field_metadata(self, field_names: Iterable[str], *, run_id: str | None = None) -> None:
        normalized = tuple(sorted({str(name).strip() for name in field_names if str(name).strip()}))
        if not normalized:
            return
        placeholders = ", ".join("?" for _ in normalized)
        self.connection.execute(
            f"DELETE FROM field_catalog WHERE field_name IN ({placeholders})",
            normalized,
        )
        if run_id:
            self.connection.execute(
                f"DELETE FROM run_field_scores WHERE run_id = ? AND field_name IN ({placeholders})",
                (run_id, *normalized),
            )
        else:
            self.connection.execute(
                f"DELETE FROM run_field_scores WHERE field_name IN ({placeholders})",
                normalized,
            )
        self.connection.commit()

    def save_field_catalog(self, records: list[FieldCatalogRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO field_catalog
                (field_name, dataset, field_type, coverage, alpha_usage_count, category, delay, region, universe,
                 runtime_available, description, subcategory, user_count, category_weight, field_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(field_name) DO UPDATE SET
                    dataset = excluded.dataset,
                    field_type = excluded.field_type,
                    coverage = excluded.coverage,
                    alpha_usage_count = excluded.alpha_usage_count,
                    category = excluded.category,
                    delay = excluded.delay,
                    region = excluded.region,
                    universe = excluded.universe,
                    runtime_available = excluded.runtime_available,
                    description = excluded.description,
                    subcategory = excluded.subcategory,
                    user_count = excluded.user_count,
                    category_weight = excluded.category_weight,
                    field_score = excluded.field_score,
                    updated_at = excluded.updated_at
                """,
                (
                    record.field_name,
                    record.dataset,
                    record.field_type,
                    record.coverage,
                    record.alpha_usage_count,
                    record.category,
                    record.delay,
                    record.region,
                    record.universe,
                    int(record.runtime_available),
                    record.description,
                    record.subcategory,
                    record.user_count,
                    record.category_weight,
                    record.field_score,
                    record.updated_at,
                ),
            )
        self.connection.commit()

    def replace_run_field_scores(self, run_id: str, records: list[RunFieldScoreRecord]) -> None:
        self.connection.execute("DELETE FROM run_field_scores WHERE run_id = ?", (run_id,))
        for record in records:
            self.connection.execute(
                """
                INSERT INTO run_field_scores
                (run_id, field_name, runtime_available, field_type, category, field_score, coverage, alpha_usage_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.field_name,
                    int(record.runtime_available),
                    record.field_type,
                    record.category,
                    record.field_score,
                    record.coverage,
                    record.alpha_usage_count,
                    record.created_at,
                ),
            )
        self.connection.commit()

    def list_run_field_scores(self, run_id: str) -> dict[str, float]:
        rows = self.connection.execute(
            """
            SELECT field_name, field_score
            FROM run_field_scores
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
        return {str(row["field_name"]): float(row["field_score"] or 0.0) for row in rows}

    def list_run_field_scores_for_runs(self, run_ids: Iterable[str]) -> dict[str, dict[str, float]]:
        normalized = tuple(sorted({str(run_id).strip() for run_id in run_ids if str(run_id).strip()}))
        if not normalized:
            return {}
        placeholders = ", ".join("?" for _ in normalized)
        rows = self.connection.execute(
            f"""
            SELECT run_id, field_name, field_score
            FROM run_field_scores
            WHERE run_id IN ({placeholders})
            """,
            normalized,
        ).fetchall()
        scores: dict[str, dict[str, float]] = {}
        for row in rows:
            run_score_map = scores.setdefault(str(row["run_id"]), {})
            run_score_map[str(row["field_name"])] = float(row["field_score"] or 0.0)
        return scores
