from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from typing import Any


from domain.candidate import AlphaCandidate
from memory.pattern_memory import PatternMemoryService
from storage.models import (
    AlphaRecord,
)


class AlphaRepository:
    def __init__(self, connection: sqlite3.Connection, memory_service: PatternMemoryService) -> None:
        self.connection = connection
        self._memory_service = memory_service

    def save_alpha_candidates(self, run_id: str, candidates: list[AlphaCandidate]) -> int:
        inserted = 0
        for candidate in candidates:
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO alphas
                (run_id, alpha_id, expression, normalized_expression, generation_mode, template_name, fields_used_json,
                 operators_used_json, depth, generation_metadata, structural_signature_json, complexity, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    candidate.alpha_id,
                    candidate.expression,
                    candidate.normalized_expression,
                    candidate.generation_mode,
                    candidate.template_name,
                    json.dumps(list(candidate.fields_used), sort_keys=True),
                    json.dumps(list(candidate.operators_used), sort_keys=True),
                    candidate.depth,
                    json.dumps(candidate.generation_metadata, sort_keys=True),
                    self._candidate_structural_signature_json(candidate),
                    candidate.complexity,
                    candidate.created_at,
                    "generated",
                ),
            )
            if cursor.rowcount:
                inserted += 1
            parent_refs = candidate.generation_metadata.get("parent_refs")
            if isinstance(parent_refs, list):
                normalized_parent_refs = [
                    (
                        str(parent.get("run_id") or run_id),
                        str(parent.get("alpha_id")),
                    )
                    for parent in parent_refs
                    if parent.get("alpha_id")
                ]
            else:
                normalized_parent_refs = [(run_id, parent_id) for parent_id in candidate.parent_ids]
            for parent_run_id, parent_id in normalized_parent_refs:
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO alpha_parents (run_id, alpha_id, parent_run_id, parent_alpha_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, candidate.alpha_id, parent_run_id, parent_id),
                )
        self.connection.commit()
        return inserted

    def _candidate_structural_signature_json(self, candidate: AlphaCandidate) -> str:
        payload = candidate.generation_metadata.get("canonical_structural_signature")
        if isinstance(payload, dict) and payload:
            return json.dumps(payload, sort_keys=True)
        try:
            signature = self._memory_service.extract_signature(
                candidate.normalized_expression or candidate.expression,
                generation_metadata=candidate.generation_metadata,
            )
        except Exception:  # noqa: BLE001
            return "{}"
        return json.dumps(signature.to_dict(), sort_keys=True)

    def list_alpha_records(self, run_id: str) -> list[AlphaRecord]:
        rows = self.connection.execute(
            "SELECT * FROM alphas WHERE run_id = ? ORDER BY created_at ASC, alpha_id ASC",
            (run_id,),
        ).fetchall()
        return [AlphaRecord(**dict(row)) for row in rows]

    def _list_recent_completed_parent_rows(
        self,
        *,
        run_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        rows = self.connection.execute(
            """
            SELECT
                a.run_id,
                a.alpha_id,
                a.expression,
                a.normalized_expression,
                a.generation_mode,
                a.template_name,
                a.fields_used_json,
                a.operators_used_json,
                a.depth,
                a.generation_metadata,
                a.structural_signature_json,
                a.complexity,
                a.created_at,
                a.status AS alpha_status,
                r.job_id AS result_job_id,
                r.round_index AS result_round_index,
                r.batch_id AS result_batch_id,
                r.status AS result_status,
                r.region AS result_region,
                r.universe AS result_universe,
                r.delay AS result_delay,
                r.neutralization AS result_neutralization,
                r.decay AS result_decay,
                r.sharpe AS result_sharpe,
                r.fitness AS result_fitness,
                r.turnover AS result_turnover,
                r.drawdown AS result_drawdown,
                r.returns AS result_returns,
                r.margin AS result_margin,
                r.submission_eligible AS result_submission_eligible,
                r.rejection_reason AS result_rejection_reason,
                r.quality_score AS result_quality_score,
                r.check_summary_json AS result_check_summary_json,
                r.hard_fail_checks_json AS result_hard_fail_checks_json,
                r.warning_checks_json AS result_warning_checks_json,
                r.blocking_warning_checks_json AS result_blocking_warning_checks_json,
                r.derived_submit_ready AS result_derived_submit_ready,
                r.simulated_at AS result_simulated_at
            FROM brain_results r
            JOIN alphas a
              ON a.run_id = r.run_id AND a.alpha_id = r.candidate_id
            WHERE r.run_id = ?
              AND r.status = 'completed'
              AND NOT EXISTS (
                  SELECT 1
                  FROM brain_results newer
                  WHERE newer.run_id = r.run_id
                    AND newer.candidate_id = r.candidate_id
                    AND newer.status = 'completed'
                    AND (
                        newer.simulated_at > r.simulated_at
                        OR (newer.simulated_at = r.simulated_at AND newer.job_id > r.job_id)
                    )
              )
            ORDER BY r.simulated_at DESC, r.job_id DESC
            LIMIT ?
            """,
            (run_id, int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_generation_result_rows(
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
                a.generation_metadata,
                a.structural_signature_json
            FROM brain_results r
            JOIN alphas a
              ON a.run_id = r.run_id AND a.alpha_id = r.candidate_id
            WHERE r.run_id = ?
              AND r.round_index >= ?
              AND r.round_index < ?
            ORDER BY r.round_index DESC, r.simulated_at DESC, r.job_id DESC
            """,
            (run_id, min_round_index, int(before_round_index)),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_alpha_reference_marker(self, run_id: str) -> tuple[int, str]:
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS alpha_count, MAX(created_at) AS max_created_at
            FROM alphas
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return 0, ""
        return int(row["alpha_count"] or 0), str(row["max_created_at"] or "")

    def get_existing_alpha_ids_by_normalized(
        self,
        run_id: str,
        normalized_expressions: Iterable[str],
        *,
        exclude_alpha_ids: Iterable[str] = (),
    ) -> dict[str, str]:
        expressions = tuple(
            dict.fromkeys(str(expression) for expression in normalized_expressions if str(expression))
        )
        if not expressions:
            return {}
        excluded = tuple(dict.fromkeys(str(alpha_id) for alpha_id in exclude_alpha_ids if str(alpha_id)))
        expression_placeholders = ",".join("?" for _ in expressions)
        params: list[str] = [run_id, *expressions]
        exclusion_sql = ""
        if excluded:
            exclusion_placeholders = ",".join("?" for _ in excluded)
            exclusion_sql = f" AND alpha_id NOT IN ({exclusion_placeholders})"
            params.extend(excluded)
        rows = self.connection.execute(
            f"""
            SELECT normalized_expression, alpha_id
            FROM alphas
            WHERE run_id = ?
              AND normalized_expression IN ({expression_placeholders})
              {exclusion_sql}
            ORDER BY created_at ASC, alpha_id ASC
            """,
            tuple(params),
        ).fetchall()
        existing: dict[str, str] = {}
        for row in rows:
            existing.setdefault(str(row["normalized_expression"]), str(row["alpha_id"]))
        return existing

    def get_same_run_structural_references(
        self,
        *,
        run_id: str,
        limit: int,
    ) -> list[dict]:
        if limit <= 0:
            return []
        rows = self.connection.execute(
            """
            SELECT
                run_id,
                alpha_id,
                normalized_expression,
                expression,
                generation_metadata,
                structural_signature_json,
                created_at
            FROM alphas
            WHERE run_id = ?
            ORDER BY created_at DESC, alpha_id DESC
            LIMIT ?
            """,
            (run_id, int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]

    def update_alpha_structural_signature(
        self,
        *,
        run_id: str,
        alpha_id: str,
        structural_signature_json: str,
    ) -> None:
        self.connection.execute(
            """
            UPDATE alphas
            SET structural_signature_json = ?
            WHERE run_id = ? AND alpha_id = ?
            """,
            (structural_signature_json, run_id, alpha_id),
        )
        self.connection.commit()

    def list_existing_normalized_expressions(self, run_id: str) -> set[str]:
        rows = self.connection.execute(
            "SELECT normalized_expression FROM alphas WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return {row["normalized_expression"] for row in rows}

    def get_parent_refs(self, run_id: str) -> dict[str, list[dict[str, str]]]:
        rows = self.connection.execute(
            """
            SELECT alpha_id, parent_run_id, parent_alpha_id
            FROM alpha_parents
            WHERE run_id = ?
            ORDER BY alpha_id ASC, parent_run_id ASC, parent_alpha_id ASC
            """,
            (run_id,),
        ).fetchall()
        mapping: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            mapping.setdefault(row["alpha_id"], []).append(
                {
                    "run_id": row["parent_run_id"] or run_id,
                    "alpha_id": row["parent_alpha_id"],
                }
            )
        return mapping

    def get_generation_mix(self, run_id: str) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT generation_mode, COUNT(*) AS alpha_count
            FROM alphas
            WHERE run_id = ?
            GROUP BY generation_mode
            ORDER BY alpha_count DESC, generation_mode ASC
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]
