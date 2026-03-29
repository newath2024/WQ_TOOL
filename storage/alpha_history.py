from __future__ import annotations

import json
import math
import sqlite3
from io import StringIO

import pandas as pd

from evaluation.critic import AlphaDiagnosis
from memory.pattern_memory import MemoryParent, PatternMemoryService, PatternMemorySnapshot, PatternScore
from storage.models import (
    AlphaDiagnosisRecord,
    AlphaHistoryRecord,
    AlphaPatternMembershipRecord,
    AlphaPatternRecord,
)


class AlphaHistoryStore:
    def __init__(self, connection: sqlite3.Connection, memory_service: PatternMemoryService) -> None:
        self.connection = connection
        self.memory_service = memory_service

    def persist_evaluations(
        self,
        run_id: str,
        regime_key: str,
        evaluations: list,
        pattern_decay: float,
        prior_weight: float,
        created_at: str,
    ) -> None:
        history_records: list[AlphaHistoryRecord] = []
        diagnosis_records: list[AlphaDiagnosisRecord] = []
        membership_records: list[AlphaPatternMembershipRecord] = []

        for evaluation in evaluations:
            self.connection.execute(
                "DELETE FROM alpha_diagnoses WHERE run_id = ? AND alpha_id = ?",
                (run_id, evaluation.candidate.alpha_id),
            )
            self.connection.execute(
                "DELETE FROM alpha_pattern_membership WHERE run_id = ? AND alpha_id = ?",
                (run_id, evaluation.candidate.alpha_id),
            )

        for evaluation in evaluations:
            diagnosis = evaluation.diagnosis or AlphaDiagnosis()
            parent_refs = evaluation.candidate.generation_metadata.get("parent_refs") or [
                {"alpha_id": parent_id, "run_id": run_id}
                for parent_id in evaluation.candidate.parent_ids
            ]
            observations = self.memory_service.build_observations(evaluation.structural_signature) if evaluation.structural_signature else []
            history_records.append(
                AlphaHistoryRecord(
                    run_id=run_id,
                    alpha_id=evaluation.candidate.alpha_id,
                    regime_key=regime_key,
                    expression=evaluation.candidate.expression,
                    normalized_expression=evaluation.candidate.normalized_expression,
                    generation_mode=evaluation.candidate.generation_mode,
                    generation_metadata_json=json.dumps(evaluation.candidate.generation_metadata, sort_keys=True),
                    parent_refs_json=json.dumps(parent_refs, sort_keys=True),
                    structural_signature_json=json.dumps(
                        evaluation.structural_signature.to_dict() if evaluation.structural_signature else {},
                        sort_keys=True,
                    ),
                    gene_ids_json=json.dumps(evaluation.gene_ids, sort_keys=True),
                    train_metrics_json=json.dumps(self._metrics_to_dict(evaluation.split_metrics["train"]), sort_keys=True),
                    validation_metrics_json=json.dumps(
                        self._metrics_to_dict(evaluation.split_metrics["validation"]),
                        sort_keys=True,
                    ),
                    test_metrics_json=json.dumps(self._metrics_to_dict(evaluation.split_metrics["test"]), sort_keys=True),
                    validation_signal_json=evaluation.validation_signal.to_json(orient="split", date_format="iso"),
                    validation_returns_json=evaluation.validation_returns.to_json(orient="split", date_format="iso"),
                    outcome_score=float(evaluation.outcome_score),
                    behavioral_novelty_score=float(evaluation.behavioral_novelty_score),
                    passed_filters=bool(evaluation.passed_filters),
                    selected=bool("selected_top_alpha" in diagnosis.success_tags),
                    submission_pass_count=int(evaluation.submission_passes),
                    diagnosis_summary_json=json.dumps(diagnosis.to_dict(), sort_keys=True),
                    rejection_reasons_json=json.dumps(evaluation.rejection_reasons, sort_keys=True),
                    created_at=created_at,
                )
            )
            for tag in diagnosis.fail_tags:
                diagnosis_records.append(
                    AlphaDiagnosisRecord(
                        run_id=run_id,
                        alpha_id=evaluation.candidate.alpha_id,
                        tag_type="fail",
                        tag=tag,
                        created_at=created_at,
                    )
                )
            for tag in diagnosis.success_tags:
                diagnosis_records.append(
                    AlphaDiagnosisRecord(
                        run_id=run_id,
                        alpha_id=evaluation.candidate.alpha_id,
                        tag_type="success",
                        tag=tag,
                        created_at=created_at,
                    )
                )
            for hint in diagnosis.mutation_hints:
                diagnosis_records.append(
                    AlphaDiagnosisRecord(
                        run_id=run_id,
                        alpha_id=evaluation.candidate.alpha_id,
                        tag_type="hint",
                        tag=hint.hint,
                        created_at=created_at,
                    )
                )
            for observation in observations:
                membership_records.append(
                    AlphaPatternMembershipRecord(
                        run_id=run_id,
                        alpha_id=evaluation.candidate.alpha_id,
                        regime_key=regime_key,
                        pattern_id=observation.pattern_id,
                        pattern_kind=observation.pattern_kind,
                        pattern_value=observation.pattern_value,
                        created_at=created_at,
                    )
                )

        self._upsert_history(history_records)
        self._replace_diagnoses(diagnosis_records)
        self._replace_memberships(membership_records)
        self._rebuild_pattern_scores(regime_key=regime_key, pattern_decay=pattern_decay, prior_weight=prior_weight)
        self.connection.commit()

    def load_snapshot(self, regime_key: str, parent_pool_size: int) -> PatternMemorySnapshot:
        pattern_rows = self.connection.execute(
            "SELECT * FROM alpha_patterns WHERE regime_key = ? ORDER BY pattern_score DESC, support DESC",
            (regime_key,),
        ).fetchall()
        parent_rows = self.connection.execute(
            """
            SELECT *
            FROM alpha_history
            WHERE regime_key = ? AND passed_filters = 1
            ORDER BY outcome_score DESC, created_at DESC
            LIMIT ?
            """,
            (regime_key, parent_pool_size),
        ).fetchall()
        fail_tag_rows = self.connection.execute(
            """
            SELECT d.tag, COUNT(*) AS total_count
            FROM alpha_diagnoses d
            JOIN alpha_history h
              ON h.run_id = d.run_id AND h.alpha_id = d.alpha_id
            WHERE h.regime_key = ? AND d.tag_type = 'fail'
            GROUP BY d.tag
            ORDER BY total_count DESC, d.tag ASC
            """,
            (regime_key,),
        ).fetchall()
        patterns = {
            row["pattern_id"]: PatternScore(
                pattern_id=row["pattern_id"],
                pattern_kind=row["pattern_kind"],
                pattern_value=row["pattern_value"],
                support=int(row["support"]),
                success_count=int(row["success_count"]),
                failure_count=int(row["failure_count"]),
                avg_outcome=float(row["avg_outcome"]),
                avg_behavioral_novelty=float(row["avg_behavioral_novelty"]),
                fail_tag_counts=json.loads(row["fail_tag_counts_json"]),
                pattern_score=float(row["pattern_score"]),
            )
            for row in pattern_rows
        }
        top_parents = tuple(self._memory_parent_from_row(row) for row in parent_rows)
        fail_tag_counts = {row["tag"]: int(row["total_count"]) for row in fail_tag_rows}
        return PatternMemorySnapshot(
            regime_key=regime_key,
            patterns=patterns,
            top_parents=top_parents,
            fail_tag_counts=fail_tag_counts,
        )

    def get_novelty_references(self, regime_key: str, limit: int) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT alpha_id, validation_signal_json, validation_returns_json
            FROM alpha_history
            WHERE regime_key = ? AND passed_filters = 1
            ORDER BY outcome_score DESC, created_at DESC
            LIMIT ?
            """,
            (regime_key, limit),
        ).fetchall()
        return [
            {
                "alpha_id": row["alpha_id"],
                "validation_signal": self._frame_from_json(row["validation_signal_json"]),
                "validation_returns": self._series_from_json(row["validation_returns_json"]),
            }
            for row in rows
        ]

    def get_top_patterns(self, regime_key: str, limit: int, pattern_kind: str | None = None) -> list[dict]:
        if pattern_kind:
            rows = self.connection.execute(
                """
                SELECT * FROM alpha_patterns
                WHERE regime_key = ? AND pattern_kind = ?
                ORDER BY pattern_score DESC, support DESC, pattern_value ASC
                LIMIT ?
                """,
                (regime_key, pattern_kind, limit),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT * FROM alpha_patterns
                WHERE regime_key = ?
                ORDER BY pattern_score DESC, support DESC, pattern_value ASC
                LIMIT ?
                """,
                (regime_key, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_failed_patterns(self, regime_key: str, limit: int) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT * FROM alpha_patterns
            WHERE regime_key = ?
            ORDER BY failure_count DESC, pattern_score ASC, support DESC
            LIMIT ?
            """,
            (regime_key, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_top_genes(self, regime_key: str, limit: int) -> list[dict]:
        return self.get_top_patterns(regime_key=regime_key, limit=limit, pattern_kind="subexpression")

    def get_lineage(self, run_id: str, alpha_id: str) -> list[dict]:
        rows = self.connection.execute(
            """
            WITH RECURSIVE lineage(depth, run_id, alpha_id) AS (
                SELECT 0 AS depth, ? AS run_id, ? AS alpha_id
                UNION ALL
                SELECT lineage.depth + 1, COALESCE(p.parent_run_id, lineage.run_id), p.parent_alpha_id
                FROM alpha_parents p
                JOIN lineage
                  ON p.run_id = lineage.run_id AND p.alpha_id = lineage.alpha_id
                WHERE lineage.depth < 12
            )
            SELECT
                lineage.depth,
                lineage.run_id,
                lineage.alpha_id,
                h.expression,
                h.outcome_score,
                h.diagnosis_summary_json
            FROM lineage
            LEFT JOIN alpha_history h
              ON h.run_id = lineage.run_id AND h.alpha_id = lineage.alpha_id
            ORDER BY lineage.depth ASC, lineage.run_id ASC, lineage.alpha_id ASC
            """,
            (run_id, alpha_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_run_fail_tag_summary(self, run_id: str) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT tag, COUNT(*) AS total_count
            FROM alpha_diagnoses
            WHERE run_id = ? AND tag_type = 'fail'
            GROUP BY tag
            ORDER BY total_count DESC, tag ASC
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_run_rejection_reason_summary(self, run_id: str, limit: int) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT rejection_reasons_json
            FROM alpha_history
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            for reason in json.loads(row["rejection_reasons_json"] or "[]"):
                counts[reason] = counts.get(reason, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [{"reason": reason, "total_count": total} for reason, total in ordered[:limit]]

    def _upsert_history(self, records: list[AlphaHistoryRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO alpha_history
                (run_id, alpha_id, regime_key, expression, normalized_expression, generation_mode, generation_metadata_json,
                 parent_refs_json, structural_signature_json, gene_ids_json, train_metrics_json, validation_metrics_json,
                 test_metrics_json, validation_signal_json, validation_returns_json, outcome_score, behavioral_novelty_score,
                 passed_filters, selected, submission_pass_count, diagnosis_summary_json, rejection_reasons_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, alpha_id) DO UPDATE SET
                    regime_key = excluded.regime_key,
                    expression = excluded.expression,
                    normalized_expression = excluded.normalized_expression,
                    generation_mode = excluded.generation_mode,
                    generation_metadata_json = excluded.generation_metadata_json,
                    parent_refs_json = excluded.parent_refs_json,
                    structural_signature_json = excluded.structural_signature_json,
                    gene_ids_json = excluded.gene_ids_json,
                    train_metrics_json = excluded.train_metrics_json,
                    validation_metrics_json = excluded.validation_metrics_json,
                    test_metrics_json = excluded.test_metrics_json,
                    validation_signal_json = excluded.validation_signal_json,
                    validation_returns_json = excluded.validation_returns_json,
                    outcome_score = excluded.outcome_score,
                    behavioral_novelty_score = excluded.behavioral_novelty_score,
                    passed_filters = excluded.passed_filters,
                    selected = excluded.selected,
                    submission_pass_count = excluded.submission_pass_count,
                    diagnosis_summary_json = excluded.diagnosis_summary_json,
                    rejection_reasons_json = excluded.rejection_reasons_json,
                    created_at = excluded.created_at
                """,
                (
                    record.run_id,
                    record.alpha_id,
                    record.regime_key,
                    record.expression,
                    record.normalized_expression,
                    record.generation_mode,
                    record.generation_metadata_json,
                    record.parent_refs_json,
                    record.structural_signature_json,
                    record.gene_ids_json,
                    record.train_metrics_json,
                    record.validation_metrics_json,
                    record.test_metrics_json,
                    record.validation_signal_json,
                    record.validation_returns_json,
                    record.outcome_score,
                    int(record.behavioral_novelty_score * 1_000_000) / 1_000_000.0,
                    int(record.passed_filters),
                    int(record.selected),
                    record.submission_pass_count,
                    record.diagnosis_summary_json,
                    record.rejection_reasons_json,
                    record.created_at,
                ),
            )

    def _replace_diagnoses(self, records: list[AlphaDiagnosisRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO alpha_diagnoses (run_id, alpha_id, tag_type, tag, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record.run_id, record.alpha_id, record.tag_type, record.tag, record.created_at),
            )

    def _replace_memberships(self, records: list[AlphaPatternMembershipRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO alpha_pattern_membership
                (run_id, alpha_id, regime_key, pattern_id, pattern_kind, pattern_value, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.alpha_id,
                    record.regime_key,
                    record.pattern_id,
                    record.pattern_kind,
                    record.pattern_value,
                    record.created_at,
                ),
            )

    def _rebuild_pattern_scores(self, regime_key: str, pattern_decay: float, prior_weight: float) -> None:
        rows = self.connection.execute(
            """
            SELECT
                m.pattern_id,
                m.pattern_kind,
                m.pattern_value,
                h.outcome_score,
                h.behavioral_novelty_score,
                h.passed_filters,
                h.selected,
                h.diagnosis_summary_json,
                h.created_at
            FROM alpha_pattern_membership m
            JOIN alpha_history h
              ON h.run_id = m.run_id AND h.alpha_id = m.alpha_id
            WHERE m.regime_key = ?
            ORDER BY m.pattern_id ASC, h.created_at ASC
            """,
            (regime_key,),
        ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["pattern_id"], []).append(row)

        self.connection.execute("DELETE FROM alpha_patterns WHERE regime_key = ?", (regime_key,))
        for pattern_id, group_rows in grouped.items():
            weights = [pattern_decay ** (len(group_rows) - index - 1) for index in range(len(group_rows))]
            total_weight = sum(weights) or 1.0
            avg_outcome = sum(weight * float(row["outcome_score"]) for weight, row in zip(weights, group_rows, strict=True)) / total_weight
            avg_novelty = (
                sum(weight * float(row["behavioral_novelty_score"]) for weight, row in zip(weights, group_rows, strict=True))
                / total_weight
            )
            support = len(group_rows)
            success_count = sum(int(row["passed_filters"]) or int(row["selected"]) for row in group_rows)
            failure_count = support - success_count
            fail_tag_counts: dict[str, int] = {}
            for row in group_rows:
                summary = json.loads(row["diagnosis_summary_json"])
                for tag in summary.get("fail_tags", []):
                    fail_tag_counts[tag] = fail_tag_counts.get(tag, 0) + 1
            smoothed_score = (support * avg_outcome) / (support + prior_weight)
            pattern_score = float(smoothed_score + 0.05 * math.log1p(success_count))
            pattern_record = AlphaPatternRecord(
                regime_key=regime_key,
                pattern_id=pattern_id,
                pattern_kind=group_rows[0]["pattern_kind"],
                pattern_value=group_rows[0]["pattern_value"],
                support=support,
                success_count=success_count,
                failure_count=failure_count,
                avg_outcome=float(avg_outcome),
                avg_behavioral_novelty=float(avg_novelty),
                fail_tag_counts_json=json.dumps(fail_tag_counts, sort_keys=True),
                pattern_score=pattern_score,
                last_seen_at=max(str(row["created_at"]) for row in group_rows),
                updated_at=max(str(row["created_at"]) for row in group_rows),
            )
            self.connection.execute(
                """
                INSERT INTO alpha_patterns
                (regime_key, pattern_id, pattern_kind, pattern_value, support, success_count, failure_count,
                 avg_outcome, avg_behavioral_novelty, fail_tag_counts_json, pattern_score, last_seen_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern_record.regime_key,
                    pattern_record.pattern_id,
                    pattern_record.pattern_kind,
                    pattern_record.pattern_value,
                    pattern_record.support,
                    pattern_record.success_count,
                    pattern_record.failure_count,
                    pattern_record.avg_outcome,
                    pattern_record.avg_behavioral_novelty,
                    pattern_record.fail_tag_counts_json,
                    pattern_record.pattern_score,
                    pattern_record.last_seen_at,
                    pattern_record.updated_at,
                ),
            )

    def _memory_parent_from_row(self, row: sqlite3.Row) -> MemoryParent:
        summary = json.loads(row["diagnosis_summary_json"])
        return MemoryParent(
            run_id=row["run_id"],
            alpha_id=row["alpha_id"],
            expression=row["expression"],
            normalized_expression=row["normalized_expression"],
            generation_mode=row["generation_mode"],
            generation_metadata=json.loads(row["generation_metadata_json"]),
            parent_refs=tuple(json.loads(row["parent_refs_json"])),
            family_signature=json.loads(row["structural_signature_json"]).get("family_signature", ""),
            outcome_score=float(row["outcome_score"]),
            behavioral_novelty_score=float(row["behavioral_novelty_score"]),
            fail_tags=tuple(summary.get("fail_tags", [])),
            success_tags=tuple(summary.get("success_tags", [])),
            mutation_hints=tuple(hint["hint"] for hint in summary.get("mutation_hints", [])),
        )

    def _metrics_to_dict(self, metrics) -> dict:
        return {
            "sharpe": metrics.sharpe,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "average_return": metrics.average_return,
            "turnover": metrics.turnover,
            "observation_count": metrics.observation_count,
            "cumulative_return": metrics.cumulative_return,
            "fitness": metrics.fitness,
        }

    def _frame_from_json(self, payload: str) -> pd.DataFrame:
        frame = pd.read_json(StringIO(payload), orient="split")
        frame.index = pd.to_datetime(frame.index)
        return frame

    def _series_from_json(self, payload: str) -> pd.Series:
        series = pd.read_json(StringIO(payload), orient="split", typ="series")
        series.index = pd.to_datetime(series.index)
        return series
