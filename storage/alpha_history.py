from __future__ import annotations

import json
import math
import sqlite3
from io import StringIO

import pandas as pd

from core.config import RegionLearningConfig
from domain.metrics import ObjectiveVector, StructuralSignature
from evaluation.critic import AlphaDiagnosis
from memory.case_memory import CaseMemoryService
from memory.pattern_memory import (
    MemoryParent,
    PatternMemoryService,
    PatternMemorySnapshot,
    PatternScore,
)
from storage.models import (
    AlphaCaseRecord,
    AlphaDiagnosisRecord,
    AlphaHistoryRecord,
    AlphaPatternMembershipRecord,
    AlphaPatternRecord,
)


class AlphaHistoryStore:
    def __init__(self, connection: sqlite3.Connection, memory_service: PatternMemoryService) -> None:
        self.connection = connection
        self.memory_service = memory_service
        self.case_memory_service = CaseMemoryService()

    def persist_evaluations(
        self,
        run_id: str,
        regime_key: str,
        *,
        region: str = "",
        global_regime_key: str = "",
        market_regime_key: str = "",
        effective_regime_key: str = "",
        regime_label: str = "unknown",
        regime_confidence: float = 0.0,
        evaluations: list,
        pattern_decay: float,
        prior_weight: float,
        created_at: str,
    ) -> None:
        history_records: list[AlphaHistoryRecord] = []
        diagnosis_records: list[AlphaDiagnosisRecord] = []
        membership_records: list[AlphaPatternMembershipRecord] = []
        case_records: list[AlphaCaseRecord] = []

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
            observations = (
                self.memory_service.build_observations(
                    evaluation.structural_signature,
                    template_name=evaluation.candidate.template_name,
                    rejection_reasons=evaluation.rejection_reasons,
                    generation_metadata=evaluation.candidate.generation_metadata,
                    success_tags=diagnosis.success_tags,
                    fail_tags=diagnosis.fail_tags,
                )
                if evaluation.structural_signature
                else []
            )
            history_records.append(
                AlphaHistoryRecord(
                    run_id=run_id,
                    alpha_id=evaluation.candidate.alpha_id,
                    region=region,
                    regime_key=regime_key,
                    global_regime_key=global_regime_key,
                    market_regime_key=market_regime_key,
                    effective_regime_key=effective_regime_key or regime_key,
                    regime_label=regime_label,
                    regime_confidence=float(regime_confidence),
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
                    metric_source="local_backtest",
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
                        region=region,
                        regime_key=regime_key,
                        global_regime_key=global_regime_key,
                        pattern_id=observation.pattern_id,
                        pattern_kind=observation.pattern_kind,
                        pattern_value=observation.pattern_value,
                        created_at=created_at,
                    )
                )
            if evaluation.structural_signature:
                case_records.append(
                    self._build_case_record(
                        run_id=run_id,
                        regime_key=regime_key,
                        region=region,
                        global_regime_key=global_regime_key,
                        market_regime_key=market_regime_key,
                        effective_regime_key=effective_regime_key or regime_key,
                        regime_label=regime_label,
                        regime_confidence=regime_confidence,
                        candidate=evaluation.candidate,
                        structural_signature=evaluation.structural_signature,
                        metric_source="local_backtest",
                        simulation_context=evaluation.simulation_profile,
                        fail_tags=diagnosis.fail_tags,
                        success_tags=diagnosis.success_tags,
                        objective_vector=ObjectiveVector(
                            fitness=float(evaluation.split_metrics["validation"].fitness),
                            sharpe=float(evaluation.split_metrics["validation"].sharpe),
                            eligibility=1.0 if evaluation.submission_passes > 0 else 0.0,
                            robustness=float(evaluation.stability_score),
                            novelty=float(evaluation.behavioral_novelty_score),
                            diversity=float(evaluation.behavioral_novelty_score),
                            turnover_cost=min(1.0, max(0.0, float(evaluation.split_metrics["validation"].turnover) / 3.0)),
                            complexity_cost=min(1.0, max(0.0, float(evaluation.candidate.complexity) / 20.0)),
                        ),
                        outcome_score=float(evaluation.outcome_score),
                        created_at=created_at,
                    )
                )

        self._upsert_history(history_records)
        self._replace_diagnoses(diagnosis_records)
        self._replace_memberships(membership_records)
        self._upsert_cases(case_records)
        self._rebuild_pattern_scores(regime_key=regime_key, pattern_decay=pattern_decay, prior_weight=prior_weight)
        self.connection.commit()

    def persist_brain_outcomes(
        self,
        run_id: str,
        regime_key: str,
        *,
        region: str = "",
        global_regime_key: str = "",
        market_regime_key: str = "",
        effective_regime_key: str = "",
        regime_label: str = "unknown",
        regime_confidence: float = 0.0,
        entries: list[dict],
        pattern_decay: float,
        prior_weight: float,
        created_at: str,
    ) -> None:
        history_records: list[AlphaHistoryRecord] = []
        diagnosis_records: list[AlphaDiagnosisRecord] = []
        membership_records: list[AlphaPatternMembershipRecord] = []
        case_records: list[AlphaCaseRecord] = []
        empty_signal_json = pd.DataFrame().to_json(orient="split", date_format="iso")
        empty_returns_json = pd.Series(dtype=float).to_json(orient="split", date_format="iso")

        for entry in entries:
            candidate = entry["candidate"]
            result = entry["result"]
            diagnosis = entry["diagnosis"]
            structural_signature = entry["structural_signature"]
            parent_refs = candidate.generation_metadata.get("parent_refs") or [
                {"alpha_id": parent_id, "run_id": run_id}
                for parent_id in candidate.parent_ids
            ]
            rejection_reasons = []
            if result.rejection_reason:
                rejection_reasons.append(result.rejection_reason)
            observations = self.memory_service.build_observations(
                structural_signature,
                template_name=candidate.template_name,
                rejection_reasons=rejection_reasons,
                generation_metadata=candidate.generation_metadata,
                success_tags=diagnosis.success_tags,
                fail_tags=diagnosis.fail_tags,
            )
            metrics_payload = {
                "sharpe": result.metrics.get("sharpe"),
                "max_drawdown": result.metrics.get("drawdown"),
                "win_rate": None,
                "average_return": result.metrics.get("returns"),
                "turnover": result.metrics.get("turnover"),
                "observation_count": 0,
                "cumulative_return": result.metrics.get("returns"),
                "fitness": result.metrics.get("fitness"),
            }
            history_records.append(
                AlphaHistoryRecord(
                    run_id=run_id,
                    alpha_id=candidate.alpha_id,
                    region=region or str(result.region or ""),
                    regime_key=regime_key,
                    global_regime_key=global_regime_key,
                    market_regime_key=market_regime_key,
                    effective_regime_key=effective_regime_key or regime_key,
                    regime_label=regime_label,
                    regime_confidence=float(regime_confidence),
                    expression=candidate.expression,
                    normalized_expression=candidate.normalized_expression,
                    generation_mode=candidate.generation_mode,
                    generation_metadata_json=json.dumps(candidate.generation_metadata, sort_keys=True),
                    parent_refs_json=json.dumps(parent_refs, sort_keys=True),
                    structural_signature_json=json.dumps(structural_signature.to_dict(), sort_keys=True),
                    gene_ids_json=json.dumps(entry["gene_ids"], sort_keys=True),
                    train_metrics_json=json.dumps(metrics_payload, sort_keys=True),
                    validation_metrics_json=json.dumps(metrics_payload, sort_keys=True),
                    test_metrics_json=json.dumps(metrics_payload, sort_keys=True),
                    validation_signal_json=empty_signal_json,
                    validation_returns_json=empty_returns_json,
                    outcome_score=float(entry["outcome_score"]),
                    behavioral_novelty_score=float(entry.get("behavioral_novelty_score", 0.5)),
                    passed_filters=bool(entry["passed_filters"]),
                    selected=bool(entry["selected"]),
                    submission_pass_count=1 if result.submission_eligible else 0,
                    diagnosis_summary_json=json.dumps(diagnosis.to_dict(), sort_keys=True),
                    rejection_reasons_json=json.dumps(rejection_reasons, sort_keys=True),
                    metric_source=str(entry.get("metric_source") or "external_brain"),
                    created_at=created_at,
                )
            )
            for tag in diagnosis.fail_tags:
                diagnosis_records.append(
                    AlphaDiagnosisRecord(
                        run_id=run_id,
                        alpha_id=candidate.alpha_id,
                        tag_type="fail",
                        tag=tag,
                        created_at=created_at,
                    )
                )
            for tag in diagnosis.success_tags:
                diagnosis_records.append(
                    AlphaDiagnosisRecord(
                        run_id=run_id,
                        alpha_id=candidate.alpha_id,
                        tag_type="success",
                        tag=tag,
                        created_at=created_at,
                    )
                )
            for hint in diagnosis.mutation_hints:
                diagnosis_records.append(
                    AlphaDiagnosisRecord(
                        run_id=run_id,
                        alpha_id=candidate.alpha_id,
                        tag_type="hint",
                        tag=hint.hint,
                        created_at=created_at,
                    )
                )
            for observation in observations:
                membership_records.append(
                    AlphaPatternMembershipRecord(
                        run_id=run_id,
                        alpha_id=candidate.alpha_id,
                        region=region or str(result.region or ""),
                        regime_key=regime_key,
                        global_regime_key=global_regime_key,
                        pattern_id=observation.pattern_id,
                        pattern_kind=observation.pattern_kind,
                        pattern_value=observation.pattern_value,
                        created_at=created_at,
                    )
                )
            case_records.append(
                self._build_case_record(
                    run_id=run_id,
                    regime_key=regime_key,
                    region=region or str(result.region or ""),
                    global_regime_key=global_regime_key,
                    market_regime_key=market_regime_key,
                    effective_regime_key=effective_regime_key or regime_key,
                    regime_label=regime_label,
                    regime_confidence=regime_confidence,
                    candidate=candidate,
                    structural_signature=structural_signature,
                    metric_source=str(entry.get("metric_source") or "external_brain"),
                    simulation_context={"neutralization": result.neutralization, "decay": result.decay},
                    fail_tags=diagnosis.fail_tags,
                    success_tags=diagnosis.success_tags,
                    objective_vector=ObjectiveVector(
                        fitness=float(result.metrics.get("fitness") or 0.0),
                        sharpe=float(result.metrics.get("sharpe") or 0.0),
                        eligibility=1.0 if result.submission_eligible else 0.0,
                        robustness=1.0 if not result.rejection_reason else 0.0,
                        novelty=float(entry.get("behavioral_novelty_score", 0.5)),
                        diversity=float(entry.get("behavioral_novelty_score", 0.5)),
                        turnover_cost=min(1.0, max(0.0, float(result.metrics.get("turnover") or 0.0) / 2.0)),
                        complexity_cost=min(1.0, max(0.0, float(candidate.complexity) / 20.0)),
                    ),
                    outcome_score=float(entry["outcome_score"]),
                    created_at=created_at,
                )
            )

        self._upsert_history(history_records)
        self._replace_diagnoses(diagnosis_records)
        self._replace_memberships(membership_records)
        self._upsert_cases(case_records)
        self._rebuild_pattern_scores(regime_key=regime_key, pattern_decay=pattern_decay, prior_weight=prior_weight)
        self.connection.commit()

    def get_outcome_score(self, run_id: str, alpha_id: str) -> float | None:
        row = self.connection.execute(
            """
            SELECT outcome_score
            FROM alpha_history
            WHERE run_id = ? AND alpha_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (run_id, alpha_id),
        ).fetchone()
        if row is None or row["outcome_score"] is None:
            return None
        return float(row["outcome_score"])

    def load_snapshot(
        self,
        regime_key: str,
        parent_pool_size: int,
        *,
        region: str = "",
        global_regime_key: str = "",
        region_learning_config: RegionLearningConfig | None = None,
        pattern_decay: float = 0.98,
        prior_weight: float = 3.0,
    ) -> PatternMemorySnapshot:
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
        if not parent_rows and region_learning_config and region_learning_config.allow_global_parent_fallback and global_regime_key:
            parent_rows = self.connection.execute(
                """
                SELECT *
                FROM alpha_history
                WHERE global_regime_key = ? AND regime_key <> ? AND passed_filters = 1
                ORDER BY outcome_score DESC, created_at DESC
                LIMIT ?
                """,
                (global_regime_key, regime_key, parent_pool_size),
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
        local_sample_count = int(
            self.connection.execute(
                "SELECT COUNT(*) AS total FROM alpha_history WHERE regime_key = ?",
                (regime_key,),
            ).fetchone()["total"]
            or 0
        )
        global_pattern_rows = []
        global_fail_tag_rows = []
        global_sample_count = 0
        if global_regime_key:
            global_pattern_rows = self.connection.execute(
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
                WHERE h.global_regime_key = ? AND h.regime_key <> ?
                ORDER BY m.pattern_id ASC, h.created_at ASC
                """,
                (global_regime_key, regime_key),
            ).fetchall()
            global_fail_tag_rows = self.connection.execute(
                """
                SELECT d.tag, COUNT(*) AS total_count
                FROM alpha_diagnoses d
                JOIN alpha_history h
                  ON h.run_id = d.run_id AND h.alpha_id = d.alpha_id
                WHERE h.global_regime_key = ? AND h.regime_key <> ? AND d.tag_type = 'fail'
                GROUP BY d.tag
                ORDER BY total_count DESC, d.tag ASC
                """,
                (global_regime_key, regime_key),
            ).fetchall()
            global_sample_count = int(
                self.connection.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM alpha_history
                    WHERE global_regime_key = ? AND regime_key <> ?
                    """,
                    (global_regime_key, regime_key),
                ).fetchone()["total"]
                or 0
            )
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
        global_patterns = self._build_pattern_scores_from_membership_rows(
            global_pattern_rows,
            pattern_decay=pattern_decay,
            prior_weight=prior_weight,
        )
        top_parents = tuple(self._memory_parent_from_row(row) for row in parent_rows)
        fail_tag_counts = {row["tag"]: int(row["total_count"]) for row in fail_tag_rows}
        global_fail_tag_counts = {row["tag"]: int(row["total_count"]) for row in global_fail_tag_rows}
        blend = self.memory_service.compute_blend_diagnostics(
            scope="pattern",
            local_samples=local_sample_count,
            global_samples=global_sample_count,
            config=region_learning_config or RegionLearningConfig(),
        )
        return PatternMemorySnapshot(
            regime_key=regime_key,
            global_regime_key=global_regime_key,
            region=region,
            patterns=patterns,
            top_parents=top_parents,
            fail_tag_counts=fail_tag_counts,
            sample_count=local_sample_count,
            global_patterns=global_patterns,
            global_fail_tag_counts=global_fail_tag_counts,
            global_sample_count=global_sample_count,
            blend=blend,
        )

    def load_case_snapshot(
        self,
        regime_key: str,
        *,
        region: str = "",
        global_regime_key: str = "",
        region_learning_config: RegionLearningConfig | None = None,
        limit: int = 500,
    ):
        rows = self.connection.execute(
            """
            SELECT *
            FROM alpha_cases
            WHERE regime_key = ?
            ORDER BY created_at DESC, outcome_score DESC
            LIMIT ?
            """,
            (regime_key, limit),
        ).fetchall()
        global_rows = self.connection.execute(
            """
            SELECT *
            FROM alpha_cases
            WHERE global_regime_key = ? AND regime_key <> ?
            ORDER BY created_at DESC, outcome_score DESC
            LIMIT ?
            """,
            (global_regime_key, regime_key, limit),
        ).fetchall() if global_regime_key else []
        local_sample_count = int(
            self.connection.execute(
                "SELECT COUNT(*) AS total FROM alpha_cases WHERE regime_key = ?",
                (regime_key,),
            ).fetchone()["total"]
            or 0
        )
        global_sample_count = int(
            self.connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM alpha_cases
                WHERE global_regime_key = ? AND regime_key <> ?
                """,
                (global_regime_key, regime_key),
            ).fetchone()["total"]
            or 0
        ) if global_regime_key else 0
        records = [
            self.case_memory_service.record_from_persisted_payload(
                row=dict(row),
                structural_signature=self._structural_signature_from_json(row["structural_signature_json"]),
            )
            for row in rows
        ]
        global_records = [
            self.case_memory_service.record_from_persisted_payload(
                row=dict(row),
                structural_signature=self._structural_signature_from_json(row["structural_signature_json"]),
            )
            for row in global_rows
        ]
        blend = self.memory_service.compute_blend_diagnostics(
            scope="case",
            local_samples=local_sample_count,
            global_samples=global_sample_count,
            config=region_learning_config or RegionLearningConfig(),
        )
        return self.case_memory_service.build_snapshot(
            regime_key,
            records,
            region=region,
            global_regime_key=global_regime_key,
            global_records=global_records,
            blend=blend,
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
                (run_id, alpha_id, region, regime_key, global_regime_key, market_regime_key, effective_regime_key,
                 regime_label, regime_confidence, expression, normalized_expression, generation_mode, generation_metadata_json,
                 parent_refs_json, structural_signature_json, gene_ids_json, train_metrics_json, validation_metrics_json,
                 test_metrics_json, validation_signal_json, validation_returns_json, outcome_score, behavioral_novelty_score,
                 passed_filters, selected, submission_pass_count, diagnosis_summary_json, rejection_reasons_json, metric_source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, alpha_id) DO UPDATE SET
                    region = excluded.region,
                    regime_key = excluded.regime_key,
                    global_regime_key = excluded.global_regime_key,
                    market_regime_key = excluded.market_regime_key,
                    effective_regime_key = excluded.effective_regime_key,
                    regime_label = excluded.regime_label,
                    regime_confidence = excluded.regime_confidence,
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
                    metric_source = excluded.metric_source,
                    created_at = excluded.created_at
                """,
                (
                    record.run_id,
                    record.alpha_id,
                    record.region,
                    record.regime_key,
                    record.global_regime_key,
                    record.market_regime_key,
                    record.effective_regime_key,
                    record.regime_label,
                    record.regime_confidence,
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
                    record.metric_source,
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
                (run_id, alpha_id, region, regime_key, global_regime_key, pattern_id, pattern_kind, pattern_value, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.alpha_id,
                    record.region,
                    record.regime_key,
                    record.global_regime_key,
                    record.pattern_id,
                    record.pattern_kind,
                    record.pattern_value,
                    record.created_at,
                ),
            )

    def _upsert_cases(self, records: list[AlphaCaseRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO alpha_cases
                (run_id, alpha_id, region, regime_key, global_regime_key, market_regime_key, effective_regime_key,
                 regime_label, regime_confidence, metric_source, family_signature, structural_signature_json, genome_hash, genome_json,
                 motif, field_families_json, operator_path_json, complexity_bucket, turnover_bucket, horizon_bucket, mutation_mode,
                 parent_family_signatures_json, fail_tags_json, success_tags_json, objective_vector_json, outcome_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, alpha_id, metric_source) DO UPDATE SET
                    region = excluded.region,
                    regime_key = excluded.regime_key,
                    global_regime_key = excluded.global_regime_key,
                    market_regime_key = excluded.market_regime_key,
                    effective_regime_key = excluded.effective_regime_key,
                    regime_label = excluded.regime_label,
                    regime_confidence = excluded.regime_confidence,
                    family_signature = excluded.family_signature,
                    structural_signature_json = excluded.structural_signature_json,
                    genome_hash = excluded.genome_hash,
                    genome_json = excluded.genome_json,
                    motif = excluded.motif,
                    field_families_json = excluded.field_families_json,
                    operator_path_json = excluded.operator_path_json,
                    complexity_bucket = excluded.complexity_bucket,
                    turnover_bucket = excluded.turnover_bucket,
                    horizon_bucket = excluded.horizon_bucket,
                    mutation_mode = excluded.mutation_mode,
                    parent_family_signatures_json = excluded.parent_family_signatures_json,
                    fail_tags_json = excluded.fail_tags_json,
                    success_tags_json = excluded.success_tags_json,
                    objective_vector_json = excluded.objective_vector_json,
                    outcome_score = excluded.outcome_score,
                    created_at = excluded.created_at
                """,
                (
                    record.run_id,
                    record.alpha_id,
                    record.region,
                    record.regime_key,
                    record.global_regime_key,
                    record.market_regime_key,
                    record.effective_regime_key,
                    record.regime_label,
                    record.regime_confidence,
                    record.metric_source,
                    record.family_signature,
                    record.structural_signature_json,
                    record.genome_hash,
                    record.genome_json,
                    record.motif,
                    record.field_families_json,
                    record.operator_path_json,
                    record.complexity_bucket,
                    record.turnover_bucket,
                    record.horizon_bucket,
                    record.mutation_mode,
                    record.parent_family_signatures_json,
                    record.fail_tags_json,
                    record.success_tags_json,
                    record.objective_vector_json,
                    record.outcome_score,
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
        self.connection.execute("DELETE FROM alpha_patterns WHERE regime_key = ?", (regime_key,))
        for pattern_record in self._build_pattern_records_from_membership_rows(
            regime_key=regime_key,
            rows=rows,
            pattern_decay=pattern_decay,
            prior_weight=prior_weight,
        ):
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

    def _build_pattern_scores_from_membership_rows(
        self,
        rows: list[sqlite3.Row],
        *,
        pattern_decay: float,
        prior_weight: float,
    ) -> dict[str, PatternScore]:
        return {
            record.pattern_id: PatternScore(
                pattern_id=record.pattern_id,
                pattern_kind=record.pattern_kind,
                pattern_value=record.pattern_value,
                support=record.support,
                success_count=record.success_count,
                failure_count=record.failure_count,
                avg_outcome=record.avg_outcome,
                avg_behavioral_novelty=record.avg_behavioral_novelty,
                fail_tag_counts=json.loads(record.fail_tag_counts_json),
                pattern_score=record.pattern_score,
            )
            for record in self._build_pattern_records_from_membership_rows(
                regime_key="",
                rows=rows,
                pattern_decay=pattern_decay,
                prior_weight=prior_weight,
            )
        }

    def _build_pattern_records_from_membership_rows(
        self,
        *,
        regime_key: str,
        rows: list[sqlite3.Row],
        pattern_decay: float,
        prior_weight: float,
    ) -> list[AlphaPatternRecord]:
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["pattern_id"], []).append(row)
        records: list[AlphaPatternRecord] = []
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
            records.append(
                AlphaPatternRecord(
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
            )
        return records

    def _memory_parent_from_row(self, row: sqlite3.Row) -> MemoryParent:
        summary = json.loads(row["diagnosis_summary_json"])
        structural_signature = self._structural_signature_from_json(row["structural_signature_json"])
        return MemoryParent(
            run_id=row["run_id"],
            alpha_id=row["alpha_id"],
            expression=row["expression"],
            normalized_expression=row["normalized_expression"],
            generation_mode=row["generation_mode"],
            generation_metadata=json.loads(row["generation_metadata_json"]),
            parent_refs=tuple(json.loads(row["parent_refs_json"])),
            family_signature=structural_signature.family_signature,
            outcome_score=float(row["outcome_score"]),
            behavioral_novelty_score=float(row["behavioral_novelty_score"]),
            fail_tags=tuple(summary.get("fail_tags", [])),
            success_tags=tuple(summary.get("success_tags", [])),
            mutation_hints=tuple(hint["hint"] for hint in summary.get("mutation_hints", [])),
            structural_signature=structural_signature,
        )

    def _build_case_record(
        self,
        *,
        run_id: str,
        regime_key: str,
        region: str,
        global_regime_key: str,
        market_regime_key: str,
        effective_regime_key: str,
        regime_label: str,
        regime_confidence: float,
        candidate,
        structural_signature,
        metric_source: str,
        simulation_context: dict[str, object] | None = None,
        fail_tags: list[str] | tuple[str, ...],
        success_tags: list[str] | tuple[str, ...],
        objective_vector: ObjectiveVector,
        outcome_score: float,
        created_at: str,
    ) -> AlphaCaseRecord:
        metadata = dict(candidate.generation_metadata)
        raw_genome_payload = metadata.get("genome") if isinstance(metadata.get("genome"), dict) else {}
        genome_payload = dict(raw_genome_payload)
        genome_payload["_case_memory_context"] = self._case_memory_context(
            simulation_context=simulation_context,
            generation_metadata=metadata,
        )
        parent_family_signatures = [
            str(parent.get("family_signature") or "")
            for parent in metadata.get("parent_refs", [])
            if parent.get("family_signature")
        ]
        return AlphaCaseRecord(
            run_id=run_id,
            alpha_id=candidate.alpha_id,
            region=region,
            regime_key=regime_key,
            global_regime_key=global_regime_key,
            market_regime_key=market_regime_key,
            effective_regime_key=effective_regime_key,
            regime_label=regime_label,
            regime_confidence=float(regime_confidence),
            metric_source=metric_source,
            family_signature=structural_signature.family_signature,
            structural_signature_json=json.dumps(structural_signature.to_dict(), sort_keys=True),
            genome_hash=str(metadata.get("genome_hash") or ""),
            genome_json=json.dumps(genome_payload, sort_keys=True),
            motif=str(metadata.get("motif") or structural_signature.motif or candidate.template_name or ""),
            field_families_json=json.dumps(list(structural_signature.field_families), sort_keys=True),
            operator_path_json=json.dumps(list(structural_signature.operator_path), sort_keys=True),
            complexity_bucket=structural_signature.complexity_bucket,
            turnover_bucket=structural_signature.turnover_bucket,
            horizon_bucket=structural_signature.horizon_bucket,
            mutation_mode=str(metadata.get("mutation_mode") or candidate.generation_mode or ""),
            parent_family_signatures_json=json.dumps(parent_family_signatures, sort_keys=True),
            fail_tags_json=json.dumps(list(fail_tags), sort_keys=True),
            success_tags_json=json.dumps(list(success_tags), sort_keys=True),
            objective_vector_json=json.dumps(objective_vector.to_dict(), sort_keys=True),
            outcome_score=float(outcome_score),
            created_at=created_at,
        )

    def _case_memory_context(
        self,
        *,
        simulation_context: dict[str, object] | None,
        generation_metadata: dict[str, object],
    ) -> dict[str, object]:
        payload = simulation_context if isinstance(simulation_context, dict) else {}
        return {
            "neutralization": self._normalize_case_neutralization(
                payload.get("neutralization")
                or generation_metadata.get("neutralization")
                or generation_metadata.get("sim_neutralization")
            ),
            "decay": self._normalize_case_decay(
                payload.get("decay")
                or generation_metadata.get("decay")
                or generation_metadata.get("sim_decay")
            ),
        }

    @staticmethod
    def _normalize_case_neutralization(value: object) -> str:
        normalized = str(value or "").strip().lower()
        return normalized or "none"

    @staticmethod
    def _normalize_case_decay(value: object) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def _structural_signature_from_json(self, payload: str) -> StructuralSignature:
        values = json.loads(payload or "{}")
        if not values.get("operators"):
            return StructuralSignature(
                operators=(),
                operator_families=(),
                operator_path=(),
                fields=(),
                field_families=(),
                lookbacks=(),
                wrappers=(),
                depth=0,
                complexity=0,
                complexity_bucket="",
                horizon_bucket="",
                turnover_bucket="",
                motif="",
                family_signature="",
                subexpressions=(),
            )
        return self._signature_from_payload(values)

    def _signature_from_payload(self, payload: dict) -> StructuralSignature:
        return StructuralSignature(
            operators=tuple(payload.get("operators", ())),
            operator_families=tuple(payload.get("operator_families", ())),
            operator_path=tuple(payload.get("operator_path", ())),
            fields=tuple(payload.get("fields", ())),
            field_families=tuple(payload.get("field_families", ())),
            lookbacks=tuple(payload.get("lookbacks", ())),
            wrappers=tuple(payload.get("wrappers", ())),
            depth=int(payload.get("depth", 0) or 0),
            complexity=int(payload.get("complexity", 0) or 0),
            complexity_bucket=str(payload.get("complexity_bucket", "")),
            horizon_bucket=str(payload.get("horizon_bucket", "")),
            turnover_bucket=str(payload.get("turnover_bucket", "")),
            motif=str(payload.get("motif", "")),
            family_signature=str(payload.get("family_signature", "")),
            subexpressions=tuple(payload.get("subexpressions", ())),
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
