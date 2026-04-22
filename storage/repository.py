from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from io import StringIO
from typing import Any

import pandas as pd

from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemoryService
from storage.alpha_history import AlphaHistoryStore
from storage.brain_result_store import BrainResultStore
from storage.models import (
    AlphaRecord,
    CrowdingScoreRecord,
    DuplicateDecisionRecord,
    FieldCatalogRecord,
    MetricRecord,
    MutationOutcomeRecord,
    RegimeSnapshotRecord,
    RunFieldScoreRecord,
    RunRecord,
    SelectionRecord,
    SelectionScoreRecord,
    SimulationCacheRecord,
    StageMetricRecord,
    SubmissionTestRecord,
)
from storage.service_runtime_store import ServiceRuntimeStore
from storage.service_dispatch_queue_store import ServiceDispatchQueueStore
from storage.sqlite import connect_sqlite
from storage.submission_store import SubmissionStore


class SQLiteRepository:
    def __init__(self, path: str) -> None:
        self.connection = connect_sqlite(path)
        self._memory_service = PatternMemoryService()
        self.alpha_history = AlphaHistoryStore(self.connection, self._memory_service)
        self.submissions = SubmissionStore(self.connection)
        self.brain_results = BrainResultStore(self.connection)
        self.service_runtime = ServiceRuntimeStore(self.connection)
        self.service_dispatch_queue = ServiceDispatchQueueStore(self.connection)

    def close(self) -> None:
        self.connection.close()

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

    def list_existing_normalized_expressions(self, run_id: str) -> set[str]:
        rows = self.connection.execute(
            "SELECT normalized_expression FROM alphas WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return {row["normalized_expression"] for row in rows}

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

    def save_metrics(self, metric_records: list[MetricRecord]) -> None:
        for record in metric_records:
            self.connection.execute(
                """
                INSERT INTO metrics
                (run_id, alpha_id, split, sharpe, max_drawdown, win_rate, average_return, turnover,
                 observation_count, cumulative_return, fitness, stability_score, passed_filters,
                 simulation_signature, simulation_config_snapshot, delay_mode, neutralization,
                 neutralization_profile, submission_pass_count, cache_hit, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, alpha_id, split) DO UPDATE SET
                    sharpe = excluded.sharpe,
                    max_drawdown = excluded.max_drawdown,
                    win_rate = excluded.win_rate,
                    average_return = excluded.average_return,
                    turnover = excluded.turnover,
                    observation_count = excluded.observation_count,
                    cumulative_return = excluded.cumulative_return,
                    fitness = excluded.fitness,
                    stability_score = excluded.stability_score,
                    passed_filters = excluded.passed_filters,
                    simulation_signature = excluded.simulation_signature,
                    simulation_config_snapshot = excluded.simulation_config_snapshot,
                    delay_mode = excluded.delay_mode,
                    neutralization = excluded.neutralization,
                    neutralization_profile = excluded.neutralization_profile,
                    submission_pass_count = excluded.submission_pass_count,
                    cache_hit = excluded.cache_hit,
                    created_at = excluded.created_at
                """,
                (
                    record.run_id,
                    record.alpha_id,
                    record.split,
                    record.sharpe,
                    record.max_drawdown,
                    record.win_rate,
                    record.average_return,
                    record.turnover,
                    record.observation_count,
                    record.cumulative_return,
                    record.fitness,
                    record.stability_score,
                    int(record.passed_filters),
                    record.simulation_signature,
                    record.simulation_config_snapshot,
                    record.delay_mode,
                    record.neutralization,
                    record.neutralization_profile,
                    record.submission_pass_count,
                    int(record.cache_hit),
                    record.created_at,
                ),
            )
        if metric_records:
            run_id = metric_records[0].run_id
            self.connection.execute(
                "UPDATE alphas SET status = 'evaluated' WHERE run_id = ? AND alpha_id IN (SELECT alpha_id FROM metrics WHERE run_id = ?)",
                (run_id, run_id),
            )
        self.connection.commit()

    def replace_submission_tests(self, run_id: str, alpha_id: str, records: list[SubmissionTestRecord]) -> None:
        self.connection.execute(
            "DELETE FROM submission_tests WHERE run_id = ? AND alpha_id = ?",
            (run_id, alpha_id),
        )
        for record in records:
            self.connection.execute(
                """
                INSERT INTO submission_tests (run_id, alpha_id, test_name, passed, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.alpha_id,
                    record.test_name,
                    int(record.passed),
                    record.details_json,
                    record.created_at,
                ),
            )
        self.connection.commit()

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

    def save_simulation_cache(self, record: SimulationCacheRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO simulation_cache
            (simulation_signature, normalized_expression, simulation_config_snapshot, delay_mode, neutralization,
             neutralization_profile, split_metrics_json, submission_tests_json, subuniverse_metrics_json,
             validation_signal_json, validation_returns_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(simulation_signature) DO UPDATE SET
                normalized_expression = excluded.normalized_expression,
                simulation_config_snapshot = excluded.simulation_config_snapshot,
                delay_mode = excluded.delay_mode,
                neutralization = excluded.neutralization,
                neutralization_profile = excluded.neutralization_profile,
                split_metrics_json = excluded.split_metrics_json,
                submission_tests_json = excluded.submission_tests_json,
                subuniverse_metrics_json = excluded.subuniverse_metrics_json,
                validation_signal_json = excluded.validation_signal_json,
                validation_returns_json = excluded.validation_returns_json,
                created_at = excluded.created_at
            """,
            (
                record.simulation_signature,
                record.normalized_expression,
                record.simulation_config_snapshot,
                record.delay_mode,
                record.neutralization,
                record.neutralization_profile,
                record.split_metrics_json,
                record.submission_tests_json,
                record.subuniverse_metrics_json,
                record.validation_signal_json,
                record.validation_returns_json,
                record.created_at,
            ),
        )
        self.connection.commit()

    def get_cached_simulation(self, simulation_signature: str) -> SimulationCacheRecord | None:
        row = self.connection.execute(
            "SELECT * FROM simulation_cache WHERE simulation_signature = ?",
            (simulation_signature,),
        ).fetchone()
        return SimulationCacheRecord(**dict(row)) if row else None

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

    def save_stage_metrics(self, records: list[StageMetricRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO round_stage_metrics (run_id, round_index, stage, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, round_index, stage) DO UPDATE SET
                    metrics_json = excluded.metrics_json,
                    created_at = excluded.created_at
                """,
                (
                    record.run_id,
                    record.round_index,
                    record.stage,
                    record.metrics_json,
                    record.created_at,
                ),
            )
        self.connection.commit()

    def save_selection_scores(self, records: list[SelectionScoreRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO alpha_selection_scores
                (run_id, round_index, alpha_id, score_stage, composite_score, selected, rank, reason_codes_json,
                 breakdown_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, round_index, score_stage, alpha_id) DO UPDATE SET
                    composite_score = excluded.composite_score,
                    selected = excluded.selected,
                    rank = excluded.rank,
                    reason_codes_json = excluded.reason_codes_json,
                    breakdown_json = excluded.breakdown_json,
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
                    record.created_at,
                ),
            )
        self.connection.commit()

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

    def save_mutation_outcomes(self, records: list[MutationOutcomeRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO mutation_outcomes
                (run_id, child_alpha_id, parent_alpha_id, parent_run_id, mutation_mode, family_signature,
                 effective_regime_key, outcome_source, parent_post_sim_score, child_post_sim_score, outcome_delta,
                 selected_for_simulation, selected_for_mutation, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, child_alpha_id, parent_alpha_id, outcome_source) DO UPDATE SET
                    parent_run_id = excluded.parent_run_id,
                    mutation_mode = excluded.mutation_mode,
                    family_signature = excluded.family_signature,
                    effective_regime_key = excluded.effective_regime_key,
                    parent_post_sim_score = excluded.parent_post_sim_score,
                    child_post_sim_score = excluded.child_post_sim_score,
                    outcome_delta = excluded.outcome_delta,
                    selected_for_simulation = excluded.selected_for_simulation,
                    selected_for_mutation = excluded.selected_for_mutation,
                    created_at = excluded.created_at
                """,
                (
                    record.run_id,
                    record.child_alpha_id,
                    record.parent_alpha_id,
                    record.parent_run_id,
                    record.mutation_mode,
                    record.family_signature,
                    record.effective_regime_key,
                    record.outcome_source,
                    record.parent_post_sim_score,
                    record.child_post_sim_score,
                    record.outcome_delta,
                    int(record.selected_for_simulation),
                    int(record.selected_for_mutation),
                    record.created_at,
                ),
            )
        self.connection.commit()

    def get_stage_metrics(self, run_id: str) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT *
            FROM round_stage_metrics
            WHERE run_id = ?
            ORDER BY round_index ASC, stage ASC
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_recent_generation_stage_metrics(
        self,
        run_id: str,
        *,
        limit: int,
        before_round_index: int | None = None,
    ) -> list[dict]:
        if limit <= 0:
            return []
        round_filter = ""
        params: list[object] = [run_id]
        if before_round_index is not None:
            round_filter = "AND round_index < ?"
            params.append(int(before_round_index))
        params.append(int(limit))
        rows = self.connection.execute(
            f"""
            SELECT run_id, round_index, stage, metrics_json, created_at
            FROM round_stage_metrics
            WHERE run_id = ? AND stage = 'generation'
              {round_filter}
            ORDER BY round_index DESC, created_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]

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

    def list_mutation_outcomes(
        self,
        *,
        effective_regime_key: str | None = None,
        family_signature: str | None = None,
    ) -> list[dict]:
        clauses = ["1 = 1"]
        params: list[object] = []
        if effective_regime_key:
            clauses.append("effective_regime_key = ?")
            params.append(effective_regime_key)
        if family_signature:
            clauses.append("family_signature = ?")
            params.append(family_signature)
        rows = self.connection.execute(
            f"""
            SELECT *
            FROM mutation_outcomes
            WHERE {" AND ".join(clauses)}
            ORDER BY created_at DESC
            """,
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]


    def list_mutation_outcomes_with_motif(
        self,
        *,
        effective_regime_key: str | None = None,
        limit: int = 2000,
    ) -> list[dict]:
        """Return structural mutation outcomes enriched with the child alpha motif.

        ``child_motif`` is lifted from ``alpha_cases.motif`` so that
        ``MutationPolicy._motif_success_weights()`` can bias structural mutation
        toward historically high-success motifs without a raw SQL JOIN inside the
        policy layer.
        """
        clauses = ["mo.mutation_mode = 'structural'"]
        params: list[object] = []
        if effective_regime_key:
            clauses.append("mo.effective_regime_key = ?")
            params.append(effective_regime_key)
        params.append(limit)
        rows = self.connection.execute(
            f"""
            SELECT
                mo.mutation_mode,
                mo.family_signature,
                mo.effective_regime_key,
                mo.outcome_delta,
                mo.selected_for_simulation,
                mo.selected_for_mutation,
                mo.created_at,
                COALESCE(ac.motif, '') AS child_motif
            FROM mutation_outcomes mo
            LEFT JOIN alpha_cases ac
                ON ac.run_id  = mo.run_id
               AND ac.alpha_id = mo.child_alpha_id
            WHERE {" AND ".join(clauses)}
            ORDER BY mo.created_at DESC
            LIMIT ?
            """,
            tuple(params),
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

    def get_submission_tests_for_run(self, run_id: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM submission_tests WHERE run_id = ? ORDER BY alpha_id ASC, test_name ASC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_cache_stats(self, run_id: str) -> dict[str, int]:
        row = self.connection.execute(
            """
            SELECT
                COUNT(*) AS validation_rows,
                SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) AS cache_hits
            FROM metrics
            WHERE run_id = ? AND split = 'validation'
            """,
            (run_id,),
        ).fetchone()
        return {
            "validation_rows": int(row["validation_rows"] or 0),
            "cache_hits": int(row["cache_hits"] or 0),
        }

    def get_validation_metric_rows(self, run_id: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM metrics WHERE run_id = ? AND split = 'validation' ORDER BY fitness DESC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def dataframe_to_json(frame: pd.DataFrame) -> str:
        return frame.to_json(orient="split", date_format="iso")

    @staticmethod
    def dataframe_from_json(payload: str) -> pd.DataFrame:
        return pd.read_json(StringIO(payload), orient="split")

    @staticmethod
    def series_to_json(series: pd.Series) -> str:
        return series.to_json(orient="split", date_format="iso")

    @staticmethod
    def series_from_json(payload: str) -> pd.Series:
        return pd.read_json(StringIO(payload), orient="split", typ="series")
