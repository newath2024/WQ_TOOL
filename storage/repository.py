from __future__ import annotations

import json
from datetime import UTC, datetime
from io import StringIO

import pandas as pd

from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemoryService
from storage.alpha_history import AlphaHistoryStore
from storage.brain_result_store import BrainResultStore
from storage.models import (
    AlphaRecord,
    FieldCatalogRecord,
    MetricRecord,
    RunFieldScoreRecord,
    RunRecord,
    SelectionRecord,
    SimulationCacheRecord,
    SubmissionTestRecord,
)
from storage.sqlite import connect_sqlite
from storage.submission_store import SubmissionStore


class SQLiteRepository:
    def __init__(self, path: str) -> None:
        self.connection = connect_sqlite(path)
        self.alpha_history = AlphaHistoryStore(self.connection, PatternMemoryService())
        self.submissions = SubmissionStore(self.connection)
        self.brain_results = BrainResultStore(self.connection)

    def close(self) -> None:
        self.connection.close()

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
        entry_command: str = "",
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO runs
            (run_id, seed, config_path, config_snapshot, status, started_at, profile_name, selected_timeframe, entry_command)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                seed = excluded.seed,
                config_path = excluded.config_path,
                config_snapshot = excluded.config_snapshot,
                status = excluded.status,
                profile_name = excluded.profile_name,
                selected_timeframe = excluded.selected_timeframe,
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
    ) -> None:
        self.connection.execute(
            """
            UPDATE runs
            SET dataset_summary = ?,
                dataset_fingerprint = COALESCE(?, dataset_fingerprint),
                selected_timeframe = COALESCE(?, selected_timeframe),
                regime_key = COALESCE(?, regime_key)
            WHERE run_id = ?
            """,
            (
                json.dumps(summary, sort_keys=True),
                dataset_fingerprint,
                selected_timeframe,
                regime_key,
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
                 operators_used_json, depth, generation_metadata, complexity, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def list_alpha_records(self, run_id: str) -> list[AlphaRecord]:
        rows = self.connection.execute(
            "SELECT * FROM alphas WHERE run_id = ? ORDER BY created_at ASC, alpha_id ASC",
            (run_id,),
        ).fetchall()
        return [AlphaRecord(**dict(row)) for row in rows]

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
