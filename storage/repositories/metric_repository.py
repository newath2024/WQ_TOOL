from __future__ import annotations

import sqlite3
from types import SimpleNamespace


from core.quality_score import MultiObjectiveQualityScorer
from storage.models import (
    MetricRecord,
    StageMetricRecord,
    SubmissionTestRecord,
)


class MetricRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def get_latest_brain_quality_score(self, run_id: str, alpha_id: str) -> float | None:
        row = self.connection.execute(
            """
            SELECT *
            FROM brain_results
            WHERE run_id = ?
              AND candidate_id = ?
            ORDER BY simulated_at DESC, created_at DESC, job_id DESC
            LIMIT 1
            """,
            (run_id, alpha_id),
        ).fetchone()
        if row is None:
            return None
        return MultiObjectiveQualityScorer.score_record(SimpleNamespace(**dict(row)))

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
