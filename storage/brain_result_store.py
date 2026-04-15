from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from core.brain_rejections import extract_invalid_field_from_rejection
from storage.models import BrainResultRecord, ClosedLoopRoundRecord, ClosedLoopRunRecord


class BrainResultStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def save_results(self, records: Iterable[BrainResultRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO brain_results
                (job_id, run_id, round_index, batch_id, candidate_id, expression, status, region, universe, delay,
                 neutralization, decay, sharpe, fitness, turnover, drawdown, returns, margin,
                 submission_eligible, rejection_reason, raw_result_json, metric_source, simulated_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    region = excluded.region,
                    universe = excluded.universe,
                    delay = excluded.delay,
                    neutralization = excluded.neutralization,
                    decay = excluded.decay,
                    sharpe = excluded.sharpe,
                    fitness = excluded.fitness,
                    turnover = excluded.turnover,
                    drawdown = excluded.drawdown,
                    returns = excluded.returns,
                    margin = excluded.margin,
                    submission_eligible = excluded.submission_eligible,
                    rejection_reason = excluded.rejection_reason,
                    raw_result_json = excluded.raw_result_json,
                    metric_source = excluded.metric_source,
                    simulated_at = excluded.simulated_at,
                    created_at = excluded.created_at
                """,
                (
                    record.job_id,
                    record.run_id,
                    record.round_index,
                    record.batch_id,
                    record.candidate_id,
                    record.expression,
                    record.status,
                    record.region,
                    record.universe,
                    record.delay,
                    record.neutralization,
                    record.decay,
                    record.sharpe,
                    record.fitness,
                    record.turnover,
                    record.drawdown,
                    record.returns,
                    record.margin,
                    None if record.submission_eligible is None else int(record.submission_eligible),
                    record.rejection_reason,
                    record.raw_result_json,
                    record.metric_source,
                    record.simulated_at,
                    record.created_at,
                ),
            )
        self.connection.commit()

    def get_result(self, job_id: str) -> BrainResultRecord | None:
        row = self.connection.execute(
            "SELECT * FROM brain_results WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        return self._row_to_result(row) if row else None

    def list_results(
        self,
        *,
        run_id: str,
        round_index: int | None = None,
        statuses: Iterable[str] | None = None,
    ) -> list[BrainResultRecord]:
        clauses = ["run_id = ?"]
        params: list[object] = [run_id]
        if round_index is not None:
            clauses.append("round_index = ?")
            params.append(round_index)
        if statuses:
            status_list = list(statuses)
            placeholders = ", ".join("?" for _ in status_list)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status_list)
        rows = self.connection.execute(
            f"""
            SELECT *
            FROM brain_results
            WHERE {" AND ".join(clauses)}
            ORDER BY simulated_at ASC, job_id ASC
            """,
            tuple(params),
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def list_latest_results_by_candidate(self, run_id: str) -> list[BrainResultRecord]:
        rows = self.connection.execute(
            """
            SELECT r.*
            FROM brain_results r
            JOIN (
                SELECT candidate_id, MAX(simulated_at) AS max_simulated_at
                FROM brain_results
                WHERE run_id = ?
                GROUP BY candidate_id
            ) latest
              ON latest.candidate_id = r.candidate_id AND latest.max_simulated_at = r.simulated_at
            WHERE r.run_id = ?
            ORDER BY r.simulated_at DESC, r.candidate_id ASC
            """,
            (run_id, run_id),
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def list_invalid_generation_fields(
        self,
        *,
        region: str,
        universe: str,
        delay: int,
    ) -> set[str]:
        clauses = [
            "rejection_reason IS NOT NULL",
            "rejection_reason <> ''",
            "delay = ?",
        ]
        params: list[object] = [int(delay)]
        normalized_region = str(region or "").strip().upper()
        normalized_universe = str(universe or "").strip().upper()
        if normalized_region:
            clauses.append("UPPER(region) = ?")
            params.append(normalized_region)
        if normalized_universe:
            clauses.append("UPPER(universe) = ?")
            params.append(normalized_universe)
        rows = self.connection.execute(
            f"""
            SELECT rejection_reason
            FROM brain_results
            WHERE {" AND ".join(clauses)}
            ORDER BY simulated_at DESC, created_at DESC, job_id DESC
            """,
            tuple(params),
        ).fetchall()
        blocked_fields: set[str] = set()
        for row in rows:
            field_name = extract_invalid_field_from_rejection(row["rejection_reason"])
            if field_name:
                blocked_fields.add(field_name)
        return blocked_fields

    def upsert_closed_loop_run(self, record: ClosedLoopRunRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO closed_loop_runs
            (run_id, backend, status, requested_rounds, completed_rounds, config_snapshot, started_at, finished_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                backend = excluded.backend,
                status = excluded.status,
                requested_rounds = excluded.requested_rounds,
                completed_rounds = excluded.completed_rounds,
                config_snapshot = excluded.config_snapshot,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at
            """,
            (
                record.run_id,
                record.backend,
                record.status,
                record.requested_rounds,
                record.completed_rounds,
                record.config_snapshot,
                record.started_at,
                record.finished_at,
            ),
        )
        self.connection.commit()

    def get_closed_loop_run(self, run_id: str) -> ClosedLoopRunRecord | None:
        row = self.connection.execute(
            "SELECT * FROM closed_loop_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return ClosedLoopRunRecord(**dict(row)) if row else None

    def get_closed_loop_round(self, run_id: str, round_index: int) -> ClosedLoopRoundRecord | None:
        row = self.connection.execute(
            "SELECT * FROM closed_loop_rounds WHERE run_id = ? AND round_index = ?",
            (run_id, round_index),
        ).fetchone()
        return ClosedLoopRoundRecord(**dict(row)) if row else None

    def upsert_closed_loop_round(self, record: ClosedLoopRoundRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO closed_loop_rounds
            (run_id, round_index, status, generated_count, validated_count, submitted_count, completed_count,
             selected_for_mutation_count, mutated_children_count, summary_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, round_index) DO UPDATE SET
                status = excluded.status,
                generated_count = excluded.generated_count,
                validated_count = excluded.validated_count,
                submitted_count = excluded.submitted_count,
                completed_count = excluded.completed_count,
                selected_for_mutation_count = excluded.selected_for_mutation_count,
                mutated_children_count = excluded.mutated_children_count,
                summary_json = excluded.summary_json,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            (
                record.run_id,
                record.round_index,
                record.status,
                record.generated_count,
                record.validated_count,
                record.submitted_count,
                record.completed_count,
                record.selected_for_mutation_count,
                record.mutated_children_count,
                record.summary_json,
                record.created_at,
                record.updated_at,
            ),
        )
        self.connection.commit()

    def list_closed_loop_rounds(self, run_id: str) -> list[ClosedLoopRoundRecord]:
        rows = self.connection.execute(
            """
            SELECT *
            FROM closed_loop_rounds
            WHERE run_id = ?
            ORDER BY round_index ASC
            """,
            (run_id,),
        ).fetchall()
        return [ClosedLoopRoundRecord(**dict(row)) for row in rows]

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> BrainResultRecord:
        payload = dict(row)
        eligible = payload.get("submission_eligible")
        payload["submission_eligible"] = None if eligible is None else bool(eligible)
        return BrainResultRecord(**payload)
