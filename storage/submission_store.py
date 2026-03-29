from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from storage.models import ManualImportRecord, SubmissionBatchRecord, SubmissionRecord


class SubmissionStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def upsert_batch(self, record: SubmissionBatchRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO submission_batches
            (batch_id, run_id, round_index, backend, status, candidate_count, sim_config_snapshot,
             export_path, notes_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(batch_id) DO UPDATE SET
                status = excluded.status,
                candidate_count = excluded.candidate_count,
                sim_config_snapshot = excluded.sim_config_snapshot,
                export_path = excluded.export_path,
                notes_json = excluded.notes_json,
                updated_at = excluded.updated_at
            """,
            (
                record.batch_id,
                record.run_id,
                record.round_index,
                record.backend,
                record.status,
                record.candidate_count,
                record.sim_config_snapshot,
                record.export_path,
                record.notes_json,
                record.created_at,
                record.updated_at,
            ),
        )
        self.connection.commit()

    def upsert_submissions(self, records: Iterable[SubmissionRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO submissions
                (job_id, batch_id, run_id, round_index, candidate_id, expression, backend, status,
                 sim_config_snapshot, submitted_at, updated_at, completed_at, export_path,
                 raw_submission_json, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    batch_id = excluded.batch_id,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    completed_at = excluded.completed_at,
                    export_path = excluded.export_path,
                    raw_submission_json = excluded.raw_submission_json,
                    error_message = excluded.error_message
                """,
                (
                    record.job_id,
                    record.batch_id,
                    record.run_id,
                    record.round_index,
                    record.candidate_id,
                    record.expression,
                    record.backend,
                    record.status,
                    record.sim_config_snapshot,
                    record.submitted_at,
                    record.updated_at,
                    record.completed_at,
                    record.export_path,
                    record.raw_submission_json,
                    record.error_message,
                ),
            )
        self.connection.commit()

    def update_submission_status(
        self,
        job_id: str,
        *,
        status: str,
        updated_at: str,
        completed_at: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE submissions
            SET status = ?,
                updated_at = ?,
                completed_at = COALESCE(?, completed_at),
                error_message = COALESCE(?, error_message)
            WHERE job_id = ?
            """,
            (status, updated_at, completed_at, error_message, job_id),
        )
        self.connection.commit()

    def update_batch_status(
        self,
        batch_id: str,
        *,
        status: str,
        updated_at: str,
        export_path: str | None = None,
        notes_json: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE submission_batches
            SET status = ?,
                updated_at = ?,
                export_path = COALESCE(?, export_path),
                notes_json = COALESCE(?, notes_json)
            WHERE batch_id = ?
            """,
            (status, updated_at, export_path, notes_json, batch_id),
        )
        self.connection.commit()

    def get_submission(self, job_id: str) -> SubmissionRecord | None:
        row = self.connection.execute(
            "SELECT * FROM submissions WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        return SubmissionRecord(**dict(row)) if row else None

    def list_submissions(
        self,
        *,
        run_id: str,
        round_index: int | None = None,
        batch_id: str | None = None,
        statuses: Iterable[str] | None = None,
    ) -> list[SubmissionRecord]:
        clauses = ["run_id = ?"]
        params: list[object] = [run_id]
        if round_index is not None:
            clauses.append("round_index = ?")
            params.append(round_index)
        if batch_id is not None:
            clauses.append("batch_id = ?")
            params.append(batch_id)
        if statuses:
            status_list = list(statuses)
            placeholders = ", ".join("?" for _ in status_list)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status_list)
        rows = self.connection.execute(
            f"""
            SELECT *
            FROM submissions
            WHERE {" AND ".join(clauses)}
            ORDER BY submitted_at ASC, job_id ASC
            """,
            tuple(params),
        ).fetchall()
        return [SubmissionRecord(**dict(row)) for row in rows]

    def get_batch(self, batch_id: str) -> SubmissionBatchRecord | None:
        row = self.connection.execute(
            "SELECT * FROM submission_batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        return SubmissionBatchRecord(**dict(row)) if row else None

    def list_batches(self, run_id: str) -> list[SubmissionBatchRecord]:
        rows = self.connection.execute(
            "SELECT * FROM submission_batches WHERE run_id = ? ORDER BY created_at ASC, batch_id ASC",
            (run_id,),
        ).fetchall()
        return [SubmissionBatchRecord(**dict(row)) for row in rows]

    def save_manual_import(self, record: ManualImportRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO manual_imports (import_id, run_id, batch_id, source_path, imported_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(import_id) DO UPDATE SET
                imported_count = excluded.imported_count,
                created_at = excluded.created_at
            """,
            (
                record.import_id,
                record.run_id,
                record.batch_id,
                record.source_path,
                record.imported_count,
                record.created_at,
            ),
        )
        self.connection.commit()

    def get_latest_batch(self, run_id: str) -> SubmissionBatchRecord | None:
        row = self.connection.execute(
            """
            SELECT *
            FROM submission_batches
            WHERE run_id = ?
            ORDER BY created_at DESC, batch_id DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        return SubmissionBatchRecord(**dict(row)) if row else None
