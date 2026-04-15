from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from storage.models import ManualImportRecord, SubmissionBatchRecord, SubmissionRecord

_UNSET = object()


class SubmissionStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def upsert_batch(self, record: SubmissionBatchRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO submission_batches
            (batch_id, run_id, round_index, backend, status, candidate_count, sim_config_snapshot,
             export_path, notes_json, service_status_reason, last_polled_at, quarantined_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(batch_id) DO UPDATE SET
                status = excluded.status,
                candidate_count = excluded.candidate_count,
                sim_config_snapshot = excluded.sim_config_snapshot,
                export_path = excluded.export_path,
                notes_json = excluded.notes_json,
                service_status_reason = excluded.service_status_reason,
                last_polled_at = excluded.last_polled_at,
                quarantined_at = excluded.quarantined_at,
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
                record.service_status_reason,
                record.last_polled_at,
                record.quarantined_at,
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
             raw_submission_json, error_message, retry_count, last_polled_at, next_poll_after,
             timeout_deadline_at,
             stuck_since, service_failure_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    batch_id = excluded.batch_id,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    completed_at = excluded.completed_at,
                    export_path = excluded.export_path,
                    raw_submission_json = excluded.raw_submission_json,
                    error_message = excluded.error_message,
                    retry_count = excluded.retry_count,
                    last_polled_at = excluded.last_polled_at,
                    next_poll_after = excluded.next_poll_after,
                    timeout_deadline_at = COALESCE(excluded.timeout_deadline_at, submissions.timeout_deadline_at),
                    stuck_since = excluded.stuck_since,
                    service_failure_reason = excluded.service_failure_reason
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
                    record.retry_count,
                    record.last_polled_at,
                    record.next_poll_after,
                    record.timeout_deadline_at,
                    record.stuck_since,
                    record.service_failure_reason,
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

    def update_submission_runtime(
        self,
        job_id: str,
        *,
        updated_at: str,
        status: str | object = _UNSET,
        completed_at: str | None | object = _UNSET,
        error_message: str | None | object = _UNSET,
        retry_count: int | object = _UNSET,
        last_polled_at: str | None | object = _UNSET,
        next_poll_after: str | None | object = _UNSET,
        timeout_deadline_at: str | None | object = _UNSET,
        stuck_since: str | None | object = _UNSET,
        service_failure_reason: str | None | object = _UNSET,
    ) -> None:
        updates: dict[str, object] = {"updated_at": updated_at}
        if status is not _UNSET:
            updates["status"] = status
        if completed_at is not _UNSET:
            updates["completed_at"] = completed_at
        if error_message is not _UNSET:
            updates["error_message"] = error_message
        if retry_count is not _UNSET:
            updates["retry_count"] = retry_count
        if last_polled_at is not _UNSET:
            updates["last_polled_at"] = last_polled_at
        if next_poll_after is not _UNSET:
            updates["next_poll_after"] = next_poll_after
        if timeout_deadline_at is not _UNSET:
            updates["timeout_deadline_at"] = timeout_deadline_at
        if stuck_since is not _UNSET:
            updates["stuck_since"] = stuck_since
        if service_failure_reason is not _UNSET:
            updates["service_failure_reason"] = service_failure_reason
        assignments = ", ".join(f"{name} = ?" for name in updates)
        params = list(updates.values()) + [job_id]
        self.connection.execute(
            f"UPDATE submissions SET {assignments} WHERE job_id = ?",
            tuple(params),
        )
        self.connection.commit()

    def update_batch_status(
        self,
        batch_id: str,
        *,
        status: str,
        updated_at: str,
        candidate_count: int | object = _UNSET,
        sim_config_snapshot: str | object = _UNSET,
        export_path: str | None | object = _UNSET,
        notes_json: str | None | object = _UNSET,
        service_status_reason: str | None | object = _UNSET,
        last_polled_at: str | None | object = _UNSET,
        quarantined_at: str | None | object = _UNSET,
    ) -> None:
        updates: dict[str, object] = {"status": status, "updated_at": updated_at}
        if candidate_count is not _UNSET:
            updates["candidate_count"] = candidate_count
        if sim_config_snapshot is not _UNSET:
            updates["sim_config_snapshot"] = sim_config_snapshot
        if export_path is not _UNSET:
            updates["export_path"] = export_path
        if notes_json is not _UNSET:
            updates["notes_json"] = notes_json
        if service_status_reason is not _UNSET:
            updates["service_status_reason"] = service_status_reason
        if last_polled_at is not _UNSET:
            updates["last_polled_at"] = last_polled_at
        if quarantined_at is not _UNSET:
            updates["quarantined_at"] = quarantined_at
        assignments = ", ".join(f"{name} = ?" for name in updates)
        params = list(updates.values()) + [batch_id]
        self.connection.execute(
            f"UPDATE submission_batches SET {assignments} WHERE batch_id = ?",
            tuple(params),
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

    def list_pending_submissions(self, run_id: str) -> list[SubmissionRecord]:
        return self.list_submissions(run_id=run_id, statuses=("submitted", "running"))

    def backfill_timeout_deadlines(self, *, timeout_seconds: float) -> int:
        rows = self.connection.execute(
            """
            SELECT job_id, submitted_at
            FROM submissions
            WHERE timeout_deadline_at IS NULL
            """
        ).fetchall()
        if not rows:
            return 0
        for row in rows:
            submitted_at = str(row["submitted_at"] or "")
            timeout_deadline_at = _shift_iso(submitted_at, timeout_seconds)
            self.connection.execute(
                "UPDATE submissions SET timeout_deadline_at = ? WHERE job_id = ?",
                (timeout_deadline_at, row["job_id"]),
            )
        self.connection.commit()
        return len(rows)

    def extend_pending_timeout_deadlines(
        self,
        *,
        run_id: str,
        seconds: float,
        default_timeout_seconds: float,
        updated_at: str | None = None,
    ) -> int:
        if seconds <= 0:
            return 0
        now = updated_at or _utcnow()
        rows = self.list_pending_submissions(run_id)
        if not rows:
            return 0
        for row in rows:
            baseline = row.timeout_deadline_at or _shift_iso(row.submitted_at, default_timeout_seconds)
            extended_deadline = _shift_iso(baseline, seconds)
            self.connection.execute(
                """
                UPDATE submissions
                SET timeout_deadline_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (extended_deadline, now, row.job_id),
            )
        self.connection.commit()
        return len(rows)

    def list_batches_by_status(
        self,
        *,
        run_id: str,
        statuses: Iterable[str],
    ) -> list[SubmissionBatchRecord]:
        status_list = list(statuses)
        placeholders = ", ".join("?" for _ in status_list)
        rows = self.connection.execute(
            f"""
            SELECT *
            FROM submission_batches
            WHERE run_id = ? AND status IN ({placeholders})
            ORDER BY created_at ASC, batch_id ASC
            """,
            (run_id, *status_list),
        ).fetchall()
        return [SubmissionBatchRecord(**dict(row)) for row in rows]

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

    def count_recent_all_failed_batches(self, run_id: str, lookback: int = 5) -> int:
        """Count how many of the most recent completed batches had ALL jobs failed (zero successes).

        This is used to detect auth session expiry before the system formally detects it.
        """
        rows = self.connection.execute(
            """
            SELECT sb.batch_id,
                   COUNT(*) AS total_jobs,
                   SUM(CASE WHEN s.status IN ('failed', 'rejected', 'timeout') THEN 1 ELSE 0 END) AS failed_jobs
            FROM submission_batches sb
            JOIN submissions s ON s.batch_id = sb.batch_id
            WHERE sb.run_id = ? AND sb.status = 'completed'
            GROUP BY sb.batch_id
            ORDER BY sb.created_at DESC
            LIMIT ?
            """,
            (run_id, lookback),
        ).fetchall()
        consecutive_all_failed = 0
        for row in rows:
            total = row["total_jobs"]
            failed = row["failed_jobs"]
            if total > 0 and failed == total:
                consecutive_all_failed += 1
            else:
                break
        return consecutive_all_failed


def _utcnow() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _shift_iso(timestamp: str, seconds: float) -> str:
    from datetime import datetime, timedelta

    return (datetime.fromisoformat(timestamp) + timedelta(seconds=float(seconds))).isoformat()
