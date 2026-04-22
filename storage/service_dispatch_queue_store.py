from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from storage.models import ServiceDispatchQueueRecord


class ServiceDispatchQueueStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def upsert_items(self, records: Iterable[ServiceDispatchQueueRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO service_dispatch_queue
                (queue_item_id, service_name, run_id, candidate_id, source_round_index, queue_position,
                 status, batch_id, job_id, failure_reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(queue_item_id) DO UPDATE SET
                    service_name = excluded.service_name,
                    run_id = excluded.run_id,
                    candidate_id = excluded.candidate_id,
                    source_round_index = excluded.source_round_index,
                    queue_position = excluded.queue_position,
                    status = excluded.status,
                    batch_id = excluded.batch_id,
                    job_id = excluded.job_id,
                    failure_reason = excluded.failure_reason,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    record.queue_item_id,
                    record.service_name,
                    record.run_id,
                    record.candidate_id,
                    record.source_round_index,
                    record.queue_position,
                    record.status,
                    record.batch_id,
                    record.job_id,
                    record.failure_reason,
                    record.created_at,
                    record.updated_at,
                ),
            )
        self.connection.commit()

    def get_item(self, queue_item_id: str) -> ServiceDispatchQueueRecord | None:
        row = self.connection.execute(
            "SELECT * FROM service_dispatch_queue WHERE queue_item_id = ?",
            (queue_item_id,),
        ).fetchone()
        return ServiceDispatchQueueRecord(**dict(row)) if row else None

    def list_items(
        self,
        *,
        service_name: str | None = None,
        run_id: str | None = None,
        statuses: Iterable[str] | None = None,
        source_round_index: int | None = None,
    ) -> list[ServiceDispatchQueueRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if service_name is not None:
            clauses.append("service_name = ?")
            params.append(service_name)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if source_round_index is not None:
            clauses.append("source_round_index = ?")
            params.append(source_round_index)
        if statuses is not None:
            status_list = list(statuses)
            if not status_list:
                return []
            placeholders = ", ".join("?" for _ in status_list)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status_list)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"""
            SELECT *
            FROM service_dispatch_queue
            {where_clause}
            ORDER BY queue_position ASC, created_at ASC, queue_item_id ASC
            """,
            tuple(params),
        ).fetchall()
        return [ServiceDispatchQueueRecord(**dict(row)) for row in rows]

    def count_items(
        self,
        *,
        service_name: str | None = None,
        run_id: str | None = None,
        statuses: Iterable[str] | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[object] = []
        if service_name is not None:
            clauses.append("service_name = ?")
            params.append(service_name)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if statuses is not None:
            status_list = list(statuses)
            if not status_list:
                return 0
            placeholders = ", ".join("?" for _ in status_list)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status_list)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self.connection.execute(
            f"SELECT COUNT(*) AS total FROM service_dispatch_queue {where_clause}",
            tuple(params),
        ).fetchone()
        return int(row["total"] or 0) if row is not None else 0

    def update_item(self, queue_item_id: str, **fields: object) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{name} = ?" for name in fields)
        params = list(fields.values()) + [queue_item_id]
        self.connection.execute(
            f"UPDATE service_dispatch_queue SET {assignments} WHERE queue_item_id = ?",
            tuple(params),
        )
        self.connection.commit()

    def next_queue_position(self, *, service_name: str, run_id: str) -> int:
        row = self.connection.execute(
            """
            SELECT COALESCE(MAX(queue_position), 0) AS max_position
            FROM service_dispatch_queue
            WHERE service_name = ? AND run_id = ?
            """,
            (service_name, run_id),
        ).fetchone()
        return int(row["max_position"] or 0) + 1 if row is not None else 1
