from __future__ import annotations

import sqlite3
from collections.abc import Mapping

from storage.models import ServiceRuntimeRecord


class ServiceRuntimeStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def get_state(self, service_name: str) -> ServiceRuntimeRecord | None:
        row = self.connection.execute(
            "SELECT * FROM service_runtime WHERE service_name = ?",
            (service_name,),
        ).fetchone()
        return ServiceRuntimeRecord(**dict(row)) if row else None

    def upsert_state(self, record: ServiceRuntimeRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO service_runtime
            (service_name, service_run_id, owner_token, pid, hostname, status, tick_id, active_batch_id, pending_job_count,
             consecutive_failures, cooldown_until, last_heartbeat_at, last_success_at, last_error, persona_url,
             persona_wait_started_at, persona_last_notification_at, persona_confirmation_nonce,
             persona_confirmation_last_prompt_at, persona_confirmation_granted_at,
             persona_confirmation_last_update_id, counters_json, lock_expires_at, started_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(service_name) DO UPDATE SET
                service_run_id = excluded.service_run_id,
                owner_token = excluded.owner_token,
                pid = excluded.pid,
                hostname = excluded.hostname,
                status = excluded.status,
                tick_id = excluded.tick_id,
                active_batch_id = excluded.active_batch_id,
                pending_job_count = excluded.pending_job_count,
                consecutive_failures = excluded.consecutive_failures,
                cooldown_until = excluded.cooldown_until,
                last_heartbeat_at = excluded.last_heartbeat_at,
                last_success_at = excluded.last_success_at,
                last_error = excluded.last_error,
                persona_url = excluded.persona_url,
                persona_wait_started_at = excluded.persona_wait_started_at,
                persona_last_notification_at = excluded.persona_last_notification_at,
                persona_confirmation_nonce = excluded.persona_confirmation_nonce,
                persona_confirmation_last_prompt_at = excluded.persona_confirmation_last_prompt_at,
                persona_confirmation_granted_at = excluded.persona_confirmation_granted_at,
                persona_confirmation_last_update_id = excluded.persona_confirmation_last_update_id,
                counters_json = excluded.counters_json,
                lock_expires_at = excluded.lock_expires_at,
                started_at = excluded.started_at,
                updated_at = excluded.updated_at
            """,
            (
                record.service_name,
                record.service_run_id,
                record.owner_token,
                record.pid,
                record.hostname,
                record.status,
                record.tick_id,
                record.active_batch_id,
                record.pending_job_count,
                record.consecutive_failures,
                record.cooldown_until,
                record.last_heartbeat_at,
                record.last_success_at,
                record.last_error,
                record.persona_url,
                record.persona_wait_started_at,
                record.persona_last_notification_at,
                record.persona_confirmation_nonce,
                record.persona_confirmation_last_prompt_at,
                record.persona_confirmation_granted_at,
                record.persona_confirmation_last_update_id,
                record.counters_json,
                record.lock_expires_at,
                record.started_at,
                record.updated_at,
            ),
        )
        self.connection.commit()

    def update_state(self, service_name: str, **fields: object) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{name} = ?" for name in fields)
        params = list(fields.values()) + [service_name]
        self.connection.execute(
            f"UPDATE service_runtime SET {assignments} WHERE service_name = ?",
            tuple(params),
        )
        self.connection.commit()

    def try_acquire_lease(
        self,
        *,
        service_name: str,
        owner_token: str,
        service_run_id: str,
        pid: int,
        hostname: str,
        status: str,
        now: str,
        lock_expires_at: str,
    ) -> bool:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT owner_token, lock_expires_at, started_at FROM service_runtime WHERE service_name = ?",
                (service_name,),
            ).fetchone()
            if row is None:
                self.connection.execute(
                    """
                    INSERT INTO service_runtime
                    (service_name, service_run_id, owner_token, pid, hostname, status, tick_id, active_batch_id,
                     pending_job_count, consecutive_failures, cooldown_until, last_heartbeat_at, last_success_at, last_error,
                     persona_url, persona_wait_started_at, persona_last_notification_at, persona_confirmation_nonce,
                     persona_confirmation_last_prompt_at, persona_confirmation_granted_at,
                     persona_confirmation_last_update_id, counters_json, lock_expires_at, started_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, NULL, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, '{}', ?, ?, ?)
                    """,
                    (service_name, service_run_id, owner_token, pid, hostname, status, now, lock_expires_at, now, now),
                )
                self.connection.commit()
                return True

            current_owner = str(row["owner_token"] or "")
            current_expiry = row["lock_expires_at"]
            if current_owner and current_owner != owner_token and current_expiry and current_expiry > now:
                self.connection.rollback()
                return False

            started_at = row["started_at"] or now
            self.connection.execute(
                """
                UPDATE service_runtime
                SET service_run_id = ?,
                    owner_token = ?,
                    pid = ?,
                    hostname = ?,
                    status = ?,
                    lock_expires_at = ?,
                    started_at = COALESCE(NULLIF(started_at, ''), ?),
                    updated_at = ?
                WHERE service_name = ?
                """,
                (service_run_id, owner_token, pid, hostname, status, lock_expires_at, started_at, now, service_name),
            )
            self.connection.commit()
            return True
        except Exception:
            self.connection.rollback()
            raise

    def renew_lease(self, *, service_name: str, owner_token: str, lock_expires_at: str, updated_at: str) -> bool:
        cursor = self.connection.execute(
            """
            UPDATE service_runtime
            SET lock_expires_at = ?, updated_at = ?
            WHERE service_name = ? AND owner_token = ?
            """,
            (lock_expires_at, updated_at, service_name, owner_token),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def release_lease(
        self,
        *,
        service_name: str,
        owner_token: str,
        status: str,
        updated_at: str,
    ) -> None:
        self.connection.execute(
            """
            UPDATE service_runtime
            SET owner_token = '',
                lock_expires_at = NULL,
                status = ?,
                updated_at = ?
            WHERE service_name = ? AND owner_token = ?
            """,
            (status, updated_at, service_name, owner_token),
        )
        self.connection.commit()

    def state_from_mapping(self, payload: Mapping[str, object]) -> ServiceRuntimeRecord:
        return ServiceRuntimeRecord(**dict(payload))
