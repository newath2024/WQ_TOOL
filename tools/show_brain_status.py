from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show the latest BRAIN service, submission, and result state from the local SQLite database."
    )
    parser.add_argument(
        "--config",
        default="config/dev.yaml",
        help="Config file used to resolve the default SQLite path when --db is not provided.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the SQLite database. Overrides the path resolved from --config.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id filter. Defaults to the latest run in the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of rows to print per section.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON document instead of human-readable sections.",
    )
    return parser


def resolve_database_path(config_path: str, override_path: str | None) -> Path:
    if override_path:
        return Path(override_path).expanduser().resolve()
    config = load_config(config_path)
    return Path(config.storage.path).expanduser().resolve()


def resolve_run_id(connection: sqlite3.Connection, requested_run_id: str | None) -> str | None:
    if requested_run_id:
        return requested_run_id
    row = connection.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    return str(row["run_id"]) if row and row["run_id"] else None


def fetch_status_payload(connection: sqlite3.Connection, run_id: str | None, limit: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "service_runtime": [],
        "service_dispatch_queue": [],
        "submission_batches": [],
        "submissions": [],
        "brain_results": [],
    }

    if run_id:
        payload["runs"] = rows_to_dicts(
            connection.execute(
                "SELECT * FROM runs WHERE run_id = ? ORDER BY started_at DESC LIMIT ?",
                (run_id, max(limit, 1)),
            ).fetchall()
        )
        payload["service_runtime"] = rows_to_dicts(
            connection.execute(
                """
                SELECT *
                FROM service_runtime
                WHERE service_run_id = ?
                ORDER BY updated_at DESC, service_name ASC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        )
        payload["submission_batches"] = rows_to_dicts(
            connection.execute(
                """
                SELECT *
                FROM submission_batches
                WHERE run_id = ?
                ORDER BY created_at DESC, batch_id DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        )
        payload["service_dispatch_queue"] = rows_to_dicts(
            connection.execute(
                """
                SELECT *
                FROM service_dispatch_queue
                WHERE run_id = ?
                ORDER BY updated_at DESC, queue_position DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        )
        payload["submissions"] = rows_to_dicts(
            connection.execute(
                """
                SELECT *
                FROM submissions
                WHERE run_id = ?
                ORDER BY submitted_at DESC, job_id DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        )
        payload["brain_results"] = rows_to_dicts(
            connection.execute(
                """
                SELECT *
                FROM brain_results
                WHERE run_id = ?
                ORDER BY simulated_at DESC, job_id DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        )
        return payload

    payload["runs"] = rows_to_dicts(
        connection.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    )
    payload["service_runtime"] = rows_to_dicts(
        connection.execute(
            "SELECT * FROM service_runtime ORDER BY updated_at DESC, service_name ASC LIMIT ?",
            (limit,),
        ).fetchall()
    )
    payload["submission_batches"] = rows_to_dicts(
        connection.execute(
            "SELECT * FROM submission_batches ORDER BY created_at DESC, batch_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    )
    payload["service_dispatch_queue"] = rows_to_dicts(
        connection.execute(
            "SELECT * FROM service_dispatch_queue ORDER BY updated_at DESC, queue_position DESC LIMIT ?",
            (limit,),
        ).fetchall()
    )
    payload["submissions"] = rows_to_dicts(
        connection.execute(
            "SELECT * FROM submissions ORDER BY submitted_at DESC, job_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    )
    payload["brain_results"] = rows_to_dicts(
        connection.execute(
            "SELECT * FROM brain_results ORDER BY simulated_at DESC, job_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    )
    return payload


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def print_human(payload: dict[str, Any]) -> None:
    run_id = payload.get("run_id")
    print(f"run_id: {run_id or '(latest not found)'}")
    print_section("runs", payload.get("runs", []))
    print_section("service_runtime", payload.get("service_runtime", []))
    print_section("service_dispatch_queue", payload.get("service_dispatch_queue", []))
    print_section("submission_batches", payload.get("submission_batches", []))
    print_section("submissions", payload.get("submissions", []))
    print_section("brain_results", payload.get("brain_results", []))


def print_section(name: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n== {name} ==")
    if not rows:
        print("(empty)")
        return
    for row in rows:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))


def main() -> int:
    args = build_parser().parse_args()
    database_path = resolve_database_path(args.config, args.db)
    if not database_path.exists():
        raise SystemExit(f"Database not found: {database_path}")

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        run_id = resolve_run_id(connection, args.run_id)
        payload = fetch_status_payload(connection, run_id, max(args.limit, 1))
    finally:
        connection.close()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
