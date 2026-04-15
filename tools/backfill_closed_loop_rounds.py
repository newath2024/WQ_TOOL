from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


TERMINAL_SUBMISSION_STATUSES = {"completed", "failed", "rejected", "timeout"}


@dataclass(frozen=True, slots=True)
class ProgressRoundSnapshot:
    prepared_timestamp: str | None = None
    candidate_count: int | None = None
    selected_count: int | None = None
    archived_count: int | None = None
    generation_stage_metrics: dict[str, object] | None = None
    batch_id: str | None = None
    submitted_count: int | None = None
    submitted_timestamp: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill closed_loop_rounds for service-mode batches from progress logs and SQLite state.",
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database.")
    parser.add_argument("--run-id", required=True, help="Run id to backfill.")
    parser.add_argument(
        "--progress-log-dir",
        default="progress_logs",
        help="Directory containing <run_id>*.jsonl progress logs.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Update existing closed_loop_rounds rows instead of only filling missing rows.",
    )
    return parser.parse_args()


def _load_progress_snapshots(progress_log_dir: Path, run_id: str) -> dict[int, ProgressRoundSnapshot]:
    snapshots: dict[int, dict[str, object]] = {}
    log_paths = sorted(
        progress_log_dir.glob(f"{run_id}*.jsonl"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    for path in log_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(payload.get("run_id") or "") != run_id:
                    continue
                round_index = payload.get("round_index")
                if not isinstance(round_index, int):
                    continue
                event = str(payload.get("event") or "")
                event_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
                entry = snapshots.setdefault(round_index, {})
                if event == "batch_prepared":
                    entry["prepared_timestamp"] = payload.get("timestamp")
                    entry["candidate_count"] = _optional_int(event_payload.get("candidate_count"))
                    entry["selected_count"] = _optional_int(event_payload.get("selected_count"))
                    entry["archived_count"] = _optional_int(event_payload.get("archived_count"))
                    generation_stage_metrics = event_payload.get("generation_stage_metrics")
                    if isinstance(generation_stage_metrics, dict):
                        entry["generation_stage_metrics"] = generation_stage_metrics
                elif event == "batch_submitted":
                    entry["batch_id"] = payload.get("batch_id")
                    entry["submitted_count"] = _optional_int(event_payload.get("submitted_count"))
                    entry["submitted_timestamp"] = payload.get("timestamp")

    return {
        round_index: ProgressRoundSnapshot(
            prepared_timestamp=str(entry.get("prepared_timestamp") or "") or None,
            candidate_count=_optional_int(entry.get("candidate_count")),
            selected_count=_optional_int(entry.get("selected_count")),
            archived_count=_optional_int(entry.get("archived_count")),
            generation_stage_metrics=entry.get("generation_stage_metrics")
            if isinstance(entry.get("generation_stage_metrics"), dict)
            else None,
            batch_id=str(entry.get("batch_id") or "") or None,
            submitted_count=_optional_int(entry.get("submitted_count")),
            submitted_timestamp=str(entry.get("submitted_timestamp") or "") or None,
        )
        for round_index, entry in snapshots.items()
    }


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _decode_json_dict(payload: str | None) -> dict[str, object]:
    if not payload:
        return {}
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _fetch_stage_metrics(connection: sqlite3.Connection, run_id: str) -> dict[tuple[int, str], dict[str, object]]:
    rows = connection.execute(
        """
        SELECT round_index, stage, metrics_json
        FROM round_stage_metrics
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchall()
    result: dict[tuple[int, str], dict[str, object]] = {}
    for row in rows:
        metrics_json = row["metrics_json"]
        result[(int(row["round_index"]), str(row["stage"]))] = _decode_json_dict(metrics_json)
    return result


def _existing_round_indices(connection: sqlite3.Connection, run_id: str) -> set[int]:
    rows = connection.execute(
        "SELECT round_index FROM closed_loop_rounds WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    return {int(row["round_index"]) for row in rows}


def _selected_parent_count(connection: sqlite3.Connection, run_id: str, candidate_ids: list[str]) -> int:
    normalized_candidate_ids = tuple(dict.fromkeys(candidate_id for candidate_id in candidate_ids if candidate_id))
    if not normalized_candidate_ids:
        return 0
    placeholders = ", ".join("?" for _ in normalized_candidate_ids)
    row = connection.execute(
        f"""
        SELECT COUNT(DISTINCT alpha_id) AS total
        FROM alpha_history
        WHERE run_id = ?
          AND metric_source = 'external_brain'
          AND selected = 1
          AND alpha_id IN ({placeholders})
        """,
        (run_id, *normalized_candidate_ids),
    ).fetchone()
    return int(row["total"] or 0) if row is not None else 0


def _backfill_rounds(
    *,
    connection: sqlite3.Connection,
    run_id: str,
    progress_snapshots: dict[int, ProgressRoundSnapshot],
    overwrite_existing: bool,
) -> tuple[int, int]:
    stage_metrics = _fetch_stage_metrics(connection, run_id)
    existing_rounds = _existing_round_indices(connection, run_id)
    batch_rows = connection.execute(
        """
        SELECT *
        FROM submission_batches
        WHERE run_id = ?
        ORDER BY round_index ASC, created_at ASC, batch_id ASC
        """,
        (run_id,),
    ).fetchall()
    latest_batch_by_round: dict[int, sqlite3.Row] = {}
    for batch in batch_rows:
        round_index = int(batch["round_index"])
        current = latest_batch_by_round.get(round_index)
        if current is None:
            latest_batch_by_round[round_index] = batch
            continue
        current_key = (str(current["created_at"] or ""), str(current["updated_at"] or ""), str(current["batch_id"] or ""))
        candidate_key = (str(batch["created_at"] or ""), str(batch["updated_at"] or ""), str(batch["batch_id"] or ""))
        if candidate_key >= current_key:
            latest_batch_by_round[round_index] = batch

    inserted = 0
    updated = 0
    now = datetime.now(UTC).isoformat()

    for round_index in sorted(latest_batch_by_round):
        batch = latest_batch_by_round[round_index]
        round_index = int(batch["round_index"])
        if round_index in existing_rounds and not overwrite_existing:
            continue

        submissions = connection.execute(
            """
            SELECT *
            FROM submissions
            WHERE run_id = ? AND batch_id = ?
            ORDER BY submitted_at ASC, job_id ASC
            """,
            (run_id, batch["batch_id"]),
        ).fetchall()
        candidate_ids = [str(row["candidate_id"]) for row in submissions if row["candidate_id"]]
        submission_status_counts = Counter(str(row["status"]) for row in submissions if row["status"])
        terminal_submission_count = sum(
            count for status, count in submission_status_counts.items() if status in TERMINAL_SUBMISSION_STATUSES
        )

        progress = progress_snapshots.get(round_index)
        generation_metrics = stage_metrics.get((round_index, "generation"), {})
        pre_sim_metrics = stage_metrics.get((round_index, "pre_sim"), {})
        generated_count = _optional_int(pre_sim_metrics.get("generated"))
        if generated_count is None:
            generated_count = _optional_int(generation_metrics.get("generated"))
        if generated_count is None and progress is not None:
            generated_count = progress.candidate_count
        if generated_count is None:
            generated_count = int(batch["candidate_count"] or 0)

        validated_count = _optional_int(pre_sim_metrics.get("kept_after_dedup"))
        if validated_count is None:
            validated_count = _optional_int(pre_sim_metrics.get("kept_after_hard_dedup"))
        if validated_count is None:
            validated_count = generated_count

        submitted_count = len(submissions)
        if submitted_count <= 0 and progress is not None and progress.submitted_count is not None:
            submitted_count = progress.submitted_count
        if submitted_count <= 0:
            submitted_count = int(batch["candidate_count"] or 0)

        selected_for_mutation_count = _selected_parent_count(connection, run_id, candidate_ids)

        existing = connection.execute(
            """
            SELECT created_at, mutated_children_count
            FROM closed_loop_rounds
            WHERE run_id = ? AND round_index = ?
            """,
            (run_id, round_index),
        ).fetchone()
        created_at = (
            str(existing["created_at"])
            if existing is not None and existing["created_at"]
            else (progress.prepared_timestamp if progress and progress.prepared_timestamp else str(batch["created_at"]))
        )
        mutated_children_count = int(existing["mutated_children_count"] or 0) if existing is not None else 0

        summary_payload: dict[str, object] = {
            "source": "service_backfill",
            "batch_id": str(batch["batch_id"]),
            "backend": str(batch["backend"]),
            "export_path": batch["export_path"],
            "service_status_reason": batch["service_status_reason"],
            "candidate_count": int(batch["candidate_count"] or 0),
            "terminal_submission_count": int(terminal_submission_count),
            "submission_status_counts": dict(submission_status_counts),
        }
        if progress is not None:
            summary_payload.update(
                {
                    "prepared_candidate_count": progress.candidate_count,
                    "prepared_selected_count": progress.selected_count,
                    "prepared_archived_count": progress.archived_count,
                    "prepared_timestamp": progress.prepared_timestamp,
                    "submitted_timestamp": progress.submitted_timestamp,
                }
            )
            if progress.generation_stage_metrics is not None:
                summary_payload["generation_stage_metrics"] = progress.generation_stage_metrics
        if pre_sim_metrics:
            summary_payload["pre_sim_metrics"] = pre_sim_metrics

        connection.execute(
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
                run_id,
                round_index,
                str(batch["status"]),
                int(generated_count),
                int(validated_count),
                int(submitted_count),
                int(terminal_submission_count),
                int(selected_for_mutation_count),
                int(mutated_children_count),
                json.dumps(summary_payload, sort_keys=True),
                created_at,
                now,
            ),
        )

        if round_index in existing_rounds:
            updated += 1
        else:
            inserted += 1

    connection.commit()
    return inserted, updated


def main() -> int:
    args = _parse_args()
    db_path = Path(args.db).resolve()
    progress_log_dir = Path(args.progress_log_dir).resolve()
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    if not progress_log_dir.exists():
        raise SystemExit(f"Progress log dir not found: {progress_log_dir}")

    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        progress_snapshots = _load_progress_snapshots(progress_log_dir, args.run_id)
        inserted, updated = _backfill_rounds(
            connection=connection,
            run_id=args.run_id,
            progress_snapshots=progress_snapshots,
            overwrite_existing=bool(args.overwrite_existing),
        )
        total_rows = connection.execute(
            "SELECT COUNT(*) AS total FROM closed_loop_rounds WHERE run_id = ?",
            (args.run_id,),
        ).fetchone()
        max_round = connection.execute(
            "SELECT MAX(round_index) AS max_round FROM closed_loop_rounds WHERE run_id = ?",
            (args.run_id,),
        ).fetchone()
        print(f"db={db_path}")
        print(f"run_id={args.run_id}")
        print(f"inserted={inserted}")
        print(f"updated={updated}")
        print(f"total_closed_loop_rounds={int(total_rows['total'] or 0)}")
        print(f"max_round_index={int(max_round['max_round'] or 0)}")
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
