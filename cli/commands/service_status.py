from __future__ import annotations

import argparse
import json

from core.config import AppConfig
from services.models import CommandEnvironment
from services.status_service import (
    ServiceStatusSnapshot,
    build_service_status_snapshot,
    service_status_snapshot_to_dict,
)
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "service-status",
        help="Show the current BRAIN service runtime, active batch, and recent alpha outcomes.",
        parents=[common],
    )
    parser.add_argument(
        "--service-name",
        default=None,
        help="Override the configured service lock name. Defaults to service.lock_name from config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of recent batches, submissions, and results to print.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON document instead of a human-readable summary.",
    )
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    del environment
    service_name = str(args.service_name or config.service.lock_name)
    snapshot = build_service_status_snapshot(
        repository,
        service_name=service_name,
        run_id=args.run_id,
        limit=max(int(args.limit), 1),
    )
    if snapshot.run_id is None and snapshot.runtime is None:
        print("No service runtime or run history found.")
        return 1

    if args.json:
        print(json.dumps(service_status_snapshot_to_dict(snapshot), ensure_ascii=False, indent=2, default=str))
    else:
        _print_human(snapshot)
    return 0


def _print_human(snapshot: ServiceStatusSnapshot) -> None:
    print(f"service_name: {snapshot.service_name}")
    print(f"run_id: {snapshot.run_id or '-'}")
    print(f"service_status: {snapshot.runtime.status if snapshot.runtime is not None else 'not_running'}")
    if snapshot.runtime is not None:
        print(f"pending_job_count: {snapshot.runtime.pending_job_count}")
        print(f"tick_id: {snapshot.runtime.tick_id}")
        print(f"updated_at: {snapshot.runtime.updated_at}")
        if snapshot.runtime.persona_url:
            print(f"persona_url: {snapshot.runtime.persona_url}")
        if snapshot.runtime.last_error:
            print(f"last_error: {snapshot.runtime.last_error}")

    if snapshot.run is not None:
        print(f"run_status: {snapshot.run.status}")
        print(f"run_started_at: {snapshot.run.started_at}")
        if snapshot.run.finished_at:
            print(f"run_finished_at: {snapshot.run.finished_at}")

    if snapshot.active_batch is not None:
        print(f"active_batch_id: {snapshot.active_batch.batch_id}")
        print(f"active_batch_status: {snapshot.active_batch.status}")
        print(f"active_batch_candidates: {snapshot.active_batch.candidate_count}")
        if snapshot.active_batch.service_status_reason:
            print(f"active_batch_reason: {snapshot.active_batch.service_status_reason}")
        print(
            "active_batch_submission_counts: "
            f"{_format_counts(snapshot.active_batch_submission_counts)}"
        )
    else:
        print("active_batch_id: -")

    print(f"batch_counts: {_format_counts(snapshot.batch_counts)}")
    print(f"submission_counts: {_format_counts(snapshot.submission_counts)}")
    print(f"result_counts: {_format_counts(snapshot.result_counts)}")

    if snapshot.recent_batches:
        print("recent_batches:")
        for row in snapshot.recent_batches:
            suffix = f" reason={row.service_status_reason}" if row.service_status_reason else ""
            print(
                f"  batch_id={row.batch_id} round={row.round_index} status={row.status} "
                f"candidates={row.candidate_count}{suffix}"
            )

    if snapshot.recent_submissions:
        print("recent_submissions:")
        for row in snapshot.recent_submissions:
            print(
                f"  job_id={row.job_id} batch={row.batch_id} status={row.status} "
                f"candidate={row.candidate_id} updated_at={row.updated_at}"
            )

    if snapshot.recent_results:
        print("recent_results:")
        for row in snapshot.recent_results:
            print(
                f"  job_id={row.job_id} batch={row.batch_id} status={row.status} "
                f"fitness={_format_metric(row.fitness)} sharpe={_format_metric(row.sharpe)} "
                f"turnover={_format_metric(row.turnover)}"
            )


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return " ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"
