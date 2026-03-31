from __future__ import annotations

import argparse

from core.config import AppConfig
from storage.repository import SQLiteRepository
from workflows.run_service import run_service


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "run-service",
        help="Run the long-lived BRAIN service loop in the foreground.",
        parents=[common],
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Optional cap for service ticks. Useful for smoke tests and supervised rollouts.",
    )
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment,
) -> int:
    summary = run_service(repository, config, environment, max_ticks=args.max_ticks)
    print(f"run_id: {summary.run_id}")
    print(f"service_name: {summary.service_name}")
    print(f"status: {summary.status}")
    print(f"ticks_executed: {summary.ticks_executed}")
    print(f"pending_job_count: {summary.pending_job_count}")
    if summary.progress_log_path:
        print(f"progress_log_path: {summary.progress_log_path}")
    return 0 if summary.status != "lock_denied" else 1
