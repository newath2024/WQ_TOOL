from __future__ import annotations

import argparse

from core.config import AppConfig
from services.models import CommandEnvironment
from services.runtime_service import init_run
from storage.repository import SQLiteRepository
from workflows.run_brain_simulation import run_brain_simulation


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "run-brain-sim",
        help="Generate/select candidates and run one BRAIN simulation batch with the configured backend.",
        parents=[common],
    )
    parser.add_argument("--count", type=int, default=None, help="Override the generation batch size.")
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    init_run(repository, config, environment, status="running_brain_sim")
    batch = run_brain_simulation(repository, config, environment, count=args.count)
    repository.update_run_status(environment.context.run_id, "brain_sim_completed")
    print(f"run_id: {environment.context.run_id}")
    print(f"batch_id: {batch.batch_id}")
    print(f"backend: {batch.backend}")
    print(f"status: {batch.status}")
    print(f"submitted: {batch.submitted_count}")
    print(f"completed: {batch.completed_count}")
    if batch.export_path:
        print(f"export_path: {batch.export_path}")
    return 0
