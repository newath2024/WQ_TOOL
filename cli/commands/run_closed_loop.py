from __future__ import annotations

import argparse

from core.config import AppConfig
from services.models import CommandEnvironment
from services.runtime_service import init_run
from storage.repository import SQLiteRepository
from workflows.run_closed_loop import run_closed_loop


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "run-closed-loop",
        help="Run the BRAIN-first closed-loop alpha research workflow.",
        parents=[common],
    )
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    del args
    init_run(repository, config, environment, status="running_closed_loop")
    summary = run_closed_loop(repository, config, environment)
    repository.update_run_status(environment.context.run_id, summary.status, finished=summary.status == "completed")
    print(f"run_id: {summary.run_id}")
    print(f"backend: {summary.backend}")
    print(f"status: {summary.status}")
    print(f"rounds_completed: {len(summary.rounds)}")
    for round_summary in summary.rounds:
        print(
            f"round_{round_summary.round_index}: status={round_summary.status}; generated={round_summary.generated_count}; submitted={round_summary.submitted_count}; "
            f"completed={round_summary.completed_count}; next_mutations={round_summary.mutated_children_count}"
        )
    return 0
