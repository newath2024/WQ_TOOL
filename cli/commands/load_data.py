from __future__ import annotations

import argparse

from core.config import AppConfig
from services.data_service import load_and_persist_dataset
from services.models import CommandEnvironment
from services.runtime_service import init_run
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the load-data command."""
    parser = subparsers.add_parser(
        "load-data",
        help="Load, validate, and summarize the configured dataset.",
        parents=[common],
    )
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    """Execute the load-data command."""
    del args
    init_run(repository, config, environment, status="loading_data")
    load_and_persist_dataset(repository, config, environment, stage="load-data")
    repository.update_run_status(environment.context.run_id, "data_loaded")
    return 0
