from __future__ import annotations

import argparse

from core.config import AppConfig
from services.evaluation_service import evaluate_and_persist
from services.models import CommandEnvironment
from services.runtime_service import init_run
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the evaluate command."""
    parser = subparsers.add_parser("evaluate", help="Backtest and evaluate stored alpha candidates.", parents=[common])
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    """Execute the evaluate command."""
    del args
    init_run(repository, config, environment, status="evaluating")
    result = evaluate_and_persist(repository, config, environment, status="evaluated")
    evaluated_path = result.export_paths.get("evaluated_alphas_latest_csv")
    selected_path = result.export_paths.get("selected_alphas_latest_csv")
    if evaluated_path:
        print(f"evaluated_alphas_csv: {evaluated_path}")
    if selected_path:
        print(f"selected_alphas_csv: {selected_path}")
    return 0
