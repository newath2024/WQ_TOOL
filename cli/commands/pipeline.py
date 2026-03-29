from __future__ import annotations

import argparse

from core.config import AppConfig
from services.models import CommandEnvironment
from services.pipeline_service import run_full_pipeline
from services.runtime_service import init_run
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the run-full-pipeline command."""
    parser = subparsers.add_parser("run-full-pipeline", help="Run load, generate, evaluate, and select.", parents=[common])
    parser.add_argument("--count", type=int, default=None, help="Override the initial generation count.")
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    """Execute the run-full-pipeline command."""
    init_run(repository, config, environment, status="pipeline_started")
    return run_full_pipeline(repository, config, environment, count=args.count)
