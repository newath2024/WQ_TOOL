from __future__ import annotations

import argparse

from core.config import AppConfig
from services.generation_service import generate_and_persist
from services.models import CommandEnvironment
from services.runtime_service import init_run
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the generate command."""
    parser = subparsers.add_parser("generate", help="Generate alpha candidates.", parents=[common])
    parser.add_argument("--count", type=int, default=None, help="Total number of candidates to generate.")
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    """Execute the generate command."""
    init_run(repository, config, environment, status="generating")
    result = generate_and_persist(repository, config, environment, count=args.count)
    latest_path = result.export_paths.get("generated_alphas_latest_csv")
    if latest_path:
        print(f"generated_alphas_csv: {latest_path}")
    return 0
