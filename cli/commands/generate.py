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
    generate_and_persist(repository, config, environment, count=args.count)
    return 0
