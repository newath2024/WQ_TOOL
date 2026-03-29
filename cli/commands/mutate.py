from __future__ import annotations

import argparse

from core.config import AppConfig
from services.models import CommandEnvironment
from services.mutation_service import mutate_and_persist
from services.runtime_service import init_run
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the mutate command."""
    parser = subparsers.add_parser("mutate", help="Create mutated variants from top-ranked alphas.", parents=[common])
    parser.add_argument("--from-top", type=int, default=20, help="Number of top parents to use.")
    parser.add_argument("--count", type=int, default=200, help="Number of mutations to attempt.")
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    """Execute the mutate command."""
    init_run(repository, config, environment, status="mutating")
    result = mutate_and_persist(repository, config, environment, from_top=args.from_top, count=args.count)
    return result.exit_code
