from __future__ import annotations

import argparse

from core.config import AppConfig
from services.memory_service import get_failed_patterns, get_top_genes, get_top_patterns
from services.models import CommandEnvironment
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register memory inspection commands."""
    scope_help = "Which memory scope to inspect: local, global, or blended."
    patterns_parser = subparsers.add_parser(
        "memory-top-patterns",
        help="Display the highest scoring structural patterns for the current regime.",
        parents=[common],
    )
    patterns_parser.add_argument("--limit", type=int, default=10, help="Number of patterns to display.")
    patterns_parser.add_argument("--kind", default=None, help="Optional pattern kind filter.")
    patterns_parser.add_argument("--scope", choices=("local", "global", "blended"), default="blended", help=scope_help)
    patterns_parser.set_defaults(command_handler=handle_top_patterns)

    failed_parser = subparsers.add_parser(
        "memory-failed-patterns",
        help="Display the most failure-prone patterns for the current regime.",
        parents=[common],
    )
    failed_parser.add_argument("--limit", type=int, default=10, help="Number of patterns to display.")
    failed_parser.add_argument("--scope", choices=("local", "global", "blended"), default="blended", help=scope_help)
    failed_parser.set_defaults(command_handler=handle_failed_patterns)

    genes_parser = subparsers.add_parser(
        "memory-top-genes",
        help="Display the strongest reusable subexpression genes for the current regime.",
        parents=[common],
    )
    genes_parser.add_argument("--limit", type=int, default=10, help="Number of genes to display.")
    genes_parser.add_argument("--scope", choices=("local", "global", "blended"), default="blended", help=scope_help)
    genes_parser.set_defaults(command_handler=handle_top_genes)


def handle_top_patterns(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    """Execute the memory-top-patterns command."""
    rows = get_top_patterns(repository, config, environment, limit=args.limit, kind=args.kind, scope=args.scope)
    if not rows:
        return 1
    for row in rows:
        print(
            f"kind={row.pattern_kind:<13} score={row.pattern_score:.4f} support={row.support:<3} "
            f"success={row.success_count:<3} failure={row.failure_count:<3} value={row.pattern_value}"
        )
    return 0


def handle_failed_patterns(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    """Execute the memory-failed-patterns command."""
    rows = get_failed_patterns(repository, config, environment, limit=args.limit, scope=args.scope)
    if not rows:
        return 1
    for row in rows:
        print(
            f"kind={row.pattern_kind:<13} score={row.pattern_score:.4f} support={row.support:<3} "
            f"failure={row.failure_count:<3} value={row.pattern_value}"
        )
    return 0


def handle_top_genes(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    """Execute the memory-top-genes command."""
    rows = get_top_genes(repository, config, environment, limit=args.limit, scope=args.scope)
    if not rows:
        return 1
    for row in rows:
        print(
            f"score={row.pattern_score:.4f} support={row.support:<3} success={row.success_count:<3} "
            f"value={row.pattern_value}"
        )
    return 0
