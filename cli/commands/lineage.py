from __future__ import annotations

import argparse

from core.config import AppConfig
from services.lineage_service import get_lineage
from services.models import CommandEnvironment
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the lineage command."""
    parser = subparsers.add_parser("lineage", help="Display lineage for one alpha.", parents=[common])
    parser.add_argument("--alpha-id", required=True, help="Alpha identifier to inspect.")
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    """Execute the lineage command."""
    del config
    rows = get_lineage(repository, environment, alpha_id=args.alpha_id)
    if not rows:
        return 1
    for row in rows:
        print(
            f"depth={row.depth:<2} run_id={row.run_id} alpha_id={row.alpha_id} "
            f"outcome={row.outcome_score if row.outcome_score is not None else 'n/a'} "
            f"fail_tags={','.join(row.fail_tags) or '-'} expr={row.expression}"
        )
    return 0
