from __future__ import annotations

import argparse
from collections import Counter

from adapters.brain_manual_adapter import BrainManualAdapter
from core.config import AppConfig
from services.brain_service import BrainService
from services.models import CommandEnvironment
from services.runtime_service import init_run
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "import-brain-results",
        help="Import manually collected BRAIN results into internal storage.",
        parents=[common],
    )
    parser.add_argument("--path", required=True, help="Path to the manual BRAIN result CSV.")
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    init_run(repository, config, environment, status="importing_brain_results")
    service = BrainService(
        repository,
        config.brain,
        adapter=BrainManualAdapter(export_root=config.brain.manual_export_dir),
    )
    results = service.import_manual_results(args.path, run_id=environment.context.run_id)
    repository.update_run_status(environment.context.run_id, "brain_results_imported")
    counts = Counter(result.status for result in results)
    print(f"run_id: {environment.context.run_id}")
    print(f"imported_results: {len(results)}")
    for status, total in sorted(counts.items()):
        print(f"status_{status}: {total}")
    return 0
