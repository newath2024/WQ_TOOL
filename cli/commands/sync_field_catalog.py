from __future__ import annotations

import argparse

from core.config import AppConfig
from services.data_service import load_research_context, persist_research_metadata
from services.models import CommandEnvironment
from services.runtime_service import init_run
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "sync-field-catalog",
        help="Load field catalog/runtime metadata and persist it for the active run.",
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
    init_run(repository, config, environment, status="syncing_field_catalog")
    research_context = load_research_context(config, environment, stage="sync-field-catalog-data")
    persist_research_metadata(repository, config, environment, research_context)
    total_fields = len(research_context.field_registry.fields)
    runtime_fields = sum(1 for spec in research_context.field_registry.fields.values() if spec.runtime_available)
    repository.update_run_status(environment.context.run_id, "field_catalog_synced")
    print(f"run_id: {environment.context.run_id}")
    print(f"field_catalog_total: {total_fields}")
    print(f"runtime_fields: {runtime_fields}")
    return 0
