from __future__ import annotations

from core.config import AppConfig
from services.models import CommandEnvironment, ServiceRunSummary
from services.service_runner import ServiceRunner
from storage.repository import SQLiteRepository


def run_service(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    *,
    max_ticks: int | None = None,
) -> ServiceRunSummary:
    runner = ServiceRunner(repository, config=config, environment=environment)
    return runner.run(max_ticks=max_ticks)
