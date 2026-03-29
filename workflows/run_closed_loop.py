from __future__ import annotations

from core.config import AppConfig
from services.closed_loop_service import ClosedLoopService
from services.models import ClosedLoopRunSummary, CommandEnvironment
from storage.repository import SQLiteRepository


def run_closed_loop(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
) -> ClosedLoopRunSummary:
    service = ClosedLoopService(repository)
    return service.run(config=config, environment=environment)
