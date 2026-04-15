from __future__ import annotations

import yaml

from core.config import AppConfig
from core.run_context import RunContext
from services.models import CommandEnvironment
from storage.repository import SQLiteRepository

RESUMABLE_COMMANDS = {
    "generate",
    "evaluate",
    "mutate",
    "top",
    "report",
    "memory-top-patterns",
    "memory-failed-patterns",
    "memory-top-genes",
    "lineage",
    "sync-field-catalog",
    "export-brain-candidates",
    "import-brain-results",
    "run-brain-sim",
    "run-closed-loop",
    "brain-login",
    "service-status",
    "recover-brain-jobs",
}


def open_repository(config: AppConfig) -> SQLiteRepository:
    """Open the configured repository backend."""
    return SQLiteRepository(config.storage.path)


def resolve_run_context(
    repository: SQLiteRepository,
    config_path: str,
    seed: int,
    run_id: str | None,
    resume: bool,
    command: str,
) -> RunContext:
    """Resolve the active run context for a command."""
    if run_id:
        existing = repository.get_run(run_id)
        if existing:
            return RunContext(
                run_id=existing.run_id,
                seed=existing.seed,
                started_at=existing.started_at,
                config_path=existing.config_path,
            )
        return RunContext.create(seed=seed, config_path=config_path, run_id=run_id)

    if resume or command in RESUMABLE_COMMANDS:
        existing = repository.get_latest_run()
        if existing:
            return RunContext(
                run_id=existing.run_id,
                seed=existing.seed,
                started_at=existing.started_at,
                config_path=existing.config_path,
            )

    return RunContext.create(seed=seed, config_path=config_path)


def build_command_environment(
    config_path: str,
    command_name: str,
    context: RunContext,
) -> CommandEnvironment:
    """Create the shared environment passed to command handlers."""
    return CommandEnvironment(config_path=config_path, command_name=command_name, context=context)


def init_run(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    status: str,
) -> None:
    """Insert or update the run row with the current status and config snapshot."""
    repository.upsert_run(
        run_id=environment.context.run_id,
        seed=environment.context.seed,
        config_path=environment.context.config_path,
        config_snapshot=yaml.safe_dump(config.to_dict(), sort_keys=False),
        status=status,
        started_at=environment.context.started_at,
        profile_name=config.runtime.profile_name,
        selected_timeframe=config.backtest.timeframe,
        region=config.brain.region,
        entry_command=environment.command_name,
    )


def short_run_metadata(config: AppConfig) -> dict[str, str]:
    """Return compact metadata derived from the active config."""
    return {
        "profile_name": config.runtime.profile_name,
        "selected_timeframe": config.backtest.timeframe,
    }
