from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config import AppConfig
from services.models import CommandEnvironment

logger = logging.getLogger(__name__)

DEFAULT_PROGRESS_LOG_SUBDIR = "progress_logs"


def resolve_progress_log_path(
    config: AppConfig,
    environment: CommandEnvironment,
) -> Path | None:
    runtime = getattr(config, "runtime", None)
    if runtime is not None and not bool(getattr(runtime, "progress_log_enabled", True)):
        return None

    raw_dir = str(getattr(runtime, "progress_log_dir", "") or "").strip() if runtime is not None else ""
    if raw_dir:
        root = _resolve_path(Path(raw_dir).expanduser())
    else:
        storage_path = str(getattr(config.storage, "path", "") or "").strip()
        if not storage_path or storage_path == ":memory:":
            return None
        root = _resolve_path(Path(storage_path).expanduser()).parent / DEFAULT_PROGRESS_LOG_SUBDIR

    return root / f"{environment.context.run_id}.jsonl"


def append_progress_event(
    config: AppConfig,
    environment: CommandEnvironment,
    *,
    event: str,
    stage: str = "",
    status: str | None = None,
    tick_id: int | None = None,
    round_index: int | None = None,
    batch_id: str | None = None,
    job_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> str | None:
    path = resolve_progress_log_path(config, environment)
    if path is None:
        return None

    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": environment.context.run_id,
        "command": environment.command_name,
        "profile_name": config.runtime.profile_name,
        "event": event,
    }
    if stage:
        record["stage"] = stage
    if status:
        record["status"] = status
    if tick_id is not None:
        record["tick_id"] = int(tick_id)
    if round_index is not None:
        record["round_index"] = int(round_index)
    if batch_id:
        record["batch_id"] = batch_id
    if job_id:
        record["job_id"] = job_id
    if payload:
        record["payload"] = payload

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")
    except (OSError, TypeError) as exc:
        logger.warning("Unable to write progress log event %s to %s: %s", event, path, exc)
        return None
    return str(path)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()
