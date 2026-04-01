from __future__ import annotations

import json

from core.config import load_config
from core.run_context import RunContext
from services.models import CommandEnvironment
from services.progress_log import MAX_PROGRESS_LOG_BYTES, append_progress_event, resolve_progress_log_path


def test_progress_log_rotates_live_file_when_it_exceeds_limit(tmp_path) -> None:
    config = load_config("config/dev.yaml")
    config.runtime.progress_log_dir = str(tmp_path / "progress")
    environment = CommandEnvironment(
        config_path="config/dev.yaml",
        command_name="run-service",
        context=RunContext.create(seed=7, config_path="config/dev.yaml", run_id="run-rotate"),
    )
    path = resolve_progress_log_path(config, environment)
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * (MAX_PROGRESS_LOG_BYTES + 1), encoding="utf-8")

    append_progress_event(config, environment, event="rotated")

    rotated = path.with_name(f"{path.stem}.1{path.suffix}")
    assert rotated.exists()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[-1]["event"] == "rotated"


def test_progress_log_rotation_shifts_existing_files_and_prunes_oldest(tmp_path) -> None:
    config = load_config("config/dev.yaml")
    config.runtime.progress_log_dir = str(tmp_path / "progress")
    environment = CommandEnvironment(
        config_path="config/dev.yaml",
        command_name="run-service",
        context=RunContext.create(seed=7, config_path="config/dev.yaml", run_id="run-rotate-shift"),
    )
    path = resolve_progress_log_path(config, environment)
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text("live-file", encoding="utf-8")
    path.write_text(path.read_text(encoding="utf-8") + ("x" * MAX_PROGRESS_LOG_BYTES), encoding="utf-8")
    path.with_name(f"{path.stem}.1{path.suffix}").write_text("rotation-1", encoding="utf-8")
    path.with_name(f"{path.stem}.2{path.suffix}").write_text("rotation-2", encoding="utf-8")
    path.with_name(f"{path.stem}.3{path.suffix}").write_text("rotation-3", encoding="utf-8")
    path.with_name(f"{path.stem}.4{path.suffix}").write_text("rotation-4", encoding="utf-8")
    path.with_name(f"{path.stem}.5{path.suffix}").write_text("rotation-5", encoding="utf-8")

    append_progress_event(config, environment, event="rotate-shift")

    assert path.with_name(f"{path.stem}.1{path.suffix}").read_text(encoding="utf-8").startswith("live-file")
    assert path.with_name(f"{path.stem}.2{path.suffix}").read_text(encoding="utf-8") == "rotation-1"
    assert path.with_name(f"{path.stem}.3{path.suffix}").read_text(encoding="utf-8") == "rotation-2"
    assert path.with_name(f"{path.stem}.4{path.suffix}").read_text(encoding="utf-8") == "rotation-3"
    assert path.with_name(f"{path.stem}.5{path.suffix}").read_text(encoding="utf-8") == "rotation-4"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[-1]["event"] == "rotate-shift"
