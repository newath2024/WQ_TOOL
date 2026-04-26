from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config.builders import (
    _apply_generation_simulation_defaults,
    _build_adaptive_generation_config,
    _build_brain_config,
    _build_evaluation_config,
    _build_loop_config,
    _build_runtime_config,
    _build_service_config,
    _build_simulation_config,
    _build_submission_test_config,
    _period_from_mapping,
)
from config.models.evaluation import BacktestConfig
from config.models.generation import GenerationConfig
from config.models.runtime import AppConfig, AuxDataConfig, DataConfig, SplitConfig
from config.models.storage import StorageConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the root.")
    return payload


def load_config(path: str | Path) -> AppConfig:
    """Load and normalize YAML config, accepting both legacy and grouped evaluation schemas."""
    config_path = Path(path)
    payload = _read_yaml(config_path)
    try:
        backtest = BacktestConfig(**payload["backtest"])
        evaluation = _build_evaluation_config(payload["evaluation"])
        brain = _build_brain_config(payload.get("brain"))
        generation_payload = dict(payload["generation"])
        generation = _apply_generation_simulation_defaults(
            GenerationConfig(**generation_payload),
            generation_payload,
            brain,
        )
        return AppConfig(
            data=DataConfig(**payload["data"]),
            aux_data=AuxDataConfig(**payload.get("aux_data", {})),
            splits=SplitConfig(
                train=_period_from_mapping(payload["splits"], "train"),
                validation=_period_from_mapping(payload["splits"], "validation"),
                test=_period_from_mapping(payload["splits"], "test"),
            ),
            generation=generation,
            adaptive_generation=_build_adaptive_generation_config(
                payload.get("adaptive_generation")
            ),
            simulation=_build_simulation_config(payload.get("simulation"), backtest),
            backtest=backtest,
            evaluation=evaluation,
            submission_tests=_build_submission_test_config(
                payload.get("submission_tests"), evaluation
            ),
            storage=StorageConfig(**payload["storage"]),
            brain=brain,
            loop=_build_loop_config(payload.get("loop"), brain),
            service=_build_service_config(payload.get("service"), brain),
            runtime=_build_runtime_config(payload.get("runtime"), config_path),
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config section: {exc.args[0]}") from exc


__all__ = ["load_config", "_read_yaml"]
