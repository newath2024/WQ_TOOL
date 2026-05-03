from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from config.models.adaptive import AdaptiveGenerationConfig
from config.models.brain import BrainConfig
from config.models.evaluation import (
    BacktestConfig,
    EvaluationConfig,
    SimulationConfig,
    SubmissionTestConfig,
)
from config.models.generation import GenerationConfig
from config.models.quality import QualityScoreConfig
from config.models.service import ServiceConfig
from config.models.storage import StorageConfig


@dataclass(slots=True)
class PeriodConfig:
    start: str
    end: str


@dataclass(slots=True)
class SplitConfig:
    train: PeriodConfig
    validation: PeriodConfig
    test: PeriodConfig


@dataclass(slots=True)
class DataConfig:
    path: str
    format: str = "csv"
    input_layout: str = "canonical"
    default_timeframe: str = "1d"
    selected_timeframe: str = "1d"
    filename_pattern: str = "{symbol}_{timeframe}.csv"
    column_mapping: dict[str, str] = field(default_factory=dict)
    universe: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AuxDataConfig:
    group_path: str | None = None
    factor_path: str | None = None
    mask_path: str | None = None
    format: str = "csv"
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"
    group_columns: list[str] = field(default_factory=list)
    factor_columns: list[str] = field(default_factory=list)
    mask_columns: list[str] = field(default_factory=list)
    column_mapping: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LoopConfig:
    rounds: int = 5
    generation_batch_size: int = 100
    simulation_batch_size: int = 20
    poll_interval_seconds: int = 10
    timeout_seconds: int = 600
    mutate_top_k: int = 10
    max_children_per_parent: int = 5
    rejection_filters: list[str] = field(default_factory=list)
    archive_thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rounds <= 0:
            raise ValueError("loop.rounds must be > 0")
        if self.generation_batch_size <= 0:
            raise ValueError("loop.generation_batch_size must be > 0")
        if self.simulation_batch_size <= 0:
            raise ValueError("loop.simulation_batch_size must be > 0")
        if self.poll_interval_seconds <= 0:
            raise ValueError("loop.poll_interval_seconds must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("loop.timeout_seconds must be > 0")
        if self.mutate_top_k <= 0:
            raise ValueError("loop.mutate_top_k must be > 0")
        if self.max_children_per_parent <= 0:
            raise ValueError("loop.max_children_per_parent must be > 0")


@dataclass(slots=True)
class RuntimeConfig:
    log_level: str = "INFO"
    fail_fast: bool = False
    profile_name: str = ""
    progress_log_enabled: bool = True
    progress_log_dir: str = ""


@dataclass(slots=True)
class AppConfig:
    data: DataConfig
    aux_data: AuxDataConfig
    splits: SplitConfig
    generation: GenerationConfig
    adaptive_generation: AdaptiveGenerationConfig
    simulation: SimulationConfig
    backtest: BacktestConfig
    evaluation: EvaluationConfig
    submission_tests: SubmissionTestConfig
    storage: StorageConfig
    runtime: RuntimeConfig
    quality_score: QualityScoreConfig = field(default_factory=QualityScoreConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain nested mapping."""
        return asdict(self)


__all__ = [
    "PeriodConfig",
    "SplitConfig",
    "DataConfig",
    "AuxDataConfig",
    "LoopConfig",
    "RuntimeConfig",
    "AppConfig",
]
