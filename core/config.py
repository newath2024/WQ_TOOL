from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
class GenerationConfig:
    allowed_fields: list[str]
    allowed_operators: list[str]
    lookbacks: list[int]
    max_depth: int
    complexity_limit: int
    template_count: int
    grammar_count: int
    mutation_count: int
    normalization_wrappers: list[str]
    random_seed: int = 7


@dataclass(slots=True)
class StrategyMixConfig:
    guided_mutation: float = 0.40
    memory_templates: float = 0.30
    random_exploration: float = 0.20
    novelty_behavior: float = 0.10


@dataclass(slots=True)
class CriticThresholdConfig:
    turnover_warning_fraction: float = 0.85
    overfit_gap_threshold: float = 0.35
    complexity_warning_fraction: float = 0.80
    noisy_short_horizon_max_lookback: int = 3
    novelty_success_threshold: float = 0.70
    score_prior_weight: float = 3.0


@dataclass(slots=True)
class AdaptiveGenerationConfig:
    enabled: bool = True
    memory_scope: str = "regime"
    success_rule: str = "validation_first"
    strategy_mix: StrategyMixConfig = field(default_factory=StrategyMixConfig)
    exploration_epsilon: float = 0.10
    sampling_temperature: float = 0.75
    family_cap_fraction: float = 0.25
    parent_pool_size: int = 30
    novelty_reference_top_k: int = 20
    min_pattern_support: int = 3
    pattern_decay: float = 0.98
    critic_thresholds: CriticThresholdConfig = field(default_factory=CriticThresholdConfig)


@dataclass(slots=True)
class SubuniverseConfig:
    name: str
    mask_field: str | None = None
    top_n_by: str | None = None
    top_n: int | None = None


@dataclass(slots=True)
class SimulationConfig:
    delay_mode: str = "d1"
    neutralization: str = "none"
    secondary_neutralization: str | None = None
    pasteurize: bool = False
    signal_clip: float | None = None
    weight_clip: float | None = None
    factor_columns: list[str] = field(default_factory=list)
    subuniverses: list[SubuniverseConfig] = field(default_factory=list)
    robustness_windows: list[int] = field(default_factory=lambda: [21, 42])
    cache_enabled: bool = True


@dataclass(slots=True)
class BacktestConfig:
    timeframe: str
    mode: str = "cross_sectional"
    portfolio_construction: str = "long_short"
    selection_fraction: float = 0.2
    signal_delay: int = 1
    holding_period: int = 3
    volatility_scaling: bool = False
    volatility_lookback: int = 20
    transaction_cost_bps: float = 0.0
    annualization_factor: int = 252
    symbol_rank_window: int = 20
    upper_quantile: float = 0.8
    lower_quantile: float = 0.2
    turnover_penalty: float = 0.1
    drawdown_penalty: float = 0.5


@dataclass(slots=True)
class EvaluationConfig:
    min_sharpe: float
    max_turnover: float
    min_observations: int
    max_drawdown: float
    min_stability: float
    signal_correlation_threshold: float
    returns_correlation_threshold: float
    top_k: int


@dataclass(slots=True)
class SubmissionTestConfig:
    enable_subuniverse_test: bool = True
    enable_ladder_test: bool = True
    enable_robustness_test: bool = True
    ladder_buckets: int = 3
    ladder_min_sharpe: float = 0.0
    ladder_min_passes: int = 2
    subuniverse_min_sharpe: float = -0.25
    subuniverse_min_pass_fraction: float = 0.5
    robustness_min_fitness_ratio: float = 0.35


@dataclass(slots=True)
class StorageConfig:
    path: str


@dataclass(slots=True)
class RuntimeConfig:
    log_level: str = "INFO"
    fail_fast: bool = False


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the root.")
    return payload


def _period_from_mapping(payload: dict[str, Any], key: str) -> PeriodConfig:
    try:
        period_payload = payload[key]
        return PeriodConfig(start=str(period_payload["start"]), end=str(period_payload["end"]))
    except KeyError as exc:
        raise ValueError(f"Missing split definition for '{key}'.") from exc


def _build_subuniverses(payload: list[dict[str, Any]] | None) -> list[SubuniverseConfig]:
    return [SubuniverseConfig(**item) for item in (payload or [])]


def _build_simulation_config(payload: dict[str, Any] | None, backtest: BacktestConfig) -> SimulationConfig:
    if payload is None:
        delay_mode = "d0" if backtest.signal_delay <= 0 else "d1"
        return SimulationConfig(delay_mode=delay_mode)
    simulation_payload = dict(payload)
    simulation_payload["subuniverses"] = _build_subuniverses(simulation_payload.get("subuniverses"))
    return SimulationConfig(**simulation_payload)


def _build_adaptive_generation_config(payload: dict[str, Any] | None) -> AdaptiveGenerationConfig:
    if payload is None:
        return AdaptiveGenerationConfig()
    adaptive_payload = dict(payload)
    adaptive_payload["strategy_mix"] = StrategyMixConfig(**adaptive_payload.get("strategy_mix", {}))
    adaptive_payload["critic_thresholds"] = CriticThresholdConfig(**adaptive_payload.get("critic_thresholds", {}))
    return AdaptiveGenerationConfig(**adaptive_payload)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    payload = _read_yaml(config_path)
    try:
        backtest = BacktestConfig(**payload["backtest"])
        return AppConfig(
            data=DataConfig(**payload["data"]),
            aux_data=AuxDataConfig(**payload.get("aux_data", {})),
            splits=SplitConfig(
                train=_period_from_mapping(payload["splits"], "train"),
                validation=_period_from_mapping(payload["splits"], "validation"),
                test=_period_from_mapping(payload["splits"], "test"),
            ),
            generation=GenerationConfig(**payload["generation"]),
            adaptive_generation=_build_adaptive_generation_config(payload.get("adaptive_generation")),
            simulation=_build_simulation_config(payload.get("simulation"), backtest),
            backtest=backtest,
            evaluation=EvaluationConfig(**payload["evaluation"]),
            submission_tests=SubmissionTestConfig(**payload.get("submission_tests", {})),
            storage=StorageConfig(**payload["storage"]),
            runtime=RuntimeConfig(**payload.get("runtime", {})),
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config section: {exc.args[0]}") from exc
