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
class EvaluationHardFiltersConfig:
    min_validation_sharpe: float = -10.0
    max_validation_turnover: float = 5.0
    max_validation_drawdown: float = 0.95


@dataclass(slots=True)
class EvaluationDataRequirementsConfig:
    min_validation_observations: int = 5
    min_stability: float = 0.10


@dataclass(slots=True)
class EvaluationDiversityConfig:
    signal_correlation_threshold: float = 0.95
    returns_correlation_threshold: float = 0.95


@dataclass(slots=True)
class EvaluationRankingConfig:
    top_k: int = 20
    use_behavioral_novelty_tiebreak: bool = True


@dataclass(slots=True)
class EvaluationRobustnessConfig:
    enable_subuniverse_test: bool = True
    enable_ladder_test: bool = True
    enable_robustness_test: bool = True
    ladder_buckets: int = 3
    ladder_min_sharpe: float = 0.0
    ladder_min_passes: int = 2
    subuniverse_min_sharpe: float = -0.25
    subuniverse_min_pass_fraction: float = 0.5
    robustness_min_fitness_ratio: float = 0.35


@dataclass(slots=True, init=False)
class EvaluationConfig:
    hard_filters: EvaluationHardFiltersConfig = field(default_factory=EvaluationHardFiltersConfig)
    data_requirements: EvaluationDataRequirementsConfig = field(default_factory=EvaluationDataRequirementsConfig)
    diversity: EvaluationDiversityConfig = field(default_factory=EvaluationDiversityConfig)
    ranking: EvaluationRankingConfig = field(default_factory=EvaluationRankingConfig)
    robustness: EvaluationRobustnessConfig = field(default_factory=EvaluationRobustnessConfig)

    def __init__(
        self,
        hard_filters: EvaluationHardFiltersConfig | None = None,
        data_requirements: EvaluationDataRequirementsConfig | None = None,
        diversity: EvaluationDiversityConfig | None = None,
        ranking: EvaluationRankingConfig | None = None,
        robustness: EvaluationRobustnessConfig | None = None,
        min_sharpe: float | None = None,
        max_turnover: float | None = None,
        min_observations: int | None = None,
        max_drawdown: float | None = None,
        min_stability: float | None = None,
        signal_correlation_threshold: float | None = None,
        returns_correlation_threshold: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self.hard_filters = hard_filters or EvaluationHardFiltersConfig(
            min_validation_sharpe=-10.0 if min_sharpe is None else float(min_sharpe),
            max_validation_turnover=5.0 if max_turnover is None else float(max_turnover),
            max_validation_drawdown=0.95 if max_drawdown is None else float(max_drawdown),
        )
        self.data_requirements = data_requirements or EvaluationDataRequirementsConfig(
            min_validation_observations=5 if min_observations is None else int(min_observations),
            min_stability=0.10 if min_stability is None else float(min_stability),
        )
        self.diversity = diversity or EvaluationDiversityConfig(
            signal_correlation_threshold=0.95
            if signal_correlation_threshold is None
            else float(signal_correlation_threshold),
            returns_correlation_threshold=0.95
            if returns_correlation_threshold is None
            else float(returns_correlation_threshold),
        )
        self.ranking = ranking or EvaluationRankingConfig(top_k=20 if top_k is None else int(top_k))
        self.robustness = robustness or EvaluationRobustnessConfig()

    @property
    def min_sharpe(self) -> float:
        return self.hard_filters.min_validation_sharpe

    @property
    def max_turnover(self) -> float:
        return self.hard_filters.max_validation_turnover

    @property
    def max_drawdown(self) -> float:
        return self.hard_filters.max_validation_drawdown

    @property
    def min_observations(self) -> int:
        return self.data_requirements.min_validation_observations

    @property
    def min_stability(self) -> float:
        return self.data_requirements.min_stability

    @property
    def signal_correlation_threshold(self) -> float:
        return self.diversity.signal_correlation_threshold

    @property
    def returns_correlation_threshold(self) -> float:
        return self.diversity.returns_correlation_threshold

    @property
    def top_k(self) -> int:
        return self.ranking.top_k


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
    profile_name: str = ""


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
        """Return the config as a plain nested mapping."""
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


def _build_evaluation_config(payload: dict[str, Any]) -> EvaluationConfig:
    if any(key in payload for key in ("hard_filters", "data_requirements", "diversity", "ranking", "robustness")):
        return EvaluationConfig(
            hard_filters=EvaluationHardFiltersConfig(**payload.get("hard_filters", {})),
            data_requirements=EvaluationDataRequirementsConfig(**payload.get("data_requirements", {})),
            diversity=EvaluationDiversityConfig(**payload.get("diversity", {})),
            ranking=EvaluationRankingConfig(**payload.get("ranking", {})),
            robustness=EvaluationRobustnessConfig(**payload.get("robustness", {})),
        )

    return EvaluationConfig(
        hard_filters=EvaluationHardFiltersConfig(
            min_validation_sharpe=float(payload["min_sharpe"]),
            max_validation_turnover=float(payload["max_turnover"]),
            max_validation_drawdown=float(payload["max_drawdown"]),
        ),
        data_requirements=EvaluationDataRequirementsConfig(
            min_validation_observations=int(payload["min_observations"]),
            min_stability=float(payload["min_stability"]),
        ),
        diversity=EvaluationDiversityConfig(
            signal_correlation_threshold=float(payload["signal_correlation_threshold"]),
            returns_correlation_threshold=float(payload["returns_correlation_threshold"]),
        ),
        ranking=EvaluationRankingConfig(top_k=int(payload["top_k"])),
    )


def _build_submission_test_config(
    payload: dict[str, Any] | None,
    evaluation: EvaluationConfig,
) -> SubmissionTestConfig:
    if payload is not None:
        return SubmissionTestConfig(**payload)
    return SubmissionTestConfig(**asdict(evaluation.robustness))


def _build_runtime_config(payload: dict[str, Any] | None, config_path: Path) -> RuntimeConfig:
    runtime_payload = dict(payload or {})
    runtime_payload.setdefault("profile_name", config_path.stem)
    return RuntimeConfig(**runtime_payload)


def load_config(path: str | Path) -> AppConfig:
    """Load and normalize YAML config, accepting both legacy and grouped evaluation schemas."""
    config_path = Path(path)
    payload = _read_yaml(config_path)
    try:
        backtest = BacktestConfig(**payload["backtest"])
        evaluation = _build_evaluation_config(payload["evaluation"])
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
            evaluation=evaluation,
            submission_tests=_build_submission_test_config(payload.get("submission_tests"), evaluation),
            storage=StorageConfig(**payload["storage"]),
            runtime=_build_runtime_config(payload.get("runtime"), config_path),
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config section: {exc.args[0]}") from exc
