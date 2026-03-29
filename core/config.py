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
    field_catalog_paths: list[str] = field(default_factory=list)
    operator_catalog_paths: list[str] = field(default_factory=list)
    field_value_paths: list[str] = field(default_factory=list)
    field_score_weights: dict[str, float] = field(
        default_factory=lambda: {"coverage": 0.50, "usage": 0.30, "category": 0.20}
    )
    category_weights: dict[str, float] = field(
        default_factory=lambda: {
            "price": 1.00,
            "volume": 0.85,
            "fundamental": 0.95,
            "analyst": 0.90,
            "model": 0.85,
            "sentiment": 0.75,
            "risk": 0.70,
            "macro": 0.65,
            "group": 0.60,
            "other": 0.50,
        }
    )
    template_weights: dict[str, float] = field(default_factory=dict)
    template_pool_size: int = 200
    max_turnover_bias: float = 0.35


@dataclass(slots=True)
class StrategyMixConfig:
    guided_mutation: float = 0.40
    memory_templates: float = 0.30
    random_exploration: float = 0.20
    novelty_behavior: float = 0.10


@dataclass(slots=True)
class MutationModeWeightsConfig:
    exploit_local: float = 0.35
    structural: float = 0.25
    crossover: float = 0.15
    novelty: float = 0.15
    repair: float = 0.10


@dataclass(slots=True)
class DiversityThresholdConfig:
    max_family_fraction: float = 0.25
    max_field_category_fraction: float = 0.50
    max_horizon_bucket_fraction: float = 0.40
    max_operator_path_fraction: float = 0.40
    exploration_quota_fraction: float = 0.20
    min_structural_distance: float = 0.08


@dataclass(slots=True)
class RepairPolicyConfig:
    enabled: bool = True
    max_attempts: int = 3
    allow_complexity_reduction: bool = True
    allow_turnover_reduction: bool = True
    allow_wrapper_cleanup: bool = True
    allow_group_fixups: bool = True


@dataclass(slots=True)
class CriticThresholdConfig:
    turnover_warning_fraction: float = 0.85
    overfit_gap_threshold: float = 0.35
    complexity_warning_fraction: float = 0.80
    noisy_short_horizon_max_lookback: int = 3
    novelty_success_threshold: float = 0.70
    score_prior_weight: float = 3.0


@dataclass(slots=True)
class RegionLearningConfig:
    enabled: bool = True
    local_scope: str = "region_regime"
    global_prior_scope: str = "match_non_region_regime"
    blend_mode: str = "linear_ramp"
    min_local_pattern_samples: int = 20
    full_local_pattern_samples: int = 100
    min_local_case_samples: int = 10
    full_local_case_samples: int = 50
    allow_global_parent_fallback: bool = False

    def __post_init__(self) -> None:
        self.local_scope = str(self.local_scope or "region_regime").strip().lower()
        self.global_prior_scope = str(self.global_prior_scope or "match_non_region_regime").strip().lower()
        self.blend_mode = str(self.blend_mode or "linear_ramp").strip().lower()
        if self.min_local_pattern_samples < 0:
            raise ValueError("adaptive_generation.region_learning.min_local_pattern_samples must be >= 0")
        if self.full_local_pattern_samples < self.min_local_pattern_samples:
            raise ValueError(
                "adaptive_generation.region_learning.full_local_pattern_samples must be >= min_local_pattern_samples"
            )
        if self.min_local_case_samples < 0:
            raise ValueError("adaptive_generation.region_learning.min_local_case_samples must be >= 0")
        if self.full_local_case_samples < self.min_local_case_samples:
            raise ValueError(
                "adaptive_generation.region_learning.full_local_case_samples must be >= min_local_case_samples"
            )


@dataclass(slots=True)
class AdaptiveGenerationConfig:
    enabled: bool = True
    memory_scope: str = "regime"
    success_rule: str = "validation_first"
    strategy_mix: StrategyMixConfig = field(default_factory=StrategyMixConfig)
    exploration_ratio: float = 0.35
    novelty_weight: float = 0.25
    mutation_mode_weights: MutationModeWeightsConfig = field(default_factory=MutationModeWeightsConfig)
    crossover_rate: float = 0.15
    diversity: DiversityThresholdConfig = field(default_factory=DiversityThresholdConfig)
    repair_policy: RepairPolicyConfig = field(default_factory=RepairPolicyConfig)
    exploration_epsilon: float = 0.10
    sampling_temperature: float = 0.75
    family_cap_fraction: float = 0.25
    parent_pool_size: int = 30
    novelty_reference_top_k: int = 20
    min_pattern_support: int = 3
    pattern_decay: float = 0.98
    critic_thresholds: CriticThresholdConfig = field(default_factory=CriticThresholdConfig)
    region_learning: RegionLearningConfig = field(default_factory=RegionLearningConfig)


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
class BrainConfig:
    backend: str = "manual"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    neutralization: str = "sector"
    decay: int = 0
    truncation: float = 0.08
    pasteurization: bool = True
    unit_handling: str = "verify"
    nan_handling: str = "off"
    poll_interval_seconds: int = 10
    timeout_seconds: int = 600
    max_retries: int = 3
    batch_size: int = 20
    manual_export_dir: str = "outputs/brain_manual"
    api_base_url: str = ""
    api_auth_env: str = "BRAIN_API_TOKEN"
    email_env: str = "BRAIN_API_EMAIL"
    password_env: str = "BRAIN_API_PASSWORD"
    credentials_file: str = "secrets/brain_credentials.json"
    session_path: str = "outputs/brain_api_session.json"
    auth_expiry_seconds: int = 14400
    open_browser_for_persona: bool = True
    persona_poll_interval_seconds: int = 15
    persona_timeout_seconds: int = 1800
    rate_limit_per_minute: int = 60

    def __post_init__(self) -> None:
        self.backend = str(self.backend).strip().lower()
        self.region = str(self.region).strip().upper()
        self.universe = str(self.universe).strip().upper()
        self.neutralization = _normalize_brain_enum(self.neutralization, true_value="ON", false_value="OFF")
        self.unit_handling = _normalize_brain_enum(self.unit_handling, true_value="VERIFY", false_value="IGNORE")
        self.nan_handling = _normalize_brain_enum(self.nan_handling, true_value="ON", false_value="OFF")
        allowed_backends = {"manual", "api"}
        if self.backend not in allowed_backends:
            raise ValueError(f"brain.backend must be one of {sorted(allowed_backends)}")
        if self.delay < 0:
            raise ValueError("brain.delay must be >= 0")
        if self.poll_interval_seconds <= 0:
            raise ValueError("brain.poll_interval_seconds must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("brain.timeout_seconds must be > 0")
        if self.max_retries < 0:
            raise ValueError("brain.max_retries must be >= 0")
        if self.batch_size <= 0:
            raise ValueError("brain.batch_size must be > 0")
        if not 1 <= self.auth_expiry_seconds <= 14400:
            raise ValueError("brain.auth_expiry_seconds must be between 1 and 14400")
        if self.persona_poll_interval_seconds <= 0:
            raise ValueError("brain.persona_poll_interval_seconds must be > 0")
        if self.persona_timeout_seconds <= 0:
            raise ValueError("brain.persona_timeout_seconds must be > 0")
        if self.rate_limit_per_minute <= 0:
            raise ValueError("brain.rate_limit_per_minute must be > 0")


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
class ServiceConfig:
    enabled: bool = False
    tick_interval_seconds: int = 5
    idle_sleep_seconds: int = 30
    poll_interval_seconds: int = 15
    max_pending_jobs: int = 20
    max_consecutive_failures: int = 5
    cooldown_seconds: int = 300
    heartbeat_interval_seconds: int = 30
    lock_name: str = "brain-service"
    lock_lease_seconds: int = 60
    resume_incomplete_jobs: bool = True
    shutdown_grace_period_seconds: int = 30
    stuck_job_after_seconds: int = 1800
    persona_retry_interval_seconds: int = 300
    persona_email_cooldown_seconds: int = 900

    def __post_init__(self) -> None:
        if self.tick_interval_seconds <= 0:
            raise ValueError("service.tick_interval_seconds must be > 0")
        if self.idle_sleep_seconds <= 0:
            raise ValueError("service.idle_sleep_seconds must be > 0")
        if self.poll_interval_seconds <= 0:
            raise ValueError("service.poll_interval_seconds must be > 0")
        if self.max_pending_jobs <= 0:
            raise ValueError("service.max_pending_jobs must be > 0")
        if self.max_consecutive_failures <= 0:
            raise ValueError("service.max_consecutive_failures must be > 0")
        if self.cooldown_seconds <= 0:
            raise ValueError("service.cooldown_seconds must be > 0")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("service.heartbeat_interval_seconds must be > 0")
        if not self.lock_name.strip():
            raise ValueError("service.lock_name must not be empty")
        if self.lock_lease_seconds <= 0:
            raise ValueError("service.lock_lease_seconds must be > 0")
        if self.shutdown_grace_period_seconds <= 0:
            raise ValueError("service.shutdown_grace_period_seconds must be > 0")
        if self.stuck_job_after_seconds <= 0:
            raise ValueError("service.stuck_job_after_seconds must be > 0")
        if self.persona_retry_interval_seconds <= 0:
            raise ValueError("service.persona_retry_interval_seconds must be > 0")
        if self.persona_email_cooldown_seconds <= 0:
            raise ValueError("service.persona_email_cooldown_seconds must be > 0")


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
    brain: BrainConfig = field(default_factory=BrainConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)

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
    adaptive_payload["mutation_mode_weights"] = MutationModeWeightsConfig(
        **adaptive_payload.get("mutation_mode_weights", {})
    )
    adaptive_payload["diversity"] = DiversityThresholdConfig(**adaptive_payload.get("diversity", {}))
    adaptive_payload["repair_policy"] = RepairPolicyConfig(**adaptive_payload.get("repair_policy", {}))
    adaptive_payload["critic_thresholds"] = CriticThresholdConfig(**adaptive_payload.get("critic_thresholds", {}))
    adaptive_payload["region_learning"] = RegionLearningConfig(**adaptive_payload.get("region_learning", {}))
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


def _build_brain_config(payload: dict[str, Any] | None) -> BrainConfig:
    return BrainConfig(**(payload or {}))


def _normalize_brain_enum(value: Any, *, true_value: str, false_value: str) -> str:
    if isinstance(value, bool):
        return true_value if value else false_value
    normalized = str(value or "").strip()
    lowered = normalized.lower()
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off"}
    if lowered in truthy:
        return true_value
    if lowered in falsy:
        return false_value
    return normalized.upper()


def _build_loop_config(payload: dict[str, Any] | None, brain: BrainConfig) -> LoopConfig:
    defaults = {
        "poll_interval_seconds": brain.poll_interval_seconds,
        "timeout_seconds": brain.timeout_seconds,
        "simulation_batch_size": brain.batch_size,
    }
    merged = dict(defaults)
    merged.update(payload or {})
    return LoopConfig(**merged)


def _build_service_config(payload: dict[str, Any] | None, brain: BrainConfig) -> ServiceConfig:
    defaults = {
        "poll_interval_seconds": brain.poll_interval_seconds,
        "max_pending_jobs": brain.batch_size,
    }
    merged = dict(defaults)
    merged.update(payload or {})
    return ServiceConfig(**merged)


def load_config(path: str | Path) -> AppConfig:
    """Load and normalize YAML config, accepting both legacy and grouped evaluation schemas."""
    config_path = Path(path)
    payload = _read_yaml(config_path)
    try:
        backtest = BacktestConfig(**payload["backtest"])
        evaluation = _build_evaluation_config(payload["evaluation"])
        brain = _build_brain_config(payload.get("brain"))
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
            brain=brain,
            loop=_build_loop_config(payload.get("loop"), brain),
            service=_build_service_config(payload.get("service"), brain),
            runtime=_build_runtime_config(payload.get("runtime"), config_path),
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config section: {exc.args[0]}") from exc
