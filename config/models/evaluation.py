from __future__ import annotations

from dataclasses import dataclass, field


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
    data_requirements: EvaluationDataRequirementsConfig = field(
        default_factory=EvaluationDataRequirementsConfig
    )
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
            signal_correlation_threshold=(
                0.95
                if signal_correlation_threshold is None
                else float(signal_correlation_threshold)
            ),
            returns_correlation_threshold=(
                0.95
                if returns_correlation_threshold is None
                else float(returns_correlation_threshold)
            ),
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


__all__ = [
    "SubuniverseConfig",
    "SimulationConfig",
    "BacktestConfig",
    "EvaluationHardFiltersConfig",
    "EvaluationDataRequirementsConfig",
    "EvaluationDiversityConfig",
    "EvaluationRankingConfig",
    "EvaluationRobustnessConfig",
    "EvaluationConfig",
    "SubmissionTestConfig",
]
