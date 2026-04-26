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
    allow_catalog_fields_without_runtime: bool = False
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
    engine_validation_cache_enabled: bool = True
    sim_neutralization: str = "none"
    sim_decay: int = 0


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
    max_lineage_branch_fraction: float = 0.50
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
class DuplicateConfig:
    enabled: bool = True
    exact_match_enabled: bool = True
    structural_match_enabled: bool = True
    cross_run_enabled: bool = True
    same_run_structural_threshold: float = 0.92
    cross_run_structural_threshold: float = 0.95
    max_cross_run_references: int = 500
    same_run_structural_reference_limit: int = 5000


@dataclass(slots=True)
class CrowdingConfig:
    enabled: bool = True
    family_penalty_weight: float = 0.30
    motif_penalty_weight: float = 0.20
    operator_path_penalty_weight: float = 0.20
    lineage_penalty_weight: float = 0.15
    batch_penalty_weight: float = 0.10
    historical_penalty_weight: float = 0.25
    hard_block_penalty_threshold: float = 1.00


@dataclass(slots=True)
class PreSimSelectionWeightsConfig:
    predicted_quality: float = 0.40
    novelty: float = 0.20
    family_diversity: float = 0.15
    regime_fit: float = 0.05
    exploration_bonus: float = 0.05
    duplicate_risk: float = 0.25
    crowding_penalty: float = 0.20
    family_correlation_proxy_penalty: float = 0.10
    brain_robustness_proxy_penalty: float = 0.0
    complexity_cost: float = 0.10


@dataclass(slots=True)
class BrainRobustnessProxyConfig:
    enabled: bool = False
    lookback_rounds: int = 12
    min_support: int = 5
    sharpe_floor: float = 0.30
    fitness_floor: float = 0.10

    def __post_init__(self) -> None:
        if self.lookback_rounds <= 0:
            raise ValueError("adaptive_generation.selection.brain_robustness_proxy.lookback_rounds must be > 0")
        if self.min_support <= 0:
            raise ValueError("adaptive_generation.selection.brain_robustness_proxy.min_support must be > 0")


@dataclass(slots=True)
class PostSimSelectionWeightsConfig:
    performance_quality: float = 0.45
    robustness: float = 0.20
    regime_fit: float = 0.10
    family_diversity_bonus: float = 0.10
    turnover_margin_cost: float = 0.20
    crowding_penalty: float = 0.15
    duplicate_penalty: float = 0.10


@dataclass(slots=True)
class MutationParentSelectionWeightsConfig:
    post_sim_score: float = 0.45
    family_diversification_bonus: float = 0.20
    lineage_diversity_bonus: float = 0.15
    mutation_learnability_bonus: float = 0.20


@dataclass(slots=True)
class SelectionConfig:
    enabled: bool = True
    bias: str = "balanced_frontier"
    pre_sim: PreSimSelectionWeightsConfig = field(default_factory=PreSimSelectionWeightsConfig)
    post_sim: PostSimSelectionWeightsConfig = field(default_factory=PostSimSelectionWeightsConfig)
    mutation_parent: MutationParentSelectionWeightsConfig = field(default_factory=MutationParentSelectionWeightsConfig)
    brain_robustness_proxy: BrainRobustnessProxyConfig = field(default_factory=BrainRobustnessProxyConfig)

    def __post_init__(self) -> None:
        self.bias = str(self.bias or "balanced_frontier").strip().lower()


@dataclass(slots=True)
class RegimeDetectionConfig:
    enabled: bool = True
    short_window: int = 20
    long_window: int = 60
    min_points: int = 20
    min_confidence: float = 0.35
    low_vol_threshold: float = 0.85
    high_vol_threshold: float = 1.15
    trend_threshold: float = 0.50
    high_dispersion_threshold: float = 1.10


@dataclass(slots=True)
class MetaModelConfig:
    enabled: bool = True
    rollout_mode: str = "blend"
    target: str = "positive_outcome"
    model_type: str = "logistic_regression"
    blend_weight: float = 0.20
    min_train_rows: int = 500
    min_positive_rows: int = 50
    lookback_rounds: int = 1500
    use_cross_run_history: bool = True

    def __post_init__(self) -> None:
        self.rollout_mode = str(self.rollout_mode or "blend").strip().lower()
        self.target = str(self.target or "positive_outcome").strip().lower()
        self.model_type = str(self.model_type or "logistic_regression").strip().lower()
        if self.rollout_mode != "blend":
            raise ValueError("adaptive_generation.meta_model.rollout_mode must be 'blend'")
        if self.target != "positive_outcome":
            raise ValueError("adaptive_generation.meta_model.target must be 'positive_outcome'")
        if self.model_type != "logistic_regression":
            raise ValueError("adaptive_generation.meta_model.model_type must be 'logistic_regression'")
        if not 0.0 <= float(self.blend_weight) <= 1.0:
            raise ValueError("adaptive_generation.meta_model.blend_weight must be between 0 and 1")
        if self.min_train_rows <= 0:
            raise ValueError("adaptive_generation.meta_model.min_train_rows must be > 0")
        if self.min_positive_rows <= 0:
            raise ValueError("adaptive_generation.meta_model.min_positive_rows must be > 0")
        if self.lookback_rounds <= 0:
            raise ValueError("adaptive_generation.meta_model.lookback_rounds must be > 0")


@dataclass(slots=True)
class LearnedRegimeConfig:
    enabled: bool = True
    model_type: str = "minibatch_kmeans"
    cluster_count: int = 6
    history_window: int = 252
    feature_window: int = 63
    min_train_windows: int = 120
    confidence_floor: float = 0.35
    tsfresh_profile: str = "minimal"

    def __post_init__(self) -> None:
        self.model_type = str(self.model_type or "minibatch_kmeans").strip().lower()
        self.tsfresh_profile = str(self.tsfresh_profile or "minimal").strip().lower()
        if self.model_type != "minibatch_kmeans":
            raise ValueError("adaptive_generation.learned_regime.model_type must be 'minibatch_kmeans'")
        if self.cluster_count <= 1:
            raise ValueError("adaptive_generation.learned_regime.cluster_count must be > 1")
        if self.history_window <= 0:
            raise ValueError("adaptive_generation.learned_regime.history_window must be > 0")
        if self.feature_window <= 1:
            raise ValueError("adaptive_generation.learned_regime.feature_window must be > 1")
        if self.history_window < self.feature_window:
            raise ValueError(
                "adaptive_generation.learned_regime.history_window must be >= feature_window"
            )
        if self.min_train_windows <= 0:
            raise ValueError("adaptive_generation.learned_regime.min_train_windows must be > 0")
        if not 0.0 <= float(self.confidence_floor) <= 1.0:
            raise ValueError("adaptive_generation.learned_regime.confidence_floor must be between 0 and 1")
        if self.tsfresh_profile != "minimal":
            raise ValueError("adaptive_generation.learned_regime.tsfresh_profile must be 'minimal'")


@dataclass(slots=True)
class MutationLearningConfig:
    enabled: bool = True
    min_support: int = 3
    score_decay: float = 0.98
    negative_lift_penalty: float = 0.50
    success_rate_weight: float = 0.60
    uplift_weight: float = 0.40


@dataclass(slots=True)
class LocalValidationFieldPenaltyConfig:
    enabled: bool = True
    lookback_rounds: int = 20
    min_count: int = 2
    max_fields: int = 200
    penalty_strength: float = 1.0
    min_multiplier: float = 0.10
    sample_limit: int = 10

    def __post_init__(self) -> None:
        if self.lookback_rounds <= 0:
            raise ValueError("adaptive_generation.local_validation_field_penalty.lookback_rounds must be > 0")
        if self.min_count <= 0:
            raise ValueError("adaptive_generation.local_validation_field_penalty.min_count must be > 0")
        if self.max_fields <= 0:
            raise ValueError("adaptive_generation.local_validation_field_penalty.max_fields must be > 0")
        if self.penalty_strength < 0:
            raise ValueError("adaptive_generation.local_validation_field_penalty.penalty_strength must be >= 0")
        if not 0.0 < float(self.min_multiplier) <= 1.0:
            raise ValueError(
                "adaptive_generation.local_validation_field_penalty.min_multiplier must be between 0 and 1"
            )
        if self.sample_limit < 0:
            raise ValueError("adaptive_generation.local_validation_field_penalty.sample_limit must be >= 0")


@dataclass(slots=True)
class RecipeGenerationConfig:
    enabled: bool = True
    recipe_budget_fraction: float = 0.20
    max_recipe_candidates_per_round: int = 24
    active_bucket_count: int = 4
    max_candidates_per_bucket: int = 6
    max_field_candidates_per_side: int = 8
    max_pair_candidates_per_bucket: int = 12
    max_drafts_per_bucket: int = 64
    duplicate_retry_multiplier: int = 4
    enable_field_rotation: bool = True
    field_rotation_lookback_rounds: int = 12
    yield_lookback_rounds: int = 12
    lookback_completed_results: int = 800
    selection_prior_weight: float = 0.08
    min_bucket_support_for_penalty: int = 5
    dynamic_budget_enabled: bool = True
    dynamic_budget_min_generated_support: int = 10
    dynamic_budget_min_completed_support: int = 3
    source_exploration_floor_fractions: dict[str, float] = field(
        default_factory=lambda: {
            "quality_polish": 0.10,
            "recipe_guided": 0.10,
            "fresh": 0.30,
        }
    )
    source_reallocation_strength: float = 0.65
    max_fresh_budget_fraction: float = 1.0
    fresh_spillover_fraction: float = 1.0
    bucket_exploration_floor: int = 1
    bucket_reallocation_strength: float = 0.75
    bucket_biases: dict[str, float] = field(
        default_factory=lambda: {
            "revision_surprise|fundamental|balanced": 1.35,
            "revision_surprise|fundamental|quality": 1.25,
            "fundamental_quality|fundamental|balanced": 1.15,
            "fundamental_quality|fundamental|quality": 0.70,
        }
    )
    enabled_recipe_families: list[str] = field(
        default_factory=lambda: [
            "fundamental_quality",
            "accrual_vs_cashflow",
            "value_vs_growth",
            "revision_surprise",
        ]
    )
    objective_profiles: list[str] = field(
        default_factory=lambda: ["balanced", "quality", "low_turnover"]
    )

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.recipe_budget_fraction) <= 1.0:
            raise ValueError("adaptive_generation.recipe_generation.recipe_budget_fraction must be between 0 and 1")
        if self.max_recipe_candidates_per_round < 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.max_recipe_candidates_per_round must be >= 0"
            )
        if self.active_bucket_count <= 0:
            raise ValueError("adaptive_generation.recipe_generation.active_bucket_count must be > 0")
        if self.max_candidates_per_bucket <= 0:
            raise ValueError("adaptive_generation.recipe_generation.max_candidates_per_bucket must be > 0")
        if self.max_field_candidates_per_side <= 0:
            raise ValueError("adaptive_generation.recipe_generation.max_field_candidates_per_side must be > 0")
        if self.max_pair_candidates_per_bucket <= 0:
            raise ValueError("adaptive_generation.recipe_generation.max_pair_candidates_per_bucket must be > 0")
        if self.max_drafts_per_bucket <= 0:
            raise ValueError("adaptive_generation.recipe_generation.max_drafts_per_bucket must be > 0")
        if self.duplicate_retry_multiplier <= 0:
            raise ValueError("adaptive_generation.recipe_generation.duplicate_retry_multiplier must be > 0")
        if self.field_rotation_lookback_rounds <= 0:
            raise ValueError("adaptive_generation.recipe_generation.field_rotation_lookback_rounds must be > 0")
        if self.yield_lookback_rounds <= 0:
            raise ValueError("adaptive_generation.recipe_generation.yield_lookback_rounds must be > 0")
        if self.lookback_completed_results <= 0:
            raise ValueError("adaptive_generation.recipe_generation.lookback_completed_results must be > 0")
        if self.selection_prior_weight < 0:
            raise ValueError("adaptive_generation.recipe_generation.selection_prior_weight must be >= 0")
        if self.min_bucket_support_for_penalty <= 0:
            raise ValueError("adaptive_generation.recipe_generation.min_bucket_support_for_penalty must be > 0")
        if self.dynamic_budget_min_generated_support <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.dynamic_budget_min_generated_support must be > 0"
            )
        if self.dynamic_budget_min_completed_support <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.dynamic_budget_min_completed_support must be > 0"
            )
        if not 0.0 <= float(self.source_reallocation_strength) <= 1.0:
            raise ValueError(
                "adaptive_generation.recipe_generation.source_reallocation_strength must be between 0 and 1"
            )
        if not 0.0 <= float(self.max_fresh_budget_fraction) <= 1.0:
            raise ValueError(
                "adaptive_generation.recipe_generation.max_fresh_budget_fraction must be between 0 and 1"
            )
        if not 0.0 <= float(self.fresh_spillover_fraction) <= 1.0:
            raise ValueError(
                "adaptive_generation.recipe_generation.fresh_spillover_fraction must be between 0 and 1"
            )
        if self.bucket_exploration_floor < 0:
            raise ValueError("adaptive_generation.recipe_generation.bucket_exploration_floor must be >= 0")
        if not 0.0 <= float(self.bucket_reallocation_strength) <= 1.0:
            raise ValueError(
                "adaptive_generation.recipe_generation.bucket_reallocation_strength must be between 0 and 1"
            )
        normalized_floor_fractions: dict[str, float] = {}
        for key, value in dict(self.source_exploration_floor_fractions or {}).items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            normalized_value = float(value)
            if not 0.0 <= normalized_value <= 1.0:
                raise ValueError(
                    "adaptive_generation.recipe_generation.source_exploration_floor_fractions values must be between 0 and 1"
                )
            normalized_floor_fractions[normalized_key] = normalized_value
        self.source_exploration_floor_fractions = normalized_floor_fractions
        normalized_bucket_biases: dict[str, float] = {}
        for key, value in dict(self.bucket_biases or {}).items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            normalized_value = float(value)
            if normalized_value <= 0.0:
                raise ValueError("adaptive_generation.recipe_generation.bucket_biases values must be > 0")
            normalized_bucket_biases[normalized_key] = normalized_value
        self.bucket_biases = normalized_bucket_biases
        self.enabled_recipe_families = [
            str(item).strip()
            for item in (self.enabled_recipe_families or [])
            if str(item).strip()
        ]
        self.objective_profiles = [
            str(item).strip()
            for item in (self.objective_profiles or [])
            if str(item).strip()
        ]
        if not self.enabled_recipe_families:
            raise ValueError("adaptive_generation.recipe_generation.enabled_recipe_families must not be empty")
        if not self.objective_profiles:
            raise ValueError("adaptive_generation.recipe_generation.objective_profiles must not be empty")


@dataclass(slots=True)
class QualityOptimizationConfig:
    enabled: bool = True
    lookback_completed_results: int = 800
    polish_budget_fraction: float = 0.35
    max_polish_candidates_per_round: int = 64
    max_polish_parents_per_round: int = 8
    variants_per_parent: int = 8
    min_parent_fitness: float = 0.02
    min_parent_sharpe: float = 0.03
    max_parent_turnover: float = 1.00
    max_parent_drawdown: float = 0.75
    min_completed_parent_count: int = 5
    selection_prior_weight: float = 0.10
    enabled_transforms: list[str] = field(
        default_factory=lambda: ["wrap_rank", "wrap_zscore", "window_perturb"]
    )
    primary_transform: str = "wrap_rank"
    max_variants_per_parent_by_transform: dict[str, int] = field(
        default_factory=lambda: {"wrap_rank": 1, "wrap_zscore": 1, "window_perturb": 4}
    )
    disabled_transforms: list[str] = field(
        default_factory=lambda: [
            "cleanup_redundant_wrapper",
            "smooth_ts_mean",
            "smooth_ts_decay_linear",
            "smooth_ts_rank",
        ]
    )
    parent_transform_recent_rounds: int = 2
    max_parent_transform_uses_per_recent_window: int = 1
    transform_score_lookback_rounds: int = 4
    transform_cooldown_min_attempts: int = 3
    transform_cooldown_success_rate_floor: float = 0.20
    window_perturb_neighbor_count: int = 4

    def __post_init__(self) -> None:
        if self.lookback_completed_results <= 0:
            raise ValueError("adaptive_generation.quality_optimization.lookback_completed_results must be > 0")
        if not 0.0 <= float(self.polish_budget_fraction) <= 1.0:
            raise ValueError("adaptive_generation.quality_optimization.polish_budget_fraction must be between 0 and 1")
        if self.max_polish_candidates_per_round < 0:
            raise ValueError("adaptive_generation.quality_optimization.max_polish_candidates_per_round must be >= 0")
        if self.max_polish_parents_per_round <= 0:
            raise ValueError("adaptive_generation.quality_optimization.max_polish_parents_per_round must be > 0")
        if self.variants_per_parent <= 0:
            raise ValueError("adaptive_generation.quality_optimization.variants_per_parent must be > 0")
        if self.min_completed_parent_count < 0:
            raise ValueError("adaptive_generation.quality_optimization.min_completed_parent_count must be >= 0")
        if self.max_parent_turnover < 0:
            raise ValueError("adaptive_generation.quality_optimization.max_parent_turnover must be >= 0")
        if self.max_parent_drawdown < 0:
            raise ValueError("adaptive_generation.quality_optimization.max_parent_drawdown must be >= 0")
        if self.selection_prior_weight < 0:
            raise ValueError("adaptive_generation.quality_optimization.selection_prior_weight must be >= 0")
        if self.parent_transform_recent_rounds < 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.parent_transform_recent_rounds must be >= 0"
            )
        if self.max_parent_transform_uses_per_recent_window <= 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.max_parent_transform_uses_per_recent_window must be > 0"
            )
        if self.transform_score_lookback_rounds < 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.transform_score_lookback_rounds must be >= 0"
            )
        if self.transform_cooldown_min_attempts <= 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.transform_cooldown_min_attempts must be > 0"
            )
        if not 0.0 <= float(self.transform_cooldown_success_rate_floor) <= 1.0:
            raise ValueError(
                "adaptive_generation.quality_optimization.transform_cooldown_success_rate_floor must be between 0 and 1"
            )
        if self.window_perturb_neighbor_count <= 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.window_perturb_neighbor_count must be > 0"
            )
        self.enabled_transforms = [
            str(item).strip()
            for item in (self.enabled_transforms or [])
            if str(item).strip()
        ]
        self.primary_transform = str(self.primary_transform or "wrap_rank").strip()
        self.max_variants_per_parent_by_transform = {
            str(key).strip(): int(value)
            for key, value in dict(self.max_variants_per_parent_by_transform or {}).items()
            if str(key).strip() and int(value) > 0
        }
        self.disabled_transforms = [
            str(item).strip()
            for item in (self.disabled_transforms or [])
            if str(item).strip()
        ]


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
    duplicate: DuplicateConfig = field(default_factory=DuplicateConfig)
    crowding: CrowdingConfig = field(default_factory=CrowdingConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    regime_detection: RegimeDetectionConfig = field(default_factory=RegimeDetectionConfig)
    meta_model: MetaModelConfig = field(default_factory=MetaModelConfig)
    learned_regime: LearnedRegimeConfig = field(default_factory=LearnedRegimeConfig)
    mutation_learning: MutationLearningConfig = field(default_factory=MutationLearningConfig)
    local_validation_field_penalty: LocalValidationFieldPenaltyConfig = field(
        default_factory=LocalValidationFieldPenaltyConfig
    )
    recipe_generation: RecipeGenerationConfig = field(default_factory=RecipeGenerationConfig)
    quality_optimization: QualityOptimizationConfig = field(default_factory=QualityOptimizationConfig)
    max_generation_seconds: float = 20.0
    max_attempt_multiplier: int = 12
    exploit_budget_ratio: float = 0.60
    explore_budget_ratio: float = 0.40
    min_explore_attempts: int = 150
    min_explore_seconds: float = 2.0
    max_consecutive_failures: int = 400
    explore_max_consecutive_failures: int | None = None
    min_candidates_before_early_exit: int = 5


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
class SimulationProfile:
    name: str
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    neutralization: str = "SUBINDUSTRY"
    decay: int = 3
    truncation: float = 0.01
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.name = str(self.name or "").strip()
        if not self.name:
            raise ValueError("brain.simulation_profiles[].name must not be empty")
        self.region = str(self.region).strip().upper()
        self.universe = str(self.universe).strip().upper()
        self.neutralization = _normalize_brain_enum(self.neutralization, true_value="ON", false_value="OFF")
        if self.delay < 0:
            raise ValueError("brain.simulation_profiles[].delay must be >= 0")
        if self.decay < 0:
            raise ValueError("brain.simulation_profiles[].decay must be >= 0")
        if self.truncation < 0:
            raise ValueError("brain.simulation_profiles[].truncation must be >= 0")
        if self.weight < 0:
            raise ValueError("brain.simulation_profiles[].weight must be >= 0")


def _default_simulation_profiles() -> list[SimulationProfile]:
    return [
        SimulationProfile(
            name="stable",
            region="USA",
            universe="TOP1000",
            delay=1,
            neutralization="SUBINDUSTRY",
            decay=3,
            truncation=0.01,
            weight=0.6,
        ),
        SimulationProfile(
            name="aggressive_short",
            region="USA",
            universe="TOP500",
            delay=1,
            neutralization="SUBINDUSTRY",
            decay=1,
            truncation=0.02,
            weight=0.4,
        ),
    ]


@dataclass(slots=True)
class BrainConfig:
    backend: str = "manual"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    neutralization: str = "SUBINDUSTRY"
    decay: int = 3
    truncation: float = 0.01
    simulation_profiles: list[SimulationProfile] = field(default_factory=_default_simulation_profiles)
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
        self.simulation_profiles = [
            item
            if isinstance(item, SimulationProfile)
            else SimulationProfile(**item)
            for item in self.simulation_profiles
        ]
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
    persona_confirmation_required: bool = True
    persona_confirmation_poll_interval_seconds: int = 30
    persona_confirmation_prompt_cooldown_seconds: int = 1800
    persona_confirmation_granted_ttl_seconds: int = 300
    max_persona_wait_seconds: int = 1800
    max_consecutive_batch_failures_before_auth_check: int = 2
    ambiguous_submission_policy: str = "fail"
    research_context_cache_enabled: bool = True
    research_context_cache_ttl_seconds: int = 0
    observed_limit_ttl_seconds: int = 1800
    observed_limit_probe_interval_seconds: int = 300

    def __post_init__(self) -> None:
        self.ambiguous_submission_policy = str(self.ambiguous_submission_policy or "fail").strip().lower()
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
        if self.persona_confirmation_poll_interval_seconds <= 0:
            raise ValueError("service.persona_confirmation_poll_interval_seconds must be > 0")
        if self.persona_confirmation_prompt_cooldown_seconds <= 0:
            raise ValueError("service.persona_confirmation_prompt_cooldown_seconds must be > 0")
        if self.persona_confirmation_granted_ttl_seconds <= 0:
            raise ValueError("service.persona_confirmation_granted_ttl_seconds must be > 0")
        if self.max_persona_wait_seconds <= 0:
            raise ValueError("service.max_persona_wait_seconds must be > 0")
        if self.max_consecutive_batch_failures_before_auth_check <= 0:
            raise ValueError("service.max_consecutive_batch_failures_before_auth_check must be > 0")
        if self.ambiguous_submission_policy not in {"quarantine", "fail", "resubmit"}:
            raise ValueError(
                "service.ambiguous_submission_policy must be one of: quarantine, fail, resubmit"
            )
        if self.research_context_cache_ttl_seconds < 0:
            raise ValueError("service.research_context_cache_ttl_seconds must be >= 0")
        if self.observed_limit_ttl_seconds <= 0:
            raise ValueError("service.observed_limit_ttl_seconds must be > 0")
        if self.observed_limit_probe_interval_seconds <= 0:
            raise ValueError("service.observed_limit_probe_interval_seconds must be > 0")


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
    adaptive_payload["duplicate"] = DuplicateConfig(**adaptive_payload.get("duplicate", {}))
    adaptive_payload["crowding"] = CrowdingConfig(**adaptive_payload.get("crowding", {}))
    selection_payload = dict(adaptive_payload.get("selection", {}) or {})
    adaptive_payload["selection"] = SelectionConfig(
        **{
            **selection_payload,
            "pre_sim": PreSimSelectionWeightsConfig(
                **dict(selection_payload.get("pre_sim", {}) or {})
            ),
            "post_sim": PostSimSelectionWeightsConfig(
                **dict(selection_payload.get("post_sim", {}) or {})
            ),
            "mutation_parent": MutationParentSelectionWeightsConfig(
                **dict(selection_payload.get("mutation_parent", {}) or {})
            ),
            "brain_robustness_proxy": BrainRobustnessProxyConfig(
                **dict(selection_payload.get("brain_robustness_proxy", {}) or {})
            ),
        }
    )
    adaptive_payload["regime_detection"] = RegimeDetectionConfig(
        **adaptive_payload.get("regime_detection", {})
    )
    adaptive_payload["meta_model"] = MetaModelConfig(**adaptive_payload.get("meta_model", {}))
    adaptive_payload["learned_regime"] = LearnedRegimeConfig(
        **adaptive_payload.get("learned_regime", {})
    )
    adaptive_payload["mutation_learning"] = MutationLearningConfig(
        **adaptive_payload.get("mutation_learning", {})
    )
    adaptive_payload["local_validation_field_penalty"] = LocalValidationFieldPenaltyConfig(
        **adaptive_payload.get("local_validation_field_penalty", {})
    )
    adaptive_payload["recipe_generation"] = RecipeGenerationConfig(
        **adaptive_payload.get("recipe_generation", {})
    )
    adaptive_payload["quality_optimization"] = QualityOptimizationConfig(
        **adaptive_payload.get("quality_optimization", {})
    )
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
    if payload is None:
        return BrainConfig()
    brain_payload = dict(payload)
    profiles_payload = brain_payload.get("simulation_profiles")
    if profiles_payload is None:
        brain_payload["simulation_profiles"] = []
    else:
        brain_payload["simulation_profiles"] = [
            item if isinstance(item, SimulationProfile) else SimulationProfile(**item)
            for item in profiles_payload
        ]
    return BrainConfig(**brain_payload)


def _resolve_generation_simulation_defaults(brain: BrainConfig) -> tuple[str, int]:
    if brain.simulation_profiles:
        reference = brain.simulation_profiles[0]
        return reference.neutralization, int(reference.decay)
    return brain.neutralization, int(brain.decay)


def _apply_generation_simulation_defaults(
    generation: GenerationConfig,
    generation_payload: dict[str, Any],
    brain: BrainConfig,
) -> GenerationConfig:
    neutralization, decay = _resolve_generation_simulation_defaults(brain)
    if "sim_neutralization" not in generation_payload:
        generation.sim_neutralization = str(neutralization)
    if "sim_decay" not in generation_payload:
        generation.sim_decay = int(decay)
    return generation


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
