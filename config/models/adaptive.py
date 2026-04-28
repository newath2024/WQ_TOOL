from __future__ import annotations

from dataclasses import dataclass, field


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
        self.global_prior_scope = (
            str(self.global_prior_scope or "match_non_region_regime").strip().lower()
        )
        self.blend_mode = str(self.blend_mode or "linear_ramp").strip().lower()
        if self.min_local_pattern_samples < 0:
            raise ValueError(
                "adaptive_generation.region_learning.min_local_pattern_samples must be >= 0"
            )
        if self.full_local_pattern_samples < self.min_local_pattern_samples:
            raise ValueError(
                "adaptive_generation.region_learning.full_local_pattern_samples must be >= min_local_pattern_samples"
            )
        if self.min_local_case_samples < 0:
            raise ValueError(
                "adaptive_generation.region_learning.min_local_case_samples must be >= 0"
            )
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
    elite_motif_bonus: float = 0.0
    elite_seed_similarity_penalty: float = 0.0
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
            raise ValueError(
                "adaptive_generation.selection.brain_robustness_proxy.lookback_rounds must be > 0"
            )
        if self.min_support <= 0:
            raise ValueError(
                "adaptive_generation.selection.brain_robustness_proxy.min_support must be > 0"
            )


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
    mutation_parent: MutationParentSelectionWeightsConfig = field(
        default_factory=MutationParentSelectionWeightsConfig
    )
    brain_robustness_proxy: BrainRobustnessProxyConfig = field(
        default_factory=BrainRobustnessProxyConfig
    )

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
            raise ValueError(
                "adaptive_generation.meta_model.model_type must be 'logistic_regression'"
            )
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
            raise ValueError(
                "adaptive_generation.learned_regime.model_type must be 'minibatch_kmeans'"
            )
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
            raise ValueError(
                "adaptive_generation.learned_regime.confidence_floor must be between 0 and 1"
            )
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
            raise ValueError(
                "adaptive_generation.local_validation_field_penalty.lookback_rounds must be > 0"
            )
        if self.min_count <= 0:
            raise ValueError(
                "adaptive_generation.local_validation_field_penalty.min_count must be > 0"
            )
        if self.max_fields <= 0:
            raise ValueError(
                "adaptive_generation.local_validation_field_penalty.max_fields must be > 0"
            )
        if self.penalty_strength < 0:
            raise ValueError(
                "adaptive_generation.local_validation_field_penalty.penalty_strength must be >= 0"
            )
        if not 0.0 < float(self.min_multiplier) <= 1.0:
            raise ValueError(
                "adaptive_generation.local_validation_field_penalty.min_multiplier must be between 0 and 1"
            )
        if self.sample_limit < 0:
            raise ValueError(
                "adaptive_generation.local_validation_field_penalty.sample_limit must be >= 0"
            )


@dataclass(slots=True)
class SearchSpaceFilterConfig:
    enabled: bool = False
    profile_mismatch_multiplier: float = 0.25
    unknown_profile_multiplier: float = 0.75
    validation_field_multiplier: float = 0.35
    validation_field_min_count: int = 2
    completed_lookback_rounds: int = 20
    min_completed_support: int = 3
    sharpe_floor: float = 0.30
    fitness_floor: float = 0.10
    field_result_multiplier: float = 0.50
    operator_result_multiplier: float = 0.60
    winner_prior_enabled: bool = False
    winner_prior_lookback_rounds: int = 20
    winner_prior_min_support: int = 2
    winner_prior_sharpe_floor: float = 0.30
    winner_prior_fitness_floor: float = 0.10
    winner_prior_strong_sharpe_floor: float = 0.50
    winner_prior_strong_fitness_floor: float = 0.30
    winner_field_multiplier: float = 1.35
    strong_winner_field_multiplier: float = 1.80
    weak_field_multiplier: float = 0.65
    winner_operator_multiplier: float = 1.35
    strong_winner_operator_multiplier: float = 1.80
    weak_operator_multiplier: float = 0.65
    lane_field_caps: dict[str, int] = field(default_factory=dict)
    lane_field_min_count: int = 0
    lane_operator_allowlists: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "profile_mismatch_multiplier",
            "unknown_profile_multiplier",
            "validation_field_multiplier",
            "field_result_multiplier",
            "operator_result_multiplier",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"adaptive_generation.search_space_filter.{name} must be between 0 and 1")
            setattr(self, name, value)
        self.validation_field_min_count = int(self.validation_field_min_count)
        self.completed_lookback_rounds = int(self.completed_lookback_rounds)
        self.min_completed_support = int(self.min_completed_support)
        self.sharpe_floor = float(self.sharpe_floor)
        self.fitness_floor = float(self.fitness_floor)
        self.winner_prior_lookback_rounds = int(self.winner_prior_lookback_rounds)
        self.winner_prior_min_support = int(self.winner_prior_min_support)
        self.winner_prior_sharpe_floor = float(self.winner_prior_sharpe_floor)
        self.winner_prior_fitness_floor = float(self.winner_prior_fitness_floor)
        self.winner_prior_strong_sharpe_floor = float(self.winner_prior_strong_sharpe_floor)
        self.winner_prior_strong_fitness_floor = float(self.winner_prior_strong_fitness_floor)
        self.lane_field_min_count = int(self.lane_field_min_count)
        if self.validation_field_min_count <= 0:
            raise ValueError(
                "adaptive_generation.search_space_filter.validation_field_min_count must be > 0"
            )
        if self.completed_lookback_rounds <= 0:
            raise ValueError(
                "adaptive_generation.search_space_filter.completed_lookback_rounds must be > 0"
            )
        if self.min_completed_support <= 0:
            raise ValueError(
                "adaptive_generation.search_space_filter.min_completed_support must be > 0"
            )
        if self.winner_prior_lookback_rounds <= 0:
            raise ValueError(
                "adaptive_generation.search_space_filter.winner_prior_lookback_rounds must be > 0"
            )
        if self.winner_prior_min_support <= 0:
            raise ValueError(
                "adaptive_generation.search_space_filter.winner_prior_min_support must be > 0"
            )
        if self.lane_field_min_count < 0:
            raise ValueError(
                "adaptive_generation.search_space_filter.lane_field_min_count must be >= 0"
            )
        for name in (
            "winner_field_multiplier",
            "strong_winner_field_multiplier",
            "weak_field_multiplier",
            "winner_operator_multiplier",
            "strong_winner_operator_multiplier",
            "weak_operator_multiplier",
        ):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"adaptive_generation.search_space_filter.{name} must be > 0")
            setattr(self, name, value)
        self.lane_field_caps = {
            str(key).strip(): max(0, int(value))
            for key, value in dict(self.lane_field_caps or {}).items()
            if str(key).strip()
        }
        normalized_allowlists: dict[str, list[str]] = {}
        for key, values in dict(self.lane_operator_allowlists or {}).items():
            lane = str(key).strip()
            if not lane:
                continue
            normalized_allowlists[lane] = list(
                dict.fromkeys(str(item).strip() for item in (values or []) if str(item).strip())
            )
        self.lane_operator_allowlists = normalized_allowlists


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
    bucket_suppression_enabled: bool = False
    bucket_suppression_min_support: int = 5
    bucket_suppression_sharpe_floor: float = 0.30
    bucket_suppression_fitness_floor: float = 0.10
    bucket_suppression_max_candidates: int = 1
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
            raise ValueError(
                "adaptive_generation.recipe_generation.recipe_budget_fraction must be between 0 and 1"
            )
        if self.max_recipe_candidates_per_round < 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.max_recipe_candidates_per_round must be >= 0"
            )
        if self.active_bucket_count <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.active_bucket_count must be > 0"
            )
        if self.max_candidates_per_bucket <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.max_candidates_per_bucket must be > 0"
            )
        if self.max_field_candidates_per_side <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.max_field_candidates_per_side must be > 0"
            )
        if self.max_pair_candidates_per_bucket <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.max_pair_candidates_per_bucket must be > 0"
            )
        if self.max_drafts_per_bucket <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.max_drafts_per_bucket must be > 0"
            )
        if self.duplicate_retry_multiplier <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.duplicate_retry_multiplier must be > 0"
            )
        if self.field_rotation_lookback_rounds <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.field_rotation_lookback_rounds must be > 0"
            )
        if self.yield_lookback_rounds <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.yield_lookback_rounds must be > 0"
            )
        if self.lookback_completed_results <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.lookback_completed_results must be > 0"
            )
        if self.selection_prior_weight < 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.selection_prior_weight must be >= 0"
            )
        if self.min_bucket_support_for_penalty <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.min_bucket_support_for_penalty must be > 0"
            )
        if self.dynamic_budget_min_generated_support <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.dynamic_budget_min_generated_support must be > 0"
            )
        if self.dynamic_budget_min_completed_support <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.dynamic_budget_min_completed_support must be > 0"
            )
        if self.bucket_suppression_min_support <= 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.bucket_suppression_min_support must be > 0"
            )
        if self.bucket_suppression_max_candidates < 0:
            raise ValueError(
                "adaptive_generation.recipe_generation.bucket_suppression_max_candidates must be >= 0"
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
            raise ValueError(
                "adaptive_generation.recipe_generation.bucket_exploration_floor must be >= 0"
            )
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
                raise ValueError(
                    "adaptive_generation.recipe_generation.bucket_biases values must be > 0"
                )
            normalized_bucket_biases[normalized_key] = normalized_value
        self.bucket_biases = normalized_bucket_biases
        self.enabled_recipe_families = [
            str(item).strip() for item in (self.enabled_recipe_families or []) if str(item).strip()
        ]
        self.objective_profiles = [
            str(item).strip() for item in (self.objective_profiles or []) if str(item).strip()
        ]
        if not self.enabled_recipe_families:
            raise ValueError(
                "adaptive_generation.recipe_generation.enabled_recipe_families must not be empty"
            )
        if not self.objective_profiles:
            raise ValueError(
                "adaptive_generation.recipe_generation.objective_profiles must not be empty"
            )


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
    parent_scan_multiplier: int = 1
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
    cooldown_exempt_transform_groups: list[str] = field(default_factory=list)
    window_perturb_neighbor_count: int = 4

    def __post_init__(self) -> None:
        if self.lookback_completed_results <= 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.lookback_completed_results must be > 0"
            )
        if not 0.0 <= float(self.polish_budget_fraction) <= 1.0:
            raise ValueError(
                "adaptive_generation.quality_optimization.polish_budget_fraction must be between 0 and 1"
            )
        if self.max_polish_candidates_per_round < 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.max_polish_candidates_per_round must be >= 0"
            )
        if self.max_polish_parents_per_round <= 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.max_polish_parents_per_round must be > 0"
            )
        if self.variants_per_parent <= 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.variants_per_parent must be > 0"
            )
        if self.min_completed_parent_count < 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.min_completed_parent_count must be >= 0"
            )
        if self.max_parent_turnover < 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.max_parent_turnover must be >= 0"
            )
        if self.max_parent_drawdown < 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.max_parent_drawdown must be >= 0"
            )
        if self.selection_prior_weight < 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.selection_prior_weight must be >= 0"
            )
        if self.parent_scan_multiplier <= 0:
            raise ValueError(
                "adaptive_generation.quality_optimization.parent_scan_multiplier must be > 0"
            )
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
            str(item).strip() for item in (self.enabled_transforms or []) if str(item).strip()
        ]
        self.primary_transform = str(self.primary_transform or "wrap_rank").strip()
        self.max_variants_per_parent_by_transform = {
            str(key).strip(): int(value)
            for key, value in dict(self.max_variants_per_parent_by_transform or {}).items()
            if str(key).strip() and int(value) > 0
        }
        self.disabled_transforms = [
            str(item).strip() for item in (self.disabled_transforms or []) if str(item).strip()
        ]
        self.cooldown_exempt_transform_groups = [
            str(item).strip()
            for item in (self.cooldown_exempt_transform_groups or [])
            if str(item).strip()
        ]


@dataclass(slots=True)
class EliteMotifConfig:
    enabled: bool = False
    lookbacks: list[int] = field(default_factory=lambda: [125, 145, 150])
    seed_expressions: list[str] = field(default_factory=list)
    clone_similarity_threshold: float = 0.70
    max_quality_polish_seeds_per_round: int = 6
    max_seed_variants_per_seed: int = 4

    def __post_init__(self) -> None:
        self.lookbacks = list(dict.fromkeys(int(value) for value in (self.lookbacks or []) if int(value) > 0))
        if not self.lookbacks:
            self.lookbacks = [125, 145, 150]
        self.seed_expressions = [
            str(item).strip() for item in (self.seed_expressions or []) if str(item).strip()
        ]
        if not 0.0 <= float(self.clone_similarity_threshold) <= 1.0:
            raise ValueError(
                "adaptive_generation.elite_motifs.clone_similarity_threshold must be between 0 and 1"
            )
        if self.max_quality_polish_seeds_per_round < 0:
            raise ValueError(
                "adaptive_generation.elite_motifs.max_quality_polish_seeds_per_round must be >= 0"
            )
        if self.max_seed_variants_per_seed < 0:
            raise ValueError(
                "adaptive_generation.elite_motifs.max_seed_variants_per_seed must be >= 0"
            )


@dataclass(slots=True)
class AdaptiveGenerationConfig:
    enabled: bool = True
    memory_scope: str = "regime"
    success_rule: str = "validation_first"
    strategy_mix: StrategyMixConfig = field(default_factory=StrategyMixConfig)
    exploration_ratio: float = 0.35
    novelty_weight: float = 0.25
    mutation_mode_weights: MutationModeWeightsConfig = field(
        default_factory=MutationModeWeightsConfig
    )
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
    search_space_filter: SearchSpaceFilterConfig = field(default_factory=SearchSpaceFilterConfig)
    recipe_generation: RecipeGenerationConfig = field(default_factory=RecipeGenerationConfig)
    quality_optimization: QualityOptimizationConfig = field(
        default_factory=QualityOptimizationConfig
    )
    elite_motifs: EliteMotifConfig = field(default_factory=EliteMotifConfig)
    max_generation_seconds: float = 20.0
    max_attempt_multiplier: int = 12
    exploit_budget_ratio: float = 0.60
    explore_budget_ratio: float = 0.40
    min_explore_attempts: int = 150
    min_explore_seconds: float = 2.0
    max_consecutive_failures: int = 400
    explore_max_consecutive_failures: int | None = None
    min_candidates_before_early_exit: int = 5


__all__ = [
    "StrategyMixConfig",
    "MutationModeWeightsConfig",
    "DiversityThresholdConfig",
    "RepairPolicyConfig",
    "CriticThresholdConfig",
    "RegionLearningConfig",
    "DuplicateConfig",
    "CrowdingConfig",
    "PreSimSelectionWeightsConfig",
    "BrainRobustnessProxyConfig",
    "PostSimSelectionWeightsConfig",
    "MutationParentSelectionWeightsConfig",
    "SelectionConfig",
    "RegimeDetectionConfig",
    "MetaModelConfig",
    "LearnedRegimeConfig",
    "MutationLearningConfig",
    "LocalValidationFieldPenaltyConfig",
    "SearchSpaceFilterConfig",
    "RecipeGenerationConfig",
    "QualityOptimizationConfig",
    "EliteMotifConfig",
    "AdaptiveGenerationConfig",
]
