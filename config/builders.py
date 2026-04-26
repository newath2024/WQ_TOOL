from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from config.models.adaptive import (
    AdaptiveGenerationConfig,
    BrainRobustnessProxyConfig,
    CriticThresholdConfig,
    CrowdingConfig,
    DiversityThresholdConfig,
    DuplicateConfig,
    EliteMotifConfig,
    LearnedRegimeConfig,
    LocalValidationFieldPenaltyConfig,
    MetaModelConfig,
    MutationLearningConfig,
    MutationModeWeightsConfig,
    MutationParentSelectionWeightsConfig,
    PostSimSelectionWeightsConfig,
    PreSimSelectionWeightsConfig,
    QualityOptimizationConfig,
    RecipeGenerationConfig,
    RegionLearningConfig,
    RegimeDetectionConfig,
    RepairPolicyConfig,
    SelectionConfig,
    StrategyMixConfig,
)
from config.models.brain import BrainConfig, SimulationProfile
from config.models.evaluation import (
    BacktestConfig,
    EvaluationConfig,
    EvaluationDataRequirementsConfig,
    EvaluationDiversityConfig,
    EvaluationHardFiltersConfig,
    EvaluationRankingConfig,
    EvaluationRobustnessConfig,
    SimulationConfig,
    SubmissionTestConfig,
    SubuniverseConfig,
)
from config.models.generation import GenerationConfig
from config.models.runtime import LoopConfig, PeriodConfig, RuntimeConfig
from config.models.service import ServiceConfig


def _period_from_mapping(payload: dict[str, Any], key: str) -> PeriodConfig:
    try:
        period_payload = payload[key]
        return PeriodConfig(start=str(period_payload["start"]), end=str(period_payload["end"]))
    except KeyError as exc:
        raise ValueError(f"Missing split definition for '{key}'.") from exc


def _build_subuniverses(payload: list[dict[str, Any]] | None) -> list[SubuniverseConfig]:
    return [SubuniverseConfig(**item) for item in (payload or [])]


def _build_simulation_config(
    payload: dict[str, Any] | None, backtest: BacktestConfig
) -> SimulationConfig:
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
    adaptive_payload["diversity"] = DiversityThresholdConfig(
        **adaptive_payload.get("diversity", {})
    )
    adaptive_payload["repair_policy"] = RepairPolicyConfig(
        **adaptive_payload.get("repair_policy", {})
    )
    adaptive_payload["critic_thresholds"] = CriticThresholdConfig(
        **adaptive_payload.get("critic_thresholds", {})
    )
    adaptive_payload["region_learning"] = RegionLearningConfig(
        **adaptive_payload.get("region_learning", {})
    )
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
    adaptive_payload["elite_motifs"] = EliteMotifConfig(
        **adaptive_payload.get("elite_motifs", {})
    )
    return AdaptiveGenerationConfig(**adaptive_payload)


def _build_evaluation_config(payload: dict[str, Any]) -> EvaluationConfig:
    if any(
        key in payload
        for key in ("hard_filters", "data_requirements", "diversity", "ranking", "robustness")
    ):
        return EvaluationConfig(
            hard_filters=EvaluationHardFiltersConfig(**payload.get("hard_filters", {})),
            data_requirements=EvaluationDataRequirementsConfig(
                **payload.get("data_requirements", {})
            ),
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


__all__ = [
    "_period_from_mapping",
    "_build_subuniverses",
    "_build_simulation_config",
    "_build_adaptive_generation_config",
    "_build_evaluation_config",
    "_build_submission_test_config",
    "_build_runtime_config",
    "_build_brain_config",
    "_resolve_generation_simulation_defaults",
    "_apply_generation_simulation_defaults",
    "_build_loop_config",
    "_build_service_config",
]
