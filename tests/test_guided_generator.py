from __future__ import annotations

import random

from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import build_registry
from memory.case_memory import CaseAggregate, CaseMemoryRecord, CaseMemorySnapshot, ObjectiveVector
from memory.pattern_memory import BlendDiagnostics
from generator.guided_generator import GuidedGenerator
from generator.mutation_policy import MutationPolicy
from memory.pattern_memory import MemoryParent, PatternMemoryService, PatternMemorySnapshot


def build_generation_config() -> GenerationConfig:
    return GenerationConfig(
        allowed_fields=["open", "high", "low", "close", "volume", "returns"],
        allowed_operators=[
            "delay",
            "delta",
            "returns",
            "rank",
            "zscore",
            "correlation",
            "covariance",
            "decay_linear",
            "ts_rank",
            "ts_sum",
            "ts_mean",
            "ts_std",
            "sign",
            "abs",
            "log",
            "clip",
        ],
        lookbacks=[2, 3, 5, 10],
        max_depth=5,
        complexity_limit=20,
        template_count=8,
        grammar_count=8,
        mutation_count=4,
        normalization_wrappers=["rank", "zscore", "sign"],
        random_seed=11,
    )


def build_parent(
    *,
    alpha_id: str,
    expression: str,
    mutation_hints: tuple[str, ...],
    fail_tags: tuple[str, ...] = (),
) -> MemoryParent:
    service = PatternMemoryService()
    signature = service.extract_signature(expression)
    return MemoryParent(
        run_id="run-1",
        alpha_id=alpha_id,
        expression=expression,
        normalized_expression=expression,
        generation_mode="guided_mutation",
        generation_metadata={},
        parent_refs=(),
        family_signature=signature.family_signature,
        outcome_score=0.9,
        behavioral_novelty_score=0.8,
        fail_tags=fail_tags,
        success_tags=("passed_validation_filters",),
        mutation_hints=mutation_hints,
    )


def build_field_registry() -> FieldRegistry:
    return FieldRegistry(
        fields={
            name: FieldSpec(
                name=name,
                dataset="test",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=10,
                category="price" if name != "volume" else "volume",
                runtime_available=True,
                field_score=0.8,
                category_weight=0.8,
            )
            for name in ["open", "high", "low", "close", "volume", "returns"]
        }
    )


def test_guided_generator_allocates_strategy_mix_exactly() -> None:
    generator = GuidedGenerator(
        generation_config=build_generation_config(),
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(build_generation_config().allowed_operators),
        memory_service=PatternMemoryService(),
        field_registry=build_field_registry(),
    )

    assert generator._allocate_counts(20) == {
        "guided_mutation": 8,
        "memory_templates": 6,
        "random_exploration": 4,
        "novelty_search": 2,
    }


def test_guided_generator_emits_genome_metadata_and_full_count() -> None:
    config = build_generation_config()
    generator = GuidedGenerator(
        generation_config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(config.allowed_operators),
        memory_service=PatternMemoryService(),
        field_registry=build_field_registry(),
    )

    candidates = generator.generate(
        count=10,
        snapshot=PatternMemorySnapshot(regime_key="regime-cold", patterns={}),
    )

    assert len(candidates) == 10
    assert {candidate.generation_mode for candidate in candidates} <= {"guided_exploit", "guided_explore"}
    assert all(candidate.generation_metadata.get("genome") for candidate in candidates)
    assert all(candidate.template_name for candidate in candidates)


def test_mutation_policy_supports_repair_and_novelty_modes() -> None:
    config = build_generation_config()
    service = PatternMemoryService()
    policy = MutationPolicy(
        config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        memory_service=service,
        randomizer=random.Random(7),
        field_registry=build_field_registry(),
        registry=build_registry(config.allowed_operators),
    )

    turnover_parent = build_parent(
        alpha_id="parent-turnover",
        expression="rank(delta(close, 2))",
        mutation_hints=("smoothen_and_slow_down",),
        fail_tags=("high_turnover",),
    )
    simplify_parent = build_parent(
        alpha_id="parent-overfit",
        expression="zscore(rank(delta(close, 2)))",
        mutation_hints=("simplify_and_stabilize",),
        fail_tags=("excessive_complexity",),
    )

    turnover_variants = policy.generate(turnover_parent, PatternMemorySnapshot("r"), 10)
    simplify_variants = policy.generate(simplify_parent, PatternMemorySnapshot("r"), 10)

    assert any(
        ("ts_mean(" in expression) or ("decay_linear(" in expression) or (", 5)" in expression) or (", 10)" in expression)
        for expression, _ in turnover_variants
    )
    assert any(metadata.get("repair_actions") for _, metadata in simplify_variants)
    assert any(metadata.get("mutation_mode") in {"repair", "structural", "exploit_local"} for _, metadata in simplify_variants)


def test_guided_generator_prefers_global_motif_prior_when_local_region_is_cold() -> None:
    config = build_generation_config()
    generator = GuidedGenerator(
        generation_config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(config.allowed_operators),
        memory_service=PatternMemoryService(),
        field_registry=build_field_registry(),
    )
    case_snapshot = CaseMemorySnapshot(
        regime_key="usa-local",
        global_regime_key="global-shared",
        region="USA",
        global_motif_stats={
            "momentum": CaseAggregate(
                support=40,
                avg_outcome=3.0,
                avg_fitness=2.0,
                avg_sharpe=2.0,
                avg_eligibility=1.0,
                avg_robustness=1.0,
                avg_novelty=0.5,
                avg_turnover_cost=0.3,
                avg_complexity_cost=0.3,
                success_rate=0.9,
                failure_rate=0.1,
            )
        },
        global_sample_count=40,
        blend=BlendDiagnostics(
            scope="case",
            local_weight=0.0,
            global_weight=1.0,
            local_samples=0,
            global_samples=40,
            min_local_samples=10,
            full_local_samples=50,
        ),
    )

    candidates = generator.generate(
        count=30,
        snapshot=PatternMemorySnapshot(regime_key="usa-local", global_regime_key="global-shared", region="USA"),
        case_snapshot=case_snapshot,
    )

    momentum_count = sum(1 for candidate in candidates if candidate.template_name == "momentum")
    assert momentum_count >= 15


def test_mutation_policy_can_fall_back_to_global_mode_preferences_without_cross_region_parents() -> None:
    config = build_generation_config()
    service = PatternMemoryService()
    policy = MutationPolicy(
        config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        memory_service=service,
        randomizer=random.Random(11),
        field_registry=build_field_registry(),
        registry=build_registry(config.allowed_operators),
    )
    signature = service.extract_signature("rank(delta(close, 2))")
    global_case = CaseMemoryRecord(
        run_id="run-eur",
        alpha_id="alpha-eur",
        region="EUR",
        regime_key="eur-local",
        global_regime_key="shared-global",
        metric_source="external_brain",
        family_signature=signature.family_signature,
        structural_signature=signature,
        genome_hash="",
        genome=None,
        motif="momentum",
        field_families=("price",),
        operator_path=("rank", "delta"),
        complexity_bucket=signature.complexity_bucket,
        turnover_bucket=signature.turnover_bucket,
        horizon_bucket=signature.horizon_bucket,
        mutation_mode="repair",
        parent_family_signatures=(),
        fail_tags=("high_turnover",),
        success_tags=("turnover_acceptable",),
        objective_vector=ObjectiveVector(),
        outcome_score=1.5,
        created_at="2026-01-01T00:00:00+00:00",
    )
    snapshot = CaseMemorySnapshot(
        regime_key="usa-local",
        global_regime_key="shared-global",
        region="USA",
        cases=(),
        global_cases=(global_case,),
        global_sample_count=1,
        blend=BlendDiagnostics(
            scope="case",
            local_weight=0.0,
            global_weight=1.0,
            local_samples=0,
            global_samples=1,
            min_local_samples=10,
            full_local_samples=50,
        ),
    )
    parent = build_parent(
        alpha_id="parent-usa",
        expression="rank(delta(close, 2))",
        mutation_hints=("smoothen_and_slow_down",),
        fail_tags=("high_turnover",),
    )

    selected_modes = [policy._choose_mode(parent, case_snapshot=snapshot, force_novelty=False) for _ in range(50)]  # noqa: SLF001

    assert selected_modes.count("repair") >= 20
