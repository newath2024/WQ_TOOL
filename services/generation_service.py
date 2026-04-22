from __future__ import annotations

from core.config import AppConfig
from core.logging import get_logger
from features.registry import build_registry
from generator.engine import AlphaGenerationEngine
from generator.guided_generator import GuidedGenerator
from memory.pattern_memory import PatternMemoryService
from services.data_service import (
    apply_local_validation_field_penalties,
    build_generation_guardrails,
    filter_generation_case_snapshot,
    load_research_context,
    persist_research_metadata,
    resolve_field_registry,
    sanitize_generation_research_context,
)
from services.export_service import export_generated_alphas
from services.models import CommandEnvironment, GenerationServiceResult
from storage.repository import SQLiteRepository


def generate_and_persist(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    count: int | None,
) -> GenerationServiceResult:
    """Generate alpha candidates and persist them for the active run."""
    logger = get_logger(__name__, run_id=environment.context.run_id, stage="generate")
    registry = build_registry(
        config.generation.allowed_operators,
        operator_catalog_paths=config.generation.operator_catalog_paths,
    )
    existing = repository.list_existing_normalized_expressions(environment.context.run_id)
    total_count = count or (config.generation.template_count + config.generation.grammar_count)
    research_context = load_research_context(config, environment, stage="generate-data")
    research_context, blocked_fields = sanitize_generation_research_context(
        repository,
        config,
        research_context,
        environment,
        stage="generate",
    )
    research_context, local_validation_penalty = apply_local_validation_field_penalties(
        repository,
        config,
        research_context,
        environment,
        stage="generate",
    )
    field_registry = resolve_field_registry(config, research_context)
    persist_research_metadata(
        repository,
        config,
        environment,
        research_context,
        removed_field_names=blocked_fields,
    )
    generation_guardrails = build_generation_guardrails(repository, config, field_registry)
    blocked_field_set = set(blocked_fields)

    regime_key: str | None = None
    pattern_count = 0
    if config.adaptive_generation.enabled:
        snapshot = repository.alpha_history.load_snapshot(
            regime_key=research_context.regime_key,
            region=research_context.region,
            global_regime_key=research_context.global_regime_key,
            parent_pool_size=config.adaptive_generation.parent_pool_size,
            region_learning_config=config.adaptive_generation.region_learning,
            pattern_decay=config.adaptive_generation.pattern_decay,
            prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
        )
        case_snapshot = filter_generation_case_snapshot(
            repository.alpha_history.load_case_snapshot(
                research_context.regime_key,
                region=research_context.region,
                global_regime_key=research_context.global_regime_key,
                region_learning_config=config.adaptive_generation.region_learning,
            ),
            blocked_fields=blocked_field_set,
        )
        engine = GuidedGenerator(
            generation_config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            memory_service=PatternMemoryService(),
            field_registry=field_registry,
            region_learning_context=research_context.region_learning_context,
            generation_guardrails=generation_guardrails,
            field_penalty_multipliers=local_validation_penalty.multipliers,
        )
        candidates = engine.generate(
            count=total_count,
            snapshot=snapshot,
            existing_normalized=existing,
            case_snapshot=case_snapshot,
        )
        regime_key = research_context.regime_key
        pattern_count = len(snapshot.patterns)
        logger.info("Adaptive generation used regime %s with %s learned patterns.", regime_key[:12], pattern_count)
    else:
        case_snapshot = filter_generation_case_snapshot(
            repository.alpha_history.load_case_snapshot(
                research_context.regime_key,
                region=research_context.region,
                global_regime_key=research_context.global_regime_key,
                region_learning_config=config.adaptive_generation.region_learning,
            ),
            blocked_fields=blocked_field_set,
        )
        engine = AlphaGenerationEngine(
            config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=research_context.region_learning_context,
            generation_guardrails=generation_guardrails,
            field_penalty_multipliers=local_validation_penalty.multipliers,
        )
        candidates = engine.generate(count=total_count, existing_normalized=existing, case_snapshot=case_snapshot)

    inserted = repository.save_alpha_candidates(environment.context.run_id, candidates)
    export_paths = export_generated_alphas(repository, environment)
    repository.update_run_status(environment.context.run_id, "generated")
    logger.info("Generated %s candidates and inserted %s new rows.", len(candidates), inserted)
    logger.info("Exported generated alpha CSV to %s", export_paths["generated_alphas_latest_csv"])
    return GenerationServiceResult(
        generated_count=len(candidates),
        inserted_count=inserted,
        region=research_context.region,
        regime_key=regime_key,
        global_regime_key=research_context.global_regime_key,
        pattern_count=pattern_count,
        export_paths=export_paths,
    )
