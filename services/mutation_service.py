from __future__ import annotations

from core.config import AppConfig
from core.logging import get_logger
from features.registry import build_registry
from generator.engine import AlphaGenerationEngine
from generator.guided_generator import GuidedGenerator
from memory.pattern_memory import PatternMemoryService
from services.data_service import (
    load_research_context,
    persist_research_metadata,
    resolve_field_registry,
    sanitize_generation_research_context,
)
from services.evaluation_service import alpha_candidate_from_record
from services.models import CommandEnvironment, GenerationServiceResult
from storage.repository import SQLiteRepository


def mutate_and_persist(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    from_top: int,
    count: int,
) -> GenerationServiceResult:
    """Generate mutated alpha candidates from top-ranked parents."""
    logger = get_logger(__name__, run_id=environment.context.run_id, stage="mutate")
    parent_records = repository.get_top_alpha_records(environment.context.run_id, limit=from_top)
    if not parent_records:
        logger.warning("No top alphas available for mutation in run %s.", environment.context.run_id)
        return GenerationServiceResult(generated_count=0, inserted_count=0, exit_code=1)

    existing = repository.list_existing_normalized_expressions(environment.context.run_id)
    registry = build_registry(
        config.generation.allowed_operators,
        operator_catalog_paths=config.generation.operator_catalog_paths,
    )
    research_context = load_research_context(config, environment, stage="mutate-data")
    research_context, blocked_fields = sanitize_generation_research_context(
        repository,
        config,
        research_context,
        environment,
        stage="mutate",
    )
    field_registry = resolve_field_registry(config, research_context)
    persist_research_metadata(
        repository,
        config,
        environment,
        research_context,
        removed_field_names=blocked_fields,
    )

    regime_key: str | None = None
    if config.adaptive_generation.enabled:
        snapshot = repository.alpha_history.load_snapshot(
            regime_key=research_context.regime_key,
            region=research_context.region,
            global_regime_key=research_context.global_regime_key,
            parent_pool_size=max(config.adaptive_generation.parent_pool_size, from_top * 2),
            region_learning_config=config.adaptive_generation.region_learning,
            pattern_decay=config.adaptive_generation.pattern_decay,
            prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
        )
        case_snapshot = repository.alpha_history.load_case_snapshot(
            research_context.regime_key,
            region=research_context.region,
            global_regime_key=research_context.global_regime_key,
            region_learning_config=config.adaptive_generation.region_learning,
        )
        selected_parent_ids = {record.alpha_id for record in parent_records}
        parent_pool = [
            parent
            for parent in snapshot.top_parents
            if parent.alpha_id in selected_parent_ids and parent.run_id == environment.context.run_id
        ]
        if not parent_pool:
            parent_pool = list(snapshot.top_parents[:from_top])
        engine = GuidedGenerator(
            generation_config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            memory_service=PatternMemoryService(),
            field_registry=field_registry,
            region_learning_context=research_context.region_learning_context,
        )
        candidates = engine.generate_mutations(
            count=count,
            snapshot=snapshot,
            parent_pool=parent_pool,
            existing_normalized=existing,
            case_snapshot=case_snapshot,
        )
        regime_key = research_context.regime_key
    else:
        case_snapshot = repository.alpha_history.load_case_snapshot(
            research_context.regime_key,
            region=research_context.region,
            global_regime_key=research_context.global_regime_key,
            region_learning_config=config.adaptive_generation.region_learning,
        )
        engine = AlphaGenerationEngine(
            config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=research_context.region_learning_context,
        )
        parents = [alpha_candidate_from_record(record) for record in parent_records]
        candidates = engine.generate_mutations(
            parents=parents,
            count=count,
            existing_normalized=existing,
            case_snapshot=case_snapshot,
        )

    inserted = repository.save_alpha_candidates(environment.context.run_id, candidates)
    repository.update_run_status(environment.context.run_id, "mutated")
    logger.info("Mutated %s candidates and inserted %s new rows.", len(candidates), inserted)
    return GenerationServiceResult(
        generated_count=len(candidates),
        inserted_count=inserted,
        region=research_context.region,
        regime_key=regime_key,
        global_regime_key=research_context.global_regime_key,
    )
