from __future__ import annotations

from core.config import AppConfig
from core.logging import get_logger
from features.registry import build_registry
from generator.engine import AlphaGenerationEngine
from generator.guided_generator import GuidedGenerator
from memory.pattern_memory import PatternMemoryService
from services.data_service import load_research_context, persist_research_metadata
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
    registry = build_registry(config.generation.allowed_operators)

    regime_key: str | None = None
    if config.adaptive_generation.enabled:
        research_context = load_research_context(config, environment, stage="mutate-data")
        snapshot = repository.alpha_history.load_snapshot(
            regime_key=research_context.regime_key,
            parent_pool_size=max(config.adaptive_generation.parent_pool_size, from_top * 2),
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
        )
        candidates = engine.generate_mutations(
            count=count,
            snapshot=snapshot,
            parent_pool=parent_pool,
            existing_normalized=existing,
        )
        persist_research_metadata(repository, config, environment, research_context)
        regime_key = research_context.regime_key
    else:
        engine = AlphaGenerationEngine(config=config.generation, registry=registry)
        parents = [alpha_candidate_from_record(record) for record in parent_records]
        candidates = engine.generate_mutations(parents=parents, count=count, existing_normalized=existing)

    inserted = repository.save_alpha_candidates(environment.context.run_id, candidates)
    repository.update_run_status(environment.context.run_id, "mutated")
    logger.info("Mutated %s candidates and inserted %s new rows.", len(candidates), inserted)
    return GenerationServiceResult(generated_count=len(candidates), inserted_count=inserted, regime_key=regime_key)
