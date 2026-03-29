from __future__ import annotations

from core.config import AppConfig
from core.logging import get_logger
from features.registry import build_registry
from generator.engine import AlphaGenerationEngine
from generator.guided_generator import GuidedGenerator
from memory.pattern_memory import PatternMemoryService
from services.data_service import load_research_context, persist_research_metadata
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
    registry = build_registry(config.generation.allowed_operators)
    existing = repository.list_existing_normalized_expressions(environment.context.run_id)
    total_count = count or (config.generation.template_count + config.generation.grammar_count)

    regime_key: str | None = None
    pattern_count = 0
    if config.adaptive_generation.enabled:
        research_context = load_research_context(config, environment, stage="generate-data")
        snapshot = repository.alpha_history.load_snapshot(
            regime_key=research_context.regime_key,
            parent_pool_size=config.adaptive_generation.parent_pool_size,
        )
        engine = GuidedGenerator(
            generation_config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            memory_service=PatternMemoryService(),
        )
        candidates = engine.generate(count=total_count, snapshot=snapshot, existing_normalized=existing)
        persist_research_metadata(repository, config, environment, research_context)
        regime_key = research_context.regime_key
        pattern_count = len(snapshot.patterns)
        logger.info("Adaptive generation used regime %s with %s learned patterns.", regime_key[:12], pattern_count)
    else:
        engine = AlphaGenerationEngine(config=config.generation, registry=registry)
        candidates = engine.generate(count=total_count, existing_normalized=existing)

    inserted = repository.save_alpha_candidates(environment.context.run_id, candidates)
    repository.update_run_status(environment.context.run_id, "generated")
    logger.info("Generated %s candidates and inserted %s new rows.", len(candidates), inserted)
    return GenerationServiceResult(
        generated_count=len(candidates),
        inserted_count=inserted,
        regime_key=regime_key,
        pattern_count=pattern_count,
    )
