from __future__ import annotations

from adapters.brain_manual_adapter import BrainManualAdapter
from core.config import AppConfig
from features.registry import build_registry
from generator.engine import AlphaCandidate, AlphaGenerationEngine
from generator.guided_generator import GuidedGenerator
from services.brain_service import BrainService
from services.candidate_selection_service import CandidateSelectionService
from services.data_service import (
    load_research_context,
    persist_research_metadata,
    resolve_field_registry,
)
from services.models import BrainSimulationBatch, CandidateScore, CommandEnvironment
from storage.repository import SQLiteRepository


def generate_and_select_candidates(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    *,
    count: int | None = None,
) -> tuple[list[AlphaCandidate], list[CandidateScore]]:
    research_context = load_research_context(config, environment, stage="brain-sim-data")
    persist_research_metadata(repository, config, environment, research_context)
    field_registry = resolve_field_registry(config, research_context)
    registry = build_registry(config.generation.allowed_operators)
    snapshot = repository.alpha_history.load_snapshot(
        regime_key=research_context.regime_key,
        parent_pool_size=config.adaptive_generation.parent_pool_size,
    )
    existing_normalized = repository.list_existing_normalized_expressions(environment.context.run_id)
    generation_count = count or config.loop.generation_batch_size
    if config.adaptive_generation.enabled:
        engine = GuidedGenerator(
            generation_config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            memory_service=research_context.memory_service,
            field_registry=field_registry,
        )
        candidates = engine.generate(
            count=generation_count,
            snapshot=snapshot,
            existing_normalized=existing_normalized,
        )
    else:
        engine = AlphaGenerationEngine(config=config.generation, registry=registry, field_registry=field_registry)
        candidates = engine.generate(count=generation_count, existing_normalized=existing_normalized)
    repository.save_alpha_candidates(environment.context.run_id, candidates)
    selector = CandidateSelectionService(research_context.memory_service)
    selected, _ = selector.select_for_simulation(
        candidates,
        snapshot=snapshot,
        field_registry=field_registry,
        batch_size=config.loop.simulation_batch_size,
        min_pattern_support=config.adaptive_generation.min_pattern_support,
        rejection_filters=config.loop.rejection_filters,
    )
    return candidates, selected


def run_brain_simulation(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    *,
    count: int | None = None,
) -> BrainSimulationBatch:
    _, selected = generate_and_select_candidates(repository, config, environment, count=count)
    brain_service = BrainService(repository, config.brain)
    return brain_service.simulate_candidates(
        [item.candidate for item in selected],
        config=config,
        environment=environment,
        round_index=1,
        batch_size=config.loop.simulation_batch_size,
    )


def export_brain_candidates(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    *,
    count: int | None = None,
) -> BrainSimulationBatch:
    _, selected = generate_and_select_candidates(repository, config, environment, count=count)
    brain_service = BrainService(
        repository,
        config.brain,
        adapter=BrainManualAdapter(export_root=config.brain.manual_export_dir),
    )
    return brain_service.submit_candidates(
        [item.candidate for item in selected],
        config=config,
        environment=environment,
        round_index=1,
        batch_size=config.loop.simulation_batch_size,
    )
