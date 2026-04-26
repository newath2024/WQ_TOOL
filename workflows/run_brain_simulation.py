from __future__ import annotations

from adapters.brain_manual_adapter import BrainManualAdapter
from core.config import AppConfig
from domain.candidate import AlphaCandidate
from domain.metrics import CandidateScore
from domain.simulation import BrainSimulationBatch
from services.brain_batch_service import BrainBatchService
from services.brain_service import BrainService
from services.models import CommandEnvironment
from storage.repository import SQLiteRepository


def generate_and_select_candidates(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    *,
    count: int | None = None,
) -> tuple[list[AlphaCandidate], list[CandidateScore]]:
    return BrainBatchService(repository).prepare_next_batch(
        config=config,
        environment=environment,
        count=count,
    )


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
