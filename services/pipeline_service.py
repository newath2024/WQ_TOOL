from __future__ import annotations

from core.config import AppConfig
from core.logging import get_logger
from services.data_service import load_research_context, persist_research_metadata
from services.evaluation_service import evaluate_and_persist
from services.generation_service import generate_and_persist
from services.models import CommandEnvironment
from storage.repository import SQLiteRepository


def run_full_pipeline(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    count: int | None,
) -> int:
    """Run the full load, generate, evaluate, and select workflow."""
    logger = get_logger(__name__, run_id=environment.context.run_id, stage="pipeline")
    research_context = load_research_context(config, environment, stage="pipeline-load")
    persist_research_metadata(repository, config, environment, research_context)
    generation_result = generate_and_persist(repository, config, environment, count=count)
    logger.info("Pipeline generation inserted %s new alphas.", generation_result.inserted_count)
    evaluation_result = evaluate_and_persist(
        repository,
        config,
        environment,
        status="completed",
        finished=True,
    )
    logger.info("Pipeline completed with %s selected alphas.", len(evaluation_result.selection_records))
    return 0
