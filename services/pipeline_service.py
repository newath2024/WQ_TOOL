from __future__ import annotations

from core.config import AppConfig
from core.logging import get_logger
from services.data_service import (
    load_research_context,
    persist_research_metadata,
    sanitize_generation_research_context,
)
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
    research_context, blocked_fields = sanitize_generation_research_context(
        repository,
        config,
        research_context,
        environment,
        stage="pipeline-load",
    )
    persist_research_metadata(
        repository,
        config,
        environment,
        research_context,
        removed_field_names=blocked_fields,
    )
    generation_result = generate_and_persist(repository, config, environment, count=count)
    logger.info("Pipeline generation inserted %s new alphas.", generation_result.inserted_count)
    if generation_result.export_paths.get("generated_alphas_latest_csv"):
        logger.info(
            "Generated alpha CSV available at %s",
            generation_result.export_paths["generated_alphas_latest_csv"],
        )
    evaluation_result = evaluate_and_persist(
        repository,
        config,
        environment,
        status="completed",
        finished=True,
    )
    if evaluation_result.export_paths.get("evaluated_alphas_latest_csv"):
        logger.info(
            "Evaluated alpha CSV available at %s",
            evaluation_result.export_paths["evaluated_alphas_latest_csv"],
        )
    if evaluation_result.export_paths.get("selected_alphas_latest_csv"):
        logger.info(
            "Selected alpha CSV available at %s",
            evaluation_result.export_paths["selected_alphas_latest_csv"],
        )
    logger.info("Pipeline completed with %s selected alphas.", len(evaluation_result.selection_records))
    return 0
