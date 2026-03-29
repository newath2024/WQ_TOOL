from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from core.config import AppConfig, PeriodConfig
from core.logging import get_logger
from data.field_registry import (
    FieldRegistry,
    FieldScoreWeights,
    build_field_registry,
    load_runtime_field_values,
)
from data.loader import load_market_data
from features.transforms import build_research_matrices
from memory.pattern_memory import PatternMemoryService
from services.models import CommandEnvironment, ResearchContext
from storage.models import FieldCatalogRecord, RunFieldScoreRecord
from storage.repository import SQLiteRepository


def load_dataset(config: AppConfig, environment: CommandEnvironment, stage: str):
    """Load and log the configured dataset."""
    logger = get_logger(__name__, run_id=environment.context.run_id, stage=stage)
    bundle = load_market_data(config.data, logger, aux_config=config.aux_data)
    logger.info("Loaded market data summary: %s", bundle.summary())
    return bundle


def load_research_context(
    config: AppConfig,
    environment: CommandEnvironment,
    stage: str,
) -> ResearchContext:
    """Load bundle, matrices, and memory context for one command."""
    bundle = load_dataset(config, environment, stage=stage)
    matrices = build_research_matrices(bundle.get_timeframe_data(config.backtest.timeframe))
    runtime_bundle = load_runtime_field_values(
        config.generation.field_value_paths,
        default_timeframe=config.backtest.timeframe,
    )
    runtime_numeric_fields, runtime_group_fields = runtime_bundle.for_timeframe(config.backtest.timeframe)
    aligned_numeric_fields, aligned_group_fields = _align_runtime_fields(
        matrices=matrices,
        numeric_fields=runtime_numeric_fields,
        group_fields=runtime_group_fields,
    )
    matrices = matrices.with_additional_fields(
        numeric_fields=aligned_numeric_fields,
        group_fields=aligned_group_fields,
    )
    memory_service = PatternMemoryService()
    regime_key = memory_service.build_regime_key(bundle.fingerprint, config)
    score_weights = FieldScoreWeights(
        coverage=float(config.generation.field_score_weights.get("coverage", 0.50)),
        usage=float(config.generation.field_score_weights.get("usage", 0.30)),
        category=float(config.generation.field_score_weights.get("category", 0.20)),
    )
    field_registry = build_field_registry(
        catalog_paths=config.generation.field_catalog_paths,
        runtime_numeric_fields=matrices.numeric_fields,
        runtime_group_fields=matrices.group_fields,
        category_weights=config.generation.category_weights,
        score_weights=score_weights,
    )
    return ResearchContext(
        bundle=bundle,
        matrices=matrices,
        regime_key=regime_key,
        memory_service=memory_service,
        field_registry=field_registry,
    )


def persist_research_metadata(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    research_context: ResearchContext,
) -> None:
    """Persist dataset summary and research context metadata for the active run."""
    repository.save_dataset_summary(
        environment.context.run_id,
        research_context.bundle.summary(),
        dataset_fingerprint=research_context.bundle.fingerprint,
        selected_timeframe=config.backtest.timeframe,
        regime_key=research_context.regime_key,
    )
    timestamp = datetime.now(UTC).isoformat()
    repository.save_field_catalog(
        [
            FieldCatalogRecord(
                field_name=spec.name,
                dataset=spec.dataset,
                field_type=spec.field_type,
                coverage=spec.coverage,
                alpha_usage_count=spec.alpha_usage_count,
                category=spec.category,
                delay=spec.delay,
                region=spec.region,
                universe=spec.universe,
                runtime_available=spec.runtime_available,
                description=spec.description,
                subcategory=spec.subcategory,
                user_count=spec.user_count,
                category_weight=spec.category_weight,
                field_score=spec.field_score,
                updated_at=timestamp,
            )
            for spec in research_context.field_registry.fields.values()
        ]
    )
    repository.replace_run_field_scores(
        environment.context.run_id,
        [
            RunFieldScoreRecord(
                run_id=environment.context.run_id,
                field_name=spec.name,
                runtime_available=spec.runtime_available,
                field_type=spec.field_type,
                category=spec.category,
                field_score=spec.field_score,
                coverage=spec.coverage,
                alpha_usage_count=spec.alpha_usage_count,
                created_at=timestamp,
            )
            for spec in research_context.field_registry.fields.values()
        ],
    )


def load_and_persist_dataset(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    stage: str = "load-data",
):
    """Load the configured dataset and persist its summary."""
    bundle = load_dataset(config, environment, stage=stage)
    repository.save_dataset_summary(
        environment.context.run_id,
        bundle.summary(),
        dataset_fingerprint=bundle.fingerprint,
        selected_timeframe=config.backtest.timeframe,
    )
    return bundle


def resolve_regime_key(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    stage: str,
) -> str:
    """Resolve the active regime key, preferring persisted run metadata when present."""
    run = repository.get_run(environment.context.run_id)
    if run and run.regime_key:
        return run.regime_key
    research_context = load_research_context(config, environment, stage=stage)
    persist_research_metadata(repository, config, environment, research_context)
    return research_context.regime_key


def slice_frame_by_period(frame: pd.DataFrame, period: PeriodConfig) -> pd.DataFrame:
    """Slice a dataframe by inclusive period boundaries."""
    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end)
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def slice_series_by_period(series: pd.Series, period: PeriodConfig) -> pd.Series:
    """Slice a series by inclusive period boundaries."""
    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end)
    return series.loc[(series.index >= start) & (series.index <= end)]


def combine_series_for_periods(series: pd.Series, *periods: PeriodConfig) -> pd.Series:
    """Concatenate series slices for several periods."""
    pieces = [slice_series_by_period(series, period) for period in periods]
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


def resolve_field_registry(
    config: AppConfig,
    research_context: ResearchContext,
) -> FieldRegistry:
    """Return the active field registry for generation and validation."""
    allowed = set(config.generation.allowed_fields or [])
    if allowed:
        filtered = {
            name: spec
            for name, spec in research_context.field_registry.fields.items()
            if name in allowed or spec.operator_type == "group" or not spec.runtime_available
        }
        return FieldRegistry(fields=filtered)
    return research_context.field_registry


def _align_runtime_fields(
    matrices,
    numeric_fields: dict[str, pd.DataFrame],
    group_fields: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    close = matrices.numeric_fields["close"]
    index = close.index
    columns = close.columns
    aligned_numeric = {
        name: frame.reindex(index=index, columns=columns)
        for name, frame in numeric_fields.items()
    }
    aligned_groups = {
        name: frame.reindex(index=index, columns=columns)
        for name, frame in group_fields.items()
    }
    return aligned_numeric, aligned_groups
