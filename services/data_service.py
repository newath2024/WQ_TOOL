from __future__ import annotations

import pandas as pd

from core.config import AppConfig, PeriodConfig
from core.logging import get_logger
from data.loader import load_market_data
from memory.pattern_memory import PatternMemoryService
from services.models import CommandEnvironment, ResearchContext
from storage.repository import SQLiteRepository
from features.transforms import build_research_matrices


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
    memory_service = PatternMemoryService()
    regime_key = memory_service.build_regime_key(bundle.fingerprint, config)
    return ResearchContext(
        bundle=bundle,
        matrices=matrices,
        regime_key=regime_key,
        memory_service=memory_service,
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
