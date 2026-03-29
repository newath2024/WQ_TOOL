from __future__ import annotations

from core.config import AppConfig
from services.data_service import resolve_regime_key
from services.models import CommandEnvironment, PatternView
from storage.repository import SQLiteRepository


def get_top_patterns(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    limit: int,
    kind: str | None = None,
) -> list[PatternView]:
    """Return the highest scoring memory patterns for the active regime."""
    regime_key = resolve_regime_key(repository, config, environment, stage="memory-top-patterns-data")
    rows = repository.alpha_history.get_top_patterns(regime_key=regime_key, limit=limit, pattern_kind=kind)
    return [
        PatternView(
            pattern_kind=str(row["pattern_kind"]),
            pattern_value=str(row["pattern_value"]),
            pattern_score=float(row["pattern_score"]),
            support=int(row["support"]),
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
        )
        for row in rows
    ]


def get_failed_patterns(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    limit: int,
) -> list[PatternView]:
    """Return the most failure-prone patterns for the active regime."""
    regime_key = resolve_regime_key(repository, config, environment, stage="memory-failed-patterns-data")
    rows = repository.alpha_history.get_failed_patterns(regime_key=regime_key, limit=limit)
    return [
        PatternView(
            pattern_kind=str(row["pattern_kind"]),
            pattern_value=str(row["pattern_value"]),
            pattern_score=float(row["pattern_score"]),
            support=int(row["support"]),
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
        )
        for row in rows
    ]


def get_top_genes(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    limit: int,
) -> list[PatternView]:
    """Return the strongest reusable subexpression genes for the active regime."""
    regime_key = resolve_regime_key(repository, config, environment, stage="memory-top-genes-data")
    rows = repository.alpha_history.get_top_genes(regime_key=regime_key, limit=limit)
    return [
        PatternView(
            pattern_kind=str(row["pattern_kind"]),
            pattern_value=str(row["pattern_value"]),
            pattern_score=float(row["pattern_score"]),
            support=int(row["support"]),
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
        )
        for row in rows
    ]
