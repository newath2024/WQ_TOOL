from __future__ import annotations

from core.config import AppConfig
from services.data_service import resolve_region_learning_context
from services.models import CommandEnvironment, PatternView
from storage.repository import SQLiteRepository


def get_top_patterns(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    limit: int,
    kind: str | None = None,
    scope: str = "blended",
) -> list[PatternView]:
    """Return the highest scoring memory patterns for the active regime."""
    learning_context = resolve_region_learning_context(repository, config, environment, stage="memory-top-patterns-data")
    snapshot = repository.alpha_history.load_snapshot(
        regime_key=learning_context.regime_key,
        region=learning_context.region,
        global_regime_key=learning_context.global_regime_key,
        parent_pool_size=config.adaptive_generation.parent_pool_size,
        region_learning_config=config.adaptive_generation.region_learning,
        pattern_decay=config.adaptive_generation.pattern_decay,
        prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
    )
    rows = snapshot.ordered_patterns(scope=scope, kind=kind, limit=limit)
    return [
        PatternView(
            pattern_kind=str(row.pattern_kind),
            pattern_value=str(row.pattern_value),
            pattern_score=float(row.pattern_score),
            support=int(row.support),
            success_count=int(row.success_count),
            failure_count=int(row.failure_count),
        )
        for row in rows
    ]


def get_failed_patterns(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    limit: int,
    scope: str = "blended",
) -> list[PatternView]:
    """Return the most failure-prone patterns for the active regime."""
    learning_context = resolve_region_learning_context(repository, config, environment, stage="memory-failed-patterns-data")
    snapshot = repository.alpha_history.load_snapshot(
        regime_key=learning_context.regime_key,
        region=learning_context.region,
        global_regime_key=learning_context.global_regime_key,
        parent_pool_size=config.adaptive_generation.parent_pool_size,
        region_learning_config=config.adaptive_generation.region_learning,
        pattern_decay=config.adaptive_generation.pattern_decay,
        prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
    )
    rows = sorted(
        snapshot.patterns_for_scope(scope).values(),
        key=lambda item: (-item.failure_count, item.pattern_score, -item.support, item.pattern_kind, item.pattern_value),
    )[:limit]
    return [
        PatternView(
            pattern_kind=str(row.pattern_kind),
            pattern_value=str(row.pattern_value),
            pattern_score=float(row.pattern_score),
            support=int(row.support),
            success_count=int(row.success_count),
            failure_count=int(row.failure_count),
        )
        for row in rows
    ]


def get_top_genes(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    limit: int,
    scope: str = "blended",
) -> list[PatternView]:
    """Return the strongest reusable subexpression genes for the active regime."""
    rows = get_top_patterns(
        repository,
        config,
        environment,
        limit=limit,
        kind="subexpression",
        scope=scope,
    )
    return [
        PatternView(
            pattern_kind=str(row.pattern_kind),
            pattern_value=str(row.pattern_value),
            pattern_score=float(row.pattern_score),
            support=int(row.support),
            success_count=int(row.success_count),
            failure_count=int(row.failure_count),
        )
        for row in rows
    ]
