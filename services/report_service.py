from __future__ import annotations

from core.config import AppConfig
from services.data_service import resolve_regime_key
from services.models import CommandEnvironment, ReportSummary, TopAlphaRow
from storage.repository import SQLiteRepository


def list_top_alphas(
    repository: SQLiteRepository,
    environment: CommandEnvironment,
    limit: int,
) -> list[TopAlphaRow]:
    """Return top ranked alphas for the active run."""
    rows = repository.get_top_selections(environment.context.run_id, limit=limit)
    return [
        TopAlphaRow(
            rank=int(row["rank"]),
            alpha_id=str(row["alpha_id"]),
            validation_fitness=float(row["validation_fitness"]),
            expression=str(row["expression"]),
            generation_mode=str(row["generation_mode"]),
            complexity=int(row["complexity"]),
            delay_mode=str(row.get("delay_mode") or "-"),
            neutralization=str(row.get("neutralization") or "-"),
            submission_pass_count=int(row.get("submission_pass_count") or 0),
            cache_hit=bool(row.get("cache_hit") or 0),
        )
        for row in rows
    ]


def build_report_summary(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    limit: int,
) -> ReportSummary | None:
    """Build a report payload for one run."""
    run = repository.get_run(environment.context.run_id)
    if run is None:
        return None
    regime_key = run.regime_key or resolve_regime_key(repository, config, environment, stage="report-data")
    cache_stats = repository.get_cache_stats(environment.context.run_id)
    submission_rows = repository.get_submission_tests_for_run(environment.context.run_id)
    summary: dict[str, dict[str, int]] = {}
    for row in submission_rows:
        test_name = row["test_name"]
        bucket = summary.setdefault(test_name, {"passed": 0, "total": 0})
        bucket["passed"] += int(row["passed"])
        bucket["total"] += 1

    hard_filter_summary = {
        "min_validation_sharpe": config.evaluation.hard_filters.min_validation_sharpe,
        "max_validation_turnover": config.evaluation.hard_filters.max_validation_turnover,
        "min_validation_observations": config.evaluation.data_requirements.min_validation_observations,
        "max_validation_drawdown": config.evaluation.hard_filters.max_validation_drawdown,
        "min_stability": config.evaluation.data_requirements.min_stability,
        "signal_correlation_threshold": config.evaluation.diversity.signal_correlation_threshold,
        "returns_correlation_threshold": config.evaluation.diversity.returns_correlation_threshold,
    }

    return ReportSummary(
        run=run,
        profile_name=run.profile_name or config.runtime.profile_name,
        dataset_fingerprint=run.dataset_fingerprint or "",
        regime_key=regime_key,
        selected_timeframe=run.selected_timeframe or config.backtest.timeframe,
        cache_hits=cache_stats["cache_hits"],
        validation_rows=cache_stats["validation_rows"],
        top_alphas=list_top_alphas(repository, environment, limit),
        submission_summary=summary,
        top_gene=(repository.alpha_history.get_top_genes(regime_key, limit=1) or [None])[0],
        fail_tags=repository.alpha_history.get_run_fail_tag_summary(environment.context.run_id),
        rejection_reasons=repository.alpha_history.get_run_rejection_reason_summary(environment.context.run_id, limit),
        generation_mix=repository.get_generation_mix(environment.context.run_id),
        hard_filter_summary=hard_filter_summary,
    )
