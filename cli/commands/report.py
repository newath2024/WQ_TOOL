from __future__ import annotations

import argparse
import json

from core.config import AppConfig
from core.logging import get_logger
from services.models import CommandEnvironment
from services.report_service import build_report_summary, list_top_alphas
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the top and report commands."""
    top_parser = subparsers.add_parser("top", help="Display selected top alphas.", parents=[common])
    top_parser.add_argument("--limit", type=int, default=20, help="Number of rows to display.")
    top_parser.set_defaults(command_handler=handle_top)

    report_parser = subparsers.add_parser("report", help="Display a summary report for the run.", parents=[common])
    report_parser.add_argument("--limit", type=int, default=10, help="Number of top alphas to include.")
    report_parser.set_defaults(command_handler=handle_report)


def handle_top(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    """Execute the top command."""
    del config
    logger = get_logger(__name__, run_id=environment.context.run_id, stage="top")
    rows = list_top_alphas(repository, environment, limit=args.limit)
    if not rows:
        logger.warning("No selections found for run %s.", environment.context.run_id)
        return 1
    for row in rows:
        print(
            f"rank={row.rank:>2} alpha_id={row.alpha_id} "
            f"fitness={row.validation_fitness:.4f} mode={row.generation_mode:<8} "
            f"delay={row.delay_mode:<7} neutral={row.neutralization:<16} "
            f"submission={row.submission_pass_count:<2} cache={int(row.cache_hit)} "
            f"complexity={row.complexity:<2} expr={row.expression}"
        )
    return 0


def handle_report(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    """Execute the report command."""
    logger = get_logger(__name__, run_id=environment.context.run_id, stage="report")
    summary = build_report_summary(repository, config, environment, limit=args.limit)
    if summary is None:
        logger.warning("Run %s was not found.", environment.context.run_id)
        return 1

    print(
        f"run_id={summary.run.run_id} status={summary.run.status} started_at={summary.run.started_at} "
        f"profile={summary.profile_name} timeframe={summary.selected_timeframe}"
    )
    print(
        f"region={(summary.region or '-')}"
        f" dataset_fingerprint={(summary.dataset_fingerprint or '-')[:12]} "
        f"regime_key={(summary.regime_key or '-')[:12]} "
        f"global_regime_key={(summary.global_regime_key or '-')[:12]}"
    )
    cache_rate = (summary.cache_hits / summary.validation_rows) if summary.validation_rows > 0 else 0.0
    print(
        f"cache_hits={summary.cache_hits} validation_rows={summary.validation_rows} "
        f"cache_hit_rate={cache_rate:.2%}"
    )
    if summary.pattern_blend is not None:
        print(
            f"pattern_blend: local_weight={summary.pattern_blend.local_weight:.2f} "
            f"global_weight={summary.pattern_blend.global_weight:.2f} "
            f"local_samples={summary.pattern_blend.local_samples} "
            f"global_samples={summary.pattern_blend.global_samples}"
        )
    if summary.case_blend is not None:
        print(
            f"case_blend: local_weight={summary.case_blend.local_weight:.2f} "
            f"global_weight={summary.case_blend.global_weight:.2f} "
            f"local_samples={summary.case_blend.local_samples} "
            f"global_samples={summary.case_blend.global_samples}"
        )
    if summary.latest_regime_snapshot:
        print(
            "latest_regime: "
            f"market_key={summary.latest_regime_snapshot.get('market_regime_key') or '-'} "
            f"effective_key={str(summary.latest_regime_snapshot.get('effective_regime_key') or '-')[:12]} "
            f"label={summary.latest_regime_snapshot.get('regime_label') or '-'} "
            f"confidence={float(summary.latest_regime_snapshot.get('confidence') or 0.0):.2f}"
        )
    pre_sim_metrics = next(
        (
            json.loads(row["metrics_json"])
            for row in reversed(summary.stage_metrics)
            if row.get("stage") == "pre_sim"
        ),
        None,
    )
    if pre_sim_metrics is not None:
        print(
            "pre_sim_funnel: "
            f"generated={pre_sim_metrics.get('generated', 0)} "
            f"blocked_exact={pre_sim_metrics.get('blocked_by_exact_dedup', 0)} "
            f"blocked_near={pre_sim_metrics.get('blocked_by_near_duplicate', 0)} "
            f"blocked_cross_run={pre_sim_metrics.get('blocked_by_cross_run_dedup', 0)} "
            f"kept={pre_sim_metrics.get('kept_after_dedup', 0)} "
            f"selected={pre_sim_metrics.get('selected_for_simulation', 0)} "
            f"avg_crowding_penalty={float(summary.avg_crowding_penalty):.4f}"
        )
    if summary.duplicate_summary:
        print("duplicate_summary:")
        for row in summary.duplicate_summary[: args.limit]:
            print(f"  {row['stage']} {row['decision']} {row['reason_code']}: {row['total_count']}")
    print(
        "hard_filters: "
        + " ".join(f"{key}={value}" for key, value in summary.hard_filter_summary.items())
    )
    if summary.top_alphas:
        print("top_alphas:")
        for row in summary.top_alphas:
            print(
                f"  rank={row.rank} alpha_id={row.alpha_id} fitness={row.validation_fitness:.4f} "
                f"delay={row.delay_mode} neutral={row.neutralization} "
                f"submission={row.submission_pass_count} expr={row.expression}"
            )
    if summary.submission_summary:
        print("submission_tests:")
        for test_name, bucket in summary.submission_summary.items():
            print(f"  {test_name}: {bucket['passed']}/{bucket['total']} passed")
    if summary.top_gene:
        print(
            f"top_gene: kind={summary.top_gene['pattern_kind']} "
            f"score={summary.top_gene['pattern_score']:.4f} value={summary.top_gene['pattern_value']}"
        )
    if summary.fail_tags:
        print("common_fail_tags:")
        for row in summary.fail_tags[: args.limit]:
            print(f"  {row['tag']}: {row['total_count']}")
    if summary.rejection_reasons:
        print("top_rejection_reasons:")
        for row in summary.rejection_reasons:
            print(f"  {row['reason']}: {row['total_count']}")
    if summary.generation_mix:
        print("generation_mix:")
        for row in summary.generation_mix:
            print(f"  {row['generation_mode']}: {row['alpha_count']}")
    return 0
