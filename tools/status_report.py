from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import load_config
from services.kpi_report_service import build_run_kpi_report, run_kpi_report_to_dict
from storage.repository import SQLiteRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show the latest run KPI report for service health, funnel, quality, meta-model, and regime."
    )
    parser.add_argument(
        "--config",
        default="config/dev.yaml",
        help="Config file used to resolve the default SQLite path when --db is not provided.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the SQLite database. Overrides the path resolved from --config.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id filter. Defaults to the latest run in the database.",
    )
    parser.add_argument(
        "--recent-rounds",
        type=int,
        default=20,
        help="How many latest distinct rounds to summarize. Use 0 to summarize the whole run.",
    )
    parser.add_argument(
        "--service-name",
        default=None,
        help="Override the configured service lock name. Defaults to service.lock_name from config.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON document instead of human-readable lines.",
    )
    return parser


def resolve_database_path(config_path: str, override_path: str | None) -> Path:
    if override_path:
        return Path(override_path).expanduser().resolve()
    config = load_config(config_path)
    return Path(config.storage.path).expanduser().resolve()


def _print_human(payload: dict[str, object]) -> None:
    run = payload.get("run") or {}
    runtime = payload.get("runtime") or {}
    scope = payload.get("scope") or {}
    health = payload.get("health") or {}
    funnel = payload.get("funnel") or {}
    quality = payload.get("quality") or {}
    meta_model = payload.get("meta_model") or {}
    regime = payload.get("regime") or {}
    mutation = payload.get("mutation") or {}

    print(
        f"run_id={_display(payload.get('run_id'))} "
        f"run_status={_display(run.get('status'))} "
        f"service_status={_display(runtime.get('status') or health.get('service_status'))} "
        f"pending_jobs={runtime.get('pending_job_count', health.get('pending_jobs_runtime', 0))}"
    )
    print(
        f"scope={_display(scope.get('label'))} "
        f"latest_round={_display(scope.get('latest_round'))} "
        f"round_start={_display(scope.get('scope_round_start'))} "
        f"round_end={_display(scope.get('scope_round_end'))} "
        f"round_count={scope.get('scope_round_count') or 0}"
    )
    if health:
        print(
            "health: "
            f"terminal_jobs={health.get('terminal_jobs', 0)} "
            f"completed_rate={_fmt_pct(health.get('completed_rate'))} "
            f"timeout_rate={_fmt_pct(health.get('timeout_rate'))} "
            f"failed_rate={_fmt_pct(health.get('failed_rate'))} "
            f"median_latency_sec={_fmt_float(health.get('median_latency_sec'))} "
            f"avg_latency_sec={_fmt_float(health.get('avg_latency_sec'))}"
        )
        print(
            "timeout_breakdown: "
            f"live={health.get('poll_timeout_live_jobs', 0)} "
            f"after_downtime={health.get('poll_timeout_after_downtime_jobs', 0)} "
            f"legacy={health.get('legacy_poll_timeout_jobs', 0)} "
            f"other={health.get('other_timeout_jobs', 0)}"
        )
    if funnel:
        print(
            "funnel: "
            f"generated={funnel.get('generated_count', 0)} "
            f"validated={funnel.get('validated_count', 0)} "
            f"selected_for_simulation={funnel.get('selected_for_simulation', 0)} "
            f"validation_rate={_fmt_pct(funnel.get('validation_rate'))} "
            f"selection_rate={_fmt_pct(funnel.get('selection_rate'))} "
            f"validate_fail_count={funnel.get('validate_fail_count', 0)}"
        )
    if quality:
        print(
            "quality: "
            f"completed_results={quality.get('completed_results', 0)} "
            f"positive_fitness_rate={_fmt_pct(quality.get('positive_fitness_rate'))} "
            f"positive_sharpe_rate={_fmt_pct(quality.get('positive_sharpe_rate'))} "
            f"avg_fitness={_fmt_float(quality.get('avg_fitness'))} "
            f"avg_sharpe={_fmt_float(quality.get('avg_sharpe'))}"
        )
    if meta_model:
        print(
            "meta_model: "
            f"used_rate={_fmt_pct(meta_model.get('meta_model_used_rate'))} "
            f"avg_train_rows={_fmt_float(meta_model.get('avg_train_rows'))} "
            f"avg_positive_rows={_fmt_float(meta_model.get('avg_positive_rows'))} "
            f"avg_selected_prob={_fmt_float(meta_model.get('avg_selected_prob'))} "
            f"avg_archived_prob={_fmt_float(meta_model.get('avg_archived_prob'))} "
            f"selected_positive_outcome_rate={_fmt_pct(meta_model.get('selected_positive_outcome_rate'))} "
            f"archived_positive_outcome_rate={_fmt_pct(meta_model.get('archived_positive_outcome_rate'))}"
        )
    if regime:
        print(
            "regime: "
            f"market_key={_display(regime.get('latest_market_regime_key'))} "
            f"label={_display(regime.get('latest_regime_label'))} "
            f"confidence={_fmt_float(regime.get('latest_confidence'))} "
            f"learned_cluster={_display(regime.get('latest_learned_cluster_id'))} "
            f"learned_confidence={_fmt_float(regime.get('latest_learned_confidence'))} "
            f"learned_active_rate={_fmt_pct(regime.get('learned_active_rate'))} "
            f"fallback_rate={_fmt_pct(regime.get('fallback_rate'))}"
        )
    if mutation:
        print(
            "mutation: "
            f"selected_for_mutation={mutation.get('selected_for_mutation_count', 0)} "
            f"mutated_children={mutation.get('mutated_children_count', 0)} "
            f"child_better_than_parent_rate={_fmt_pct(mutation.get('child_better_than_parent_rate'))} "
            f"mutation_outcome_rows={mutation.get('mutation_outcome_rows', 0)}"
        )


def _fmt_pct(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2%}"


def _fmt_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _display(value: object) -> str:
    return "-" if value in (None, "") else str(value)


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    database_path = resolve_database_path(args.config, args.db)
    if not database_path.exists():
        raise SystemExit(f"Database not found: {database_path}")
    repository = SQLiteRepository(str(database_path))
    try:
        report = build_run_kpi_report(
            repository,
            service_name=str(args.service_name or config.service.lock_name),
            run_id=args.run_id,
            recent_rounds=int(args.recent_rounds),
        )
        payload = run_kpi_report_to_dict(report)
    finally:
        repository.close()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
