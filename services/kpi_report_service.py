from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import median
from typing import Any

from storage.models import RunRecord, ServiceRuntimeRecord
from storage.repository import SQLiteRepository

_SQLITE_IN_CLAUSE_CHUNK = 500


@dataclass(slots=True)
class RunKpiReport:
    run_id: str | None
    service_name: str
    run: RunRecord | None
    runtime: ServiceRuntimeRecord | None
    latest_round: int | None
    scope_round_start: int | None
    scope_round_end: int | None
    scope_round_count: int
    scope_label: str
    health: dict[str, Any]
    funnel: dict[str, Any]
    quality: dict[str, Any]
    meta_model: dict[str, Any]
    regime: dict[str, Any]
    mutation: dict[str, Any]
    recent: dict[str, Any]
    baseline: dict[str, Any]
    delta_flags: dict[str, Any]
    timeout_reasons: dict[str, Any]
    generation_fail_reasons: dict[str, Any]


def build_run_kpi_report(
    repository: SQLiteRepository,
    *,
    service_name: str,
    run_id: str | None = None,
    recent_rounds: int = 20,
) -> RunKpiReport:
    resolved_run_id = run_id or _resolve_run_id(repository)
    runtime = _resolve_runtime(repository.connection, service_name=service_name, run_id=resolved_run_id)
    if resolved_run_id is None:
        return RunKpiReport(
            run_id=None,
            service_name=service_name,
            run=None,
            runtime=runtime,
            latest_round=None,
            scope_round_start=None,
            scope_round_end=None,
            scope_round_count=0,
            scope_label="no_run",
            health={},
            funnel={},
            quality={},
            meta_model={},
            regime={},
            mutation={},
            recent={},
            baseline={},
            delta_flags={},
            timeout_reasons={},
            generation_fail_reasons={},
        )
    run = repository.get_run(resolved_run_id)
    scope = _resolve_round_scope(repository.connection, run_id=resolved_run_id, recent_rounds=recent_rounds)
    submissions = repository.submissions.list_submissions(run_id=resolved_run_id)
    results = repository.brain_results.list_results(run_id=resolved_run_id)
    stage_metrics = repository.get_stage_metrics(resolved_run_id)
    selection_scores = repository.list_selection_scores(resolved_run_id, score_stage="pre_sim")
    regime_rows = _fetch_regime_rows(repository.connection, resolved_run_id, scope)
    all_closed_loop_rows = _fetch_all_closed_loop_rows(repository.connection, resolved_run_id)
    closed_loop_rows = [row for row in all_closed_loop_rows if _round_in_scope(int(row.get("round_index") or 0), scope)]
    mutation_rows = _fetch_mutation_rows(repository.connection, resolved_run_id)

    scoped_submissions = [row for row in submissions if _round_in_scope(row.round_index, scope)]
    scoped_results = [row for row in results if _round_in_scope(row.round_index, scope)]
    scoped_stage_metrics = [row for row in stage_metrics if _round_in_scope(int(row.get("round_index") or 0), scope)]
    scoped_selection_scores = [
        row for row in selection_scores if _round_in_scope(int(row.get("round_index") or 0), scope)
    ]
    outcome_by_alpha = _fetch_alpha_outcomes(
        repository.connection,
        run_id=resolved_run_id,
        alpha_ids={str(row.get("alpha_id") or "") for row in scoped_selection_scores if str(row.get("alpha_id") or "")},
    )
    recent, baseline, delta_flags, timeout_reasons, generation_fail_reasons = _build_recent_vs_baseline(
        submissions=submissions,
        results=results,
        stage_metrics=stage_metrics,
        closed_loop_rows=all_closed_loop_rows,
    )

    return RunKpiReport(
        run_id=resolved_run_id,
        service_name=service_name,
        run=run,
        runtime=runtime,
        latest_round=scope["latest_round"],
        scope_round_start=scope["start"],
        scope_round_end=scope["end"],
        scope_round_count=scope["count"],
        scope_label=scope["label"],
        health=_build_health(runtime=runtime, submissions=scoped_submissions),
        funnel=_build_funnel(closed_loop_rows=closed_loop_rows, selection_scores=scoped_selection_scores, stage_metrics=scoped_stage_metrics),
        quality=_build_quality(submissions=scoped_submissions, results=scoped_results),
        meta_model=_build_meta_model(selection_scores=scoped_selection_scores, outcome_by_alpha=outcome_by_alpha),
        regime=_build_regime(regime_rows=regime_rows),
        mutation=_build_mutation(closed_loop_rows=closed_loop_rows, mutation_rows=mutation_rows),
        recent=recent,
        baseline=baseline,
        delta_flags=delta_flags,
        timeout_reasons=timeout_reasons,
        generation_fail_reasons=generation_fail_reasons,
    )


def run_kpi_report_to_dict(report: RunKpiReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "service_name": report.service_name,
        "run": asdict(report.run) if report.run is not None else None,
        "runtime": asdict(report.runtime) if report.runtime is not None else None,
        "scope": {
            "label": report.scope_label,
            "latest_round": report.latest_round,
            "scope_round_start": report.scope_round_start,
            "scope_round_end": report.scope_round_end,
            "scope_round_count": report.scope_round_count,
        },
        "health": report.health,
        "funnel": report.funnel,
        "quality": report.quality,
        "meta_model": report.meta_model,
        "regime": report.regime,
        "mutation": report.mutation,
        "recent": report.recent,
        "baseline": report.baseline,
        "delta_flags": report.delta_flags,
        "timeout_reasons": report.timeout_reasons,
        "generation_fail_reasons": report.generation_fail_reasons,
    }


def _resolve_run_id(repository: SQLiteRepository) -> str | None:
    latest = repository.get_latest_run()
    return latest.run_id if latest is not None else None


def _resolve_runtime(
    connection: sqlite3.Connection,
    *,
    service_name: str,
    run_id: str | None,
) -> ServiceRuntimeRecord | None:
    if run_id:
        row = connection.execute(
            """
            SELECT *
            FROM service_runtime
            WHERE service_name = ? AND service_run_id = ?
            LIMIT 1
            """,
            (service_name, run_id),
        ).fetchone()
        return ServiceRuntimeRecord(**dict(row)) if row is not None else None
    row = connection.execute(
        """
        SELECT *
        FROM service_runtime
        WHERE service_name = ?
        LIMIT 1
        """,
        (service_name,),
    ).fetchone()
    return ServiceRuntimeRecord(**dict(row)) if row is not None else None


def _resolve_round_scope(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    recent_rounds: int,
) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT round_index
        FROM closed_loop_rounds
        WHERE run_id = ?
        UNION
        SELECT round_index
        FROM submission_batches
        WHERE run_id = ?
        UNION
        SELECT round_index
        FROM alpha_selection_scores
        WHERE run_id = ? AND score_stage = 'pre_sim'
        ORDER BY round_index ASC
        """,
        (run_id, run_id, run_id),
    ).fetchall()
    rounds = [int(row["round_index"]) for row in rows]
    if not rounds:
        return {
            "latest_round": None,
            "start": None,
            "end": None,
            "count": 0,
            "label": "full_run",
            "rounds": set(),
        }
    latest_round = rounds[-1]
    use_full_run = recent_rounds <= 0
    scoped = rounds if use_full_run else rounds[-int(recent_rounds) :]
    return {
        "latest_round": latest_round,
        "start": scoped[0],
        "end": scoped[-1],
        "count": len(scoped),
        "label": "full_run" if use_full_run else f"last_{len(scoped)}_rounds",
        "rounds": set(scoped),
    }


def _fetch_regime_rows(connection: sqlite3.Connection, run_id: str, scope: dict[str, Any]) -> list[dict[str, Any]]:
    if scope["start"] is None or scope["end"] is None:
        return []
    rows = connection.execute(
        """
        SELECT *
        FROM regime_snapshots
        WHERE run_id = ? AND round_index >= ? AND round_index <= ?
        ORDER BY round_index ASC
        """,
        (run_id, scope["start"], scope["end"]),
    ).fetchall()
    return [dict(row) for row in rows]


def _fetch_closed_loop_rows(connection: sqlite3.Connection, run_id: str, scope: dict[str, Any]) -> list[dict[str, Any]]:
    if scope["start"] is None or scope["end"] is None:
        return []
    rows = connection.execute(
        """
        SELECT *
        FROM closed_loop_rounds
        WHERE run_id = ? AND round_index >= ? AND round_index <= ?
        ORDER BY round_index ASC
        """,
        (run_id, scope["start"], scope["end"]),
    ).fetchall()
    return [dict(row) for row in rows]


def _fetch_all_closed_loop_rows(connection: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT *
        FROM closed_loop_rounds
        WHERE run_id = ?
        ORDER BY round_index ASC
        """,
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _fetch_mutation_rows(connection: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT *
        FROM mutation_outcomes
        WHERE run_id = ?
        ORDER BY created_at DESC
        """,
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _fetch_alpha_outcomes(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    alpha_ids: set[str],
) -> dict[str, float]:
    normalized = sorted(alpha_id for alpha_id in alpha_ids if alpha_id)
    if not normalized:
        return {}
    outcomes: dict[str, float] = {}
    for chunk in _chunked(normalized, size=_SQLITE_IN_CLAUSE_CHUNK):
        placeholders = ", ".join("?" for _ in chunk)
        rows = connection.execute(
            f"""
            SELECT alpha_id, outcome_score
            FROM alpha_cases
            WHERE run_id = ?
              AND metric_source = 'external_brain'
              AND alpha_id IN ({placeholders})
            """,
            (run_id, *chunk),
        ).fetchall()
        outcomes.update({str(row["alpha_id"]): float(row["outcome_score"] or 0.0) for row in rows})
    return outcomes


def _build_health(
    *,
    runtime: ServiceRuntimeRecord | None,
    submissions,
) -> dict[str, Any]:
    status_counts = Counter(str(row.status or "") for row in submissions)
    timeout_reason_counts = _timeout_reason_counts_for_submissions(submissions)
    terminal_statuses = {"completed", "failed", "rejected", "timeout"}
    terminal = [row for row in submissions if str(row.status or "") in terminal_statuses or row.completed_at]
    terminal_count = len(terminal)
    latencies = [
        _latency_seconds(row.submitted_at, row.completed_at)
        for row in terminal
        if _latency_seconds(row.submitted_at, row.completed_at) is not None
    ]
    completed_count = int(status_counts.get("completed", 0))
    timeout_count = int(status_counts.get("timeout", 0))
    failed_count = int(status_counts.get("failed", 0) + status_counts.get("rejected", 0))
    return {
        "submitted_jobs": len(submissions),
        "terminal_jobs": terminal_count,
        "completed_jobs": completed_count,
        "timeout_jobs": timeout_count,
        "failed_jobs": failed_count,
        "completed_rate": _safe_ratio(completed_count, terminal_count),
        "timeout_rate": _safe_ratio(timeout_count, terminal_count),
        "failed_rate": _safe_ratio(failed_count, terminal_count),
        "poll_timeout_live_jobs": int(timeout_reason_counts.get("poll_timeout_live", 0)),
        "poll_timeout_after_downtime_jobs": int(timeout_reason_counts.get("poll_timeout_after_downtime", 0)),
        "legacy_poll_timeout_jobs": int(timeout_reason_counts.get("poll_timeout", 0)),
        "other_timeout_jobs": int(timeout_reason_counts.get("other_timeout", 0)),
        "timeout_reason_counts": dict(timeout_reason_counts),
        "median_latency_sec": float(median(latencies)) if latencies else None,
        "avg_latency_sec": float(sum(latencies) / len(latencies)) if latencies else None,
        "pending_jobs_runtime": int(runtime.pending_job_count) if runtime is not None else 0,
        "service_status": runtime.status if runtime is not None else "unknown",
    }


def _build_funnel(
    *,
    closed_loop_rows: list[dict[str, Any]],
    selection_scores: list[dict[str, Any]],
    stage_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    generated = sum(int(row.get("generated_count") or 0) for row in closed_loop_rows)
    validated = sum(int(row.get("validated_count") or 0) for row in closed_loop_rows)
    pre_sim_selected = sum(1 for row in selection_scores if bool(row.get("selected")) and row.get("score_stage") == "pre_sim")
    validate_fail_count = 0
    attempt_count = 0
    validation_disallowed_field_count = 0
    blocked_by_near_duplicate = 0
    for row in stage_metrics:
        metrics = _decode_json_object(row.get("metrics_json"))
        if row.get("stage") == "generation":
            validate_fail_count += int(metrics.get("validate_fail_count") or 0)
            attempt_count += int(metrics.get("attempt_count") or 0)
            validation_disallowed_field_count += int(
                _decode_json_object(metrics.get("failure_reason_counts")).get("validation_disallowed_field", 0)
            )
        elif row.get("stage") == "pre_sim":
            blocked_by_near_duplicate += int(metrics.get("blocked_by_near_duplicate") or 0)
    return {
        "generated_count": generated,
        "validated_count": validated,
        "selected_for_simulation": pre_sim_selected,
        "validation_rate": _safe_ratio(validated, generated),
        "selection_rate": _safe_ratio(pre_sim_selected, generated),
        "validate_fail_count": validate_fail_count,
        "attempt_count": attempt_count,
        "validation_disallowed_field_count": validation_disallowed_field_count,
        "validation_disallowed_field_rate": _safe_ratio(validation_disallowed_field_count, attempt_count),
        "blocked_by_near_duplicate": blocked_by_near_duplicate,
        "blocked_by_near_duplicate_rate": _safe_ratio(blocked_by_near_duplicate, generated),
    }


def _build_quality(
    *,
    submissions,
    results,
) -> dict[str, Any]:
    completed_results = [row for row in results if row.status == "completed"]
    fitnesses = [float(row.fitness) for row in completed_results if row.fitness is not None]
    sharpes = [float(row.sharpe) for row in completed_results if row.sharpe is not None]
    drawdowns = [float(row.drawdown) for row in completed_results if row.drawdown is not None]
    returns = [float(row.returns) for row in completed_results if row.returns is not None]
    candidate_ids = {str(row.candidate_id) for row in submissions if str(row.candidate_id or "")}
    return {
        "completed_results": len(completed_results),
        "distinct_candidates": len(candidate_ids),
        "positive_fitness_rate": _safe_ratio(sum(1 for value in fitnesses if value > 0.0), len(fitnesses)),
        "positive_sharpe_rate": _safe_ratio(sum(1 for value in sharpes if value > 0.0), len(sharpes)),
        "avg_fitness": float(sum(fitnesses) / len(fitnesses)) if fitnesses else None,
        "median_fitness": float(median(fitnesses)) if fitnesses else None,
        "avg_sharpe": float(sum(sharpes) / len(sharpes)) if sharpes else None,
        "median_sharpe": float(median(sharpes)) if sharpes else None,
        "avg_drawdown": float(sum(drawdowns) / len(drawdowns)) if drawdowns else None,
        "median_drawdown": float(median(drawdowns)) if drawdowns else None,
        "avg_returns": float(sum(returns) / len(returns)) if returns else None,
        "median_returns": float(median(returns)) if returns else None,
        "max_fitness": max(fitnesses) if fitnesses else None,
        "max_sharpe": max(sharpes) if sharpes else None,
    }


def _build_meta_model(
    *,
    selection_scores: list[dict[str, Any]],
    outcome_by_alpha: dict[str, float],
) -> dict[str, Any]:
    rows = []
    for row in selection_scores:
        breakdown = _decode_json_object(row.get("breakdown_json"))
        components = _decode_json_object(breakdown.get("components"))
        if not components:
            continue
        alpha_id = str(row.get("alpha_id") or "")
        rows.append(
            {
                "alpha_id": alpha_id,
                "selected": bool(row.get("selected")),
                "meta_model_used": bool(components.get("meta_model_used", 0.0)),
                "ml_positive_outcome_prob": _to_float(components.get("ml_positive_outcome_prob")),
                "heuristic_predicted_quality": _to_float(components.get("heuristic_predicted_quality")),
                "blended_predicted_quality": _to_float(
                    components.get("blended_predicted_quality", components.get("predicted_quality"))
                ),
                "train_rows": int(components.get("meta_model_train_rows") or 0),
                "positive_rows": int(components.get("meta_model_positive_rows") or 0),
                "outcome_score": outcome_by_alpha.get(alpha_id),
            }
        )
    if not rows:
        return {}
    selected_rows = [row for row in rows if row["selected"]]
    archived_rows = [row for row in rows if not row["selected"]]
    used_rows = [row for row in rows if row["meta_model_used"]]
    return {
        "rows": len(rows),
        "meta_model_used_rate": _safe_ratio(len(used_rows), len(rows)),
        "avg_train_rows": _avg([float(row["train_rows"]) for row in used_rows]),
        "avg_positive_rows": _avg([float(row["positive_rows"]) for row in used_rows]),
        "avg_selected_prob": _avg([row["ml_positive_outcome_prob"] for row in selected_rows if row["ml_positive_outcome_prob"] is not None]),
        "avg_archived_prob": _avg([row["ml_positive_outcome_prob"] for row in archived_rows if row["ml_positive_outcome_prob"] is not None]),
        "avg_selected_blended_score": _avg([row["blended_predicted_quality"] for row in selected_rows if row["blended_predicted_quality"] is not None]),
        "avg_archived_blended_score": _avg([row["blended_predicted_quality"] for row in archived_rows if row["blended_predicted_quality"] is not None]),
        "selected_positive_outcome_rate": _safe_ratio(
            sum(1 for row in selected_rows if row["outcome_score"] is not None and row["outcome_score"] > 0.0),
            sum(1 for row in selected_rows if row["outcome_score"] is not None),
        ),
        "archived_positive_outcome_rate": _safe_ratio(
            sum(1 for row in archived_rows if row["outcome_score"] is not None and row["outcome_score"] > 0.0),
            sum(1 for row in archived_rows if row["outcome_score"] is not None),
        ),
    }


def _build_regime(*, regime_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not regime_rows:
        return {}
    latest = regime_rows[-1]
    latest_features = _decode_json_object(latest.get("features_json"))
    learned_rows = [row for row in regime_rows if str(row.get("market_regime_key") or "").startswith("learned_cluster:")]
    learned_confidences = [
        _to_float(_decode_json_object(row.get("features_json")).get("learned_confidence"))
        for row in regime_rows
    ]
    learned_confidences = [value for value in learned_confidences if value is not None]
    return {
        "latest_market_regime_key": latest.get("market_regime_key"),
        "latest_regime_label": latest.get("regime_label"),
        "latest_confidence": _to_float(latest.get("confidence")),
        "latest_learned_cluster_id": latest_features.get("learned_cluster_id"),
        "latest_learned_confidence": _to_float(latest_features.get("learned_confidence")),
        "learned_active_rate": _safe_ratio(len(learned_rows), len(regime_rows)),
        "fallback_rate": _safe_ratio(len(regime_rows) - len(learned_rows), len(regime_rows)),
        "avg_learned_confidence": _avg(learned_confidences),
    }


def _build_mutation(
    *,
    closed_loop_rows: list[dict[str, Any]],
    mutation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    better_rows = [row for row in mutation_rows if float(row.get("outcome_delta") or 0.0) > 0.0]
    return {
        "selected_for_mutation_count": sum(int(row.get("selected_for_mutation_count") or 0) for row in closed_loop_rows),
        "mutated_children_count": sum(int(row.get("mutated_children_count") or 0) for row in closed_loop_rows),
        "child_better_than_parent_rate": _safe_ratio(len(better_rows), len(mutation_rows)),
        "mutation_outcome_rows": len(mutation_rows),
    }


def _build_recent_vs_baseline(
    *,
    submissions,
    results,
    stage_metrics: list[dict[str, Any]],
    closed_loop_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    recent: dict[str, Any] = {}
    baseline: dict[str, Any] = {}

    terminal_statuses = {"completed", "failed", "rejected", "timeout"}
    terminal_submissions = [
        row for row in submissions if str(row.status or "") in terminal_statuses or getattr(row, "completed_at", None)
    ]
    raw_recent_submissions, raw_baseline_submissions = _split_recent_and_baseline(list(terminal_submissions), size=200)
    recent["raw_results"] = _build_window_metrics(
        label="recent_raw_results",
        results=_results_for_submissions(results, raw_recent_submissions),
        submissions=raw_recent_submissions,
        stage_metrics=_stage_metrics_for_submissions(stage_metrics, raw_recent_submissions),
        closed_loop_rows=_closed_loop_rows_for_submissions(closed_loop_rows, raw_recent_submissions),
    )
    baseline["raw_results"] = _build_window_metrics(
        label="baseline_raw_results",
        results=_results_for_submissions(results, raw_baseline_submissions),
        submissions=raw_baseline_submissions,
        stage_metrics=_stage_metrics_for_submissions(stage_metrics, raw_baseline_submissions),
        closed_loop_rows=_closed_loop_rows_for_submissions(closed_loop_rows, raw_baseline_submissions),
    )

    completed_results = [row for row in results if str(row.status or "") == "completed"]
    recent_completed, baseline_completed = _split_recent_and_baseline(completed_results, size=200)
    recent["completed_results"] = _build_window_metrics(
        label="recent_completed_results",
        results=recent_completed,
        submissions=_submissions_for_results(submissions, recent_completed),
        stage_metrics=_stage_metrics_for_results(stage_metrics, recent_completed),
        closed_loop_rows=_closed_loop_rows_for_results(closed_loop_rows, recent_completed),
    )
    baseline["completed_results"] = _build_window_metrics(
        label="baseline_completed_results",
        results=baseline_completed,
        submissions=_submissions_for_results(submissions, baseline_completed),
        stage_metrics=_stage_metrics_for_results(stage_metrics, baseline_completed),
        closed_loop_rows=_closed_loop_rows_for_results(closed_loop_rows, baseline_completed),
    )

    recent_round_rows, baseline_round_rows = _split_recent_and_baseline(closed_loop_rows, size=7)
    recent_round_numbers = {int(row.get("round_index") or 0) for row in recent_round_rows}
    baseline_round_numbers = {int(row.get("round_index") or 0) for row in baseline_round_rows}
    recent["rounds"] = _build_window_metrics(
        label="recent_rounds",
        results=_filter_results_by_rounds(results, recent_round_numbers),
        submissions=_filter_submissions_by_rounds(submissions, recent_round_numbers),
        stage_metrics=_filter_stage_metrics_by_rounds(stage_metrics, recent_round_numbers),
        closed_loop_rows=recent_round_rows,
    )
    baseline["rounds"] = _build_window_metrics(
        label="baseline_rounds",
        results=_filter_results_by_rounds(results, baseline_round_numbers),
        submissions=_filter_submissions_by_rounds(submissions, baseline_round_numbers),
        stage_metrics=_filter_stage_metrics_by_rounds(stage_metrics, baseline_round_numbers),
        closed_loop_rows=baseline_round_rows,
    )

    delta_flags = {
        key: _classify_window_delta(recent.get(key, {}), baseline.get(key, {}))
        for key in ("raw_results", "completed_results", "rounds")
    }
    timeout_reasons = {
        "recent": dict(recent.get("raw_results", {}).get("top_timeout_reasons") or {}),
        "baseline": dict(baseline.get("raw_results", {}).get("top_timeout_reasons") or {}),
    }
    generation_fail_reasons = {
        "recent": dict(recent.get("rounds", {}).get("top_generation_fail_reasons") or {}),
        "baseline": dict(baseline.get("rounds", {}).get("top_generation_fail_reasons") or {}),
    }
    return recent, baseline, delta_flags, timeout_reasons, generation_fail_reasons


def _build_window_metrics(
    *,
    label: str,
    results,
    submissions,
    stage_metrics: list[dict[str, Any]],
    closed_loop_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    completed_results = [row for row in results if str(row.status or "") == "completed"]
    fitnesses = [float(row.fitness) for row in completed_results if row.fitness is not None]
    sharpes = [float(row.sharpe) for row in completed_results if row.sharpe is not None]
    drawdowns = [float(row.drawdown) for row in completed_results if row.drawdown is not None]
    returns = [float(row.returns) for row in completed_results if row.returns is not None]
    timeout_reason_counts = _timeout_reason_counts_for_submissions(submissions)
    generation_fail_reasons = _aggregate_generation_fail_reasons(stage_metrics)
    generated_count = sum(int(row.get("generated_count") or 0) for row in closed_loop_rows)
    if generated_count <= 0:
        generated_count = sum(
            int(_decode_json_object(row.get("metrics_json")).get("generated") or 0)
            for row in stage_metrics
            if row.get("stage") == "generation"
        )
    selected_for_simulation = sum(
        int(_decode_json_object(row.get("metrics_json")).get("selected_for_simulation") or 0)
        for row in stage_metrics
        if row.get("stage") == "generation"
    )
    blocked_by_near_duplicate = sum(
        int(_decode_json_object(row.get("metrics_json")).get("blocked_by_near_duplicate") or 0)
        for row in stage_metrics
        if row.get("stage") == "pre_sim"
    )
    attempt_count = sum(
        int(_decode_json_object(row.get("metrics_json")).get("attempt_count") or 0)
        for row in stage_metrics
        if row.get("stage") == "generation"
    )
    validation_disallowed_field_count = sum(
        int(
            _decode_json_object(
                _decode_json_object(row.get("metrics_json")).get("failure_reason_counts")
            ).get("validation_disallowed_field", 0)
        )
        for row in stage_metrics
        if row.get("stage") == "generation"
    )
    terminal_statuses = {"completed", "failed", "rejected", "timeout"}
    terminal_submissions = [
        row for row in submissions if str(row.status or "") in terminal_statuses or getattr(row, "completed_at", None)
    ]
    timeout_jobs = sum(1 for row in submissions if str(row.status or "") == "timeout")
    completed_jobs = sum(1 for row in submissions if str(row.status or "") == "completed")
    round_indices = sorted({int(row.get("round_index") or 0) for row in closed_loop_rows if int(row.get("round_index") or 0) > 0})
    if not round_indices:
        round_indices = sorted({int(getattr(row, "round_index", 0) or 0) for row in results if int(getattr(row, "round_index", 0) or 0) > 0})
    return {
        "label": label,
        "result_count": len(results),
        "submission_count": len(submissions),
        "round_count": len(round_indices),
        "round_start": round_indices[0] if round_indices else None,
        "round_end": round_indices[-1] if round_indices else None,
        "completed_rate": _safe_ratio(completed_jobs, len(terminal_submissions)),
        "timeout_rate": _safe_ratio(timeout_jobs, len(terminal_submissions)),
        "avg_sharpe": _avg(sharpes),
        "median_sharpe": float(median(sharpes)) if sharpes else None,
        "avg_fitness": _avg(fitnesses),
        "median_fitness": float(median(fitnesses)) if fitnesses else None,
        "positive_sharpe_rate": _safe_ratio(sum(1 for value in sharpes if value > 0.0), len(sharpes)),
        "positive_fitness_rate": _safe_ratio(sum(1 for value in fitnesses if value > 0.0), len(fitnesses)),
        "avg_drawdown": _avg(drawdowns),
        "median_drawdown": float(median(drawdowns)) if drawdowns else None,
        "avg_returns": _avg(returns),
        "median_returns": float(median(returns)) if returns else None,
        "generated_count": generated_count,
        "selected_for_simulation": selected_for_simulation,
        "attempt_count": attempt_count,
        "validation_disallowed_field_count": validation_disallowed_field_count,
        "validation_disallowed_field_rate": _safe_ratio(validation_disallowed_field_count, attempt_count),
        "blocked_by_near_duplicate": blocked_by_near_duplicate,
        "blocked_by_near_duplicate_rate": _safe_ratio(blocked_by_near_duplicate, generated_count),
        "selected_for_simulation_rate": _safe_ratio(selected_for_simulation, generated_count),
        "top_timeout_reasons": dict(timeout_reason_counts.most_common(5)),
        "top_generation_fail_reasons": dict(generation_fail_reasons.most_common(5)),
    }


def _classify_window_delta(recent_window: dict[str, Any], baseline_window: dict[str, Any]) -> dict[str, Any]:
    avg_fitness_delta = _delta(recent_window.get("avg_fitness"), baseline_window.get("avg_fitness"))
    positive_fitness_rate_delta = _delta(
        recent_window.get("positive_fitness_rate"),
        baseline_window.get("positive_fitness_rate"),
    )
    timeout_rate_delta = _delta(recent_window.get("timeout_rate"), baseline_window.get("timeout_rate"))
    quality = "flat"
    if (avg_fitness_delta is not None and avg_fitness_delta >= 0.02) or (
        positive_fitness_rate_delta is not None and positive_fitness_rate_delta >= 0.05
    ):
        quality = "better"
    elif (avg_fitness_delta is not None and avg_fitness_delta <= -0.02) or (
        positive_fitness_rate_delta is not None and positive_fitness_rate_delta <= -0.05
    ):
        quality = "worse"
    operations = "flat"
    if timeout_rate_delta is not None and timeout_rate_delta >= 0.10:
        operations = "worse"
    elif timeout_rate_delta is not None and timeout_rate_delta <= -0.10:
        operations = "better"
    return {
        "quality": quality,
        "operations": operations,
        "avg_fitness_delta": avg_fitness_delta,
        "positive_fitness_rate_delta": positive_fitness_rate_delta,
        "timeout_rate_delta": timeout_rate_delta,
    }


def _split_recent_and_baseline(rows: list, *, size: int) -> tuple[list, list]:
    if size <= 0 or not rows:
        return list(rows), []
    recent = list(rows[-size:])
    baseline_end = max(0, len(rows) - size)
    baseline_start = max(0, baseline_end - size)
    baseline = list(rows[baseline_start:baseline_end])
    return recent, baseline


def _submissions_for_results(submissions, results) -> list:
    job_ids = {str(getattr(row, "job_id", "") or "") for row in results if str(getattr(row, "job_id", "") or "")}
    return [row for row in submissions if str(getattr(row, "job_id", "") or "") in job_ids]


def _results_for_submissions(results, submissions) -> list:
    job_ids = {str(getattr(row, "job_id", "") or "") for row in submissions if str(getattr(row, "job_id", "") or "")}
    return [row for row in results if str(getattr(row, "job_id", "") or "") in job_ids]


def _closed_loop_rows_for_results(closed_loop_rows: list[dict[str, Any]], results) -> list[dict[str, Any]]:
    round_numbers = {int(getattr(row, "round_index", 0) or 0) for row in results if int(getattr(row, "round_index", 0) or 0) > 0}
    return [row for row in closed_loop_rows if int(row.get("round_index") or 0) in round_numbers]


def _closed_loop_rows_for_submissions(closed_loop_rows: list[dict[str, Any]], submissions) -> list[dict[str, Any]]:
    round_numbers = {
        int(getattr(row, "round_index", 0) or 0)
        for row in submissions
        if int(getattr(row, "round_index", 0) or 0) > 0
    }
    return [row for row in closed_loop_rows if int(row.get("round_index") or 0) in round_numbers]


def _stage_metrics_for_results(stage_metrics: list[dict[str, Any]], results) -> list[dict[str, Any]]:
    round_numbers = {int(getattr(row, "round_index", 0) or 0) for row in results if int(getattr(row, "round_index", 0) or 0) > 0}
    return _filter_stage_metrics_by_rounds(stage_metrics, round_numbers)


def _stage_metrics_for_submissions(stage_metrics: list[dict[str, Any]], submissions) -> list[dict[str, Any]]:
    round_numbers = {
        int(getattr(row, "round_index", 0) or 0)
        for row in submissions
        if int(getattr(row, "round_index", 0) or 0) > 0
    }
    return _filter_stage_metrics_by_rounds(stage_metrics, round_numbers)


def _filter_results_by_rounds(results, round_numbers: set[int]) -> list:
    if not round_numbers:
        return []
    return [row for row in results if int(getattr(row, "round_index", 0) or 0) in round_numbers]


def _filter_submissions_by_rounds(submissions, round_numbers: set[int]) -> list:
    if not round_numbers:
        return []
    return [row for row in submissions if int(getattr(row, "round_index", 0) or 0) in round_numbers]


def _filter_stage_metrics_by_rounds(stage_metrics: list[dict[str, Any]], round_numbers: set[int]) -> list[dict[str, Any]]:
    if not round_numbers:
        return []
    return [row for row in stage_metrics if int(row.get("round_index") or 0) in round_numbers]


def _timeout_reason_counts_for_submissions(submissions) -> Counter[str]:
    timeout_reason_counts: Counter[str] = Counter()
    for row in submissions:
        if str(row.status or "") != "timeout":
            continue
        reason = str(row.service_failure_reason or row.error_message or "").strip()
        if reason == "poll_timeout_live":
            timeout_reason_counts["poll_timeout_live"] += 1
        elif reason == "poll_timeout_after_downtime":
            timeout_reason_counts["poll_timeout_after_downtime"] += 1
        elif reason == "poll_timeout":
            timeout_reason_counts["poll_timeout"] += 1
        else:
            timeout_reason_counts["other_timeout"] += 1
    return timeout_reason_counts


def _aggregate_generation_fail_reasons(stage_metrics: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in stage_metrics:
        if row.get("stage") != "generation":
            continue
        metrics = _decode_json_object(row.get("metrics_json"))
        failure_reason_counts = _decode_json_object(metrics.get("failure_reason_counts"))
        if failure_reason_counts:
            counts.update({str(key): int(value or 0) for key, value in failure_reason_counts.items()})
            continue
        top_fail_reasons = _decode_json_object(metrics.get("top_fail_reasons"))
        counts.update({str(key): int(value or 0) for key, value in top_fail_reasons.items()})
    return counts


def _delta(current: Any, previous: Any) -> float | None:
    current_value = _to_float(current)
    previous_value = _to_float(previous)
    if current_value is None or previous_value is None:
        return None
    return float(current_value - previous_value)


def _round_in_scope(round_index: int, scope: dict[str, Any]) -> bool:
    rounds = scope.get("rounds") or set()
    return not rounds or int(round_index) in rounds


def _latency_seconds(submitted_at: str | None, completed_at: str | None) -> float | None:
    if not submitted_at or not completed_at:
        return None
    try:
        start = datetime.fromisoformat(str(submitted_at))
        end = datetime.fromisoformat(str(completed_at))
    except ValueError:
        return None
    return max(0.0, (end - start).total_seconds())


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _chunked(values: list[str], *, size: int) -> list[list[str]]:
    if size <= 0:
        return [values]
    return [values[index : index + size] for index in range(0, len(values), size)]


def _decode_json_object(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str) and payload.strip():
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
