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
        )
    run = repository.get_run(resolved_run_id)
    scope = _resolve_round_scope(repository.connection, run_id=resolved_run_id, recent_rounds=recent_rounds)
    submissions = repository.submissions.list_submissions(run_id=resolved_run_id)
    results = repository.brain_results.list_results(run_id=resolved_run_id)
    stage_metrics = repository.get_stage_metrics(resolved_run_id)
    selection_scores = repository.list_selection_scores(resolved_run_id, score_stage="pre_sim")
    regime_rows = _fetch_regime_rows(repository.connection, resolved_run_id, scope)
    closed_loop_rows = _fetch_closed_loop_rows(repository.connection, resolved_run_id, scope)
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
    timeout_reason_counts: Counter[str] = Counter()
    terminal_statuses = {"completed", "failed", "rejected", "timeout"}
    terminal = [row for row in submissions if str(row.status or "") in terminal_statuses or row.completed_at]
    terminal_count = len(terminal)
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
    for row in stage_metrics:
        if row.get("stage") != "generation":
            continue
        metrics = _decode_json_object(row.get("metrics_json"))
        validate_fail_count += int(metrics.get("validate_fail_count") or 0)
    return {
        "generated_count": generated,
        "validated_count": validated,
        "selected_for_simulation": pre_sim_selected,
        "validation_rate": _safe_ratio(validated, generated),
        "selection_rate": _safe_ratio(pre_sim_selected, generated),
        "validate_fail_count": validate_fail_count,
    }


def _build_quality(
    *,
    submissions,
    results,
) -> dict[str, Any]:
    completed_results = [row for row in results if row.status == "completed"]
    fitnesses = [float(row.fitness) for row in completed_results if row.fitness is not None]
    sharpes = [float(row.sharpe) for row in completed_results if row.sharpe is not None]
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
