from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from core.brain_checks import parse_names_json
from domain.brain import BrainResultRecord
from storage.models import (
    RunRecord,
    ServiceDispatchQueueRecord,
    ServiceRuntimeRecord,
    SubmissionBatchRecord,
    SubmissionRecord,
)
from storage.repository import SQLiteRepository


@dataclass(slots=True)
class ServiceStatusSnapshot:
    service_name: str
    run_id: str | None
    runtime: ServiceRuntimeRecord | None
    run: RunRecord | None
    active_batch: SubmissionBatchRecord | None
    active_batch_submissions: list[SubmissionRecord]
    queue_depth: int
    queue_counts: dict[str, int]
    recent_queue_items: list[ServiceDispatchQueueRecord]
    recent_batches: list[SubmissionBatchRecord]
    recent_submissions: list[SubmissionRecord]
    recent_results: list[BrainResultRecord]
    batch_counts: dict[str, int]
    submission_counts: dict[str, int]
    result_counts: dict[str, int]
    active_batch_submission_counts: dict[str, int]
    derived_submit_ready_counts: dict[str, int]
    top_hard_fail_checks: dict[str, int]
    top_blocking_warning_checks: dict[str, int]
    stage_metrics: list[dict]
    duplicate_summary: list[dict]
    avg_crowding_penalty: float
    latest_regime_snapshot: dict[str, Any] | None


def build_service_status_snapshot(
    repository: SQLiteRepository,
    *,
    service_name: str,
    run_id: str | None,
    limit: int,
) -> ServiceStatusSnapshot:
    runtime = repository.service_runtime.get_state(service_name)
    resolved_run_id = run_id or _resolve_run_id(repository, runtime)
    run = repository.get_run(resolved_run_id) if resolved_run_id else None

    if not resolved_run_id:
        return ServiceStatusSnapshot(
            service_name=service_name,
            run_id=None,
            runtime=runtime,
            run=None,
            active_batch=None,
            active_batch_submissions=[],
            queue_depth=0,
            queue_counts={},
            recent_queue_items=[],
            recent_batches=[],
            recent_submissions=[],
            recent_results=[],
            batch_counts={},
            submission_counts={},
            result_counts={},
            active_batch_submission_counts={},
            derived_submit_ready_counts={},
            top_hard_fail_checks={},
            top_blocking_warning_checks={},
            stage_metrics=[],
            duplicate_summary=[],
            avg_crowding_penalty=0.0,
            latest_regime_snapshot=None,
        )

    batches = repository.submissions.list_batches(resolved_run_id)
    submissions = repository.submissions.list_submissions(run_id=resolved_run_id)
    results = repository.brain_results.list_results(run_id=resolved_run_id)
    queue_items = repository.service_dispatch_queue.list_items(
        service_name=service_name,
        run_id=resolved_run_id,
    )

    active_batch = _resolve_active_batch(repository, runtime=runtime, batches=batches)
    active_batch_submissions = (
        repository.submissions.list_submissions(run_id=resolved_run_id, batch_id=active_batch.batch_id)
        if active_batch is not None
        else []
    )

    return ServiceStatusSnapshot(
        service_name=service_name,
        run_id=resolved_run_id,
        runtime=runtime,
        run=run,
        active_batch=active_batch,
        active_batch_submissions=active_batch_submissions,
        queue_depth=sum(1 for item in queue_items if item.status in {"queued", "dispatching"}),
        queue_counts=_count_statuses(item.status for item in queue_items),
        recent_queue_items=list(
            sorted(
                queue_items,
                key=lambda item: (item.updated_at, item.queue_position, item.queue_item_id),
                reverse=True,
            )
        )[:limit],
        recent_batches=list(reversed(batches))[:limit],
        recent_submissions=list(reversed(submissions))[:limit],
        recent_results=list(reversed(results))[:limit],
        batch_counts=_count_statuses(batch.status for batch in batches),
        submission_counts=_count_statuses(submission.status for submission in submissions),
        result_counts=_count_statuses(result.status for result in results),
        active_batch_submission_counts=_count_statuses(
            submission.status for submission in active_batch_submissions
        ),
        derived_submit_ready_counts=_derived_submit_ready_counts(results),
        top_hard_fail_checks=_top_check_counts(results, "hard_fail_checks_json"),
        top_blocking_warning_checks=_top_check_counts(results, "blocking_warning_checks_json"),
        stage_metrics=repository.get_stage_metrics(resolved_run_id),
        duplicate_summary=repository.get_duplicate_decision_summary(resolved_run_id),
        avg_crowding_penalty=repository.get_average_crowding_penalty(resolved_run_id),
        latest_regime_snapshot=repository.get_latest_regime_snapshot(resolved_run_id),
    )


def service_status_snapshot_to_dict(snapshot: ServiceStatusSnapshot) -> dict[str, Any]:
    return {
        "service_name": snapshot.service_name,
        "run_id": snapshot.run_id,
        "service_runtime": asdict(snapshot.runtime) if snapshot.runtime is not None else None,
        "run": _run_to_dict(snapshot.run),
        "summary": {
            "batch_counts": snapshot.batch_counts,
            "submission_counts": snapshot.submission_counts,
            "result_counts": snapshot.result_counts,
            "active_batch_submission_counts": snapshot.active_batch_submission_counts,
            "derived_submit_ready_counts": snapshot.derived_submit_ready_counts,
            "top_hard_fail_checks": snapshot.top_hard_fail_checks,
            "top_blocking_warning_checks": snapshot.top_blocking_warning_checks,
            "queue_depth": snapshot.queue_depth,
            "queue_counts": snapshot.queue_counts,
            "avg_crowding_penalty": snapshot.avg_crowding_penalty,
        },
        "stage_metrics": snapshot.stage_metrics,
        "duplicate_summary": snapshot.duplicate_summary,
        "latest_regime_snapshot": snapshot.latest_regime_snapshot,
        "active_batch": _batch_to_dict(snapshot.active_batch),
        "active_batch_submissions": [_submission_to_dict(row) for row in snapshot.active_batch_submissions],
        "recent_queue_items": [_queue_item_to_dict(row) for row in snapshot.recent_queue_items],
        "recent_batches": [_batch_to_dict(row) for row in snapshot.recent_batches],
        "recent_submissions": [_submission_to_dict(row) for row in snapshot.recent_submissions],
        "recent_results": [_result_to_dict(row) for row in snapshot.recent_results],
    }


def _resolve_run_id(
    repository: SQLiteRepository,
    runtime: ServiceRuntimeRecord | None,
) -> str | None:
    if runtime is not None and runtime.service_run_id:
        return runtime.service_run_id
    latest = repository.get_latest_run()
    return latest.run_id if latest is not None else None


def _resolve_active_batch(
    repository: SQLiteRepository,
    *,
    runtime: ServiceRuntimeRecord | None,
    batches: list[SubmissionBatchRecord],
) -> SubmissionBatchRecord | None:
    if runtime is not None and runtime.active_batch_id:
        active = repository.submissions.get_batch(runtime.active_batch_id)
        if active is not None:
            return active

    live_statuses = {"submitting", "submitted", "running", "paused_quarantine"}
    for batch in reversed(batches):
        if batch.status in live_statuses:
            return batch
    return batches[-1] if batches else None


def _count_statuses(statuses: list[str] | tuple[str, ...] | Any) -> dict[str, int]:
    counts = Counter(str(status) for status in statuses if status)
    return {key: counts[key] for key in sorted(counts)}


def _derived_submit_ready_counts(results: list[BrainResultRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        if result.derived_submit_ready is True:
            counts["yes"] += 1
        elif result.derived_submit_ready is False:
            counts["no"] += 1
        else:
            counts["unknown"] += 1
    return {key: counts[key] for key in ("yes", "no", "unknown") if counts[key]}


def _top_check_counts(
    results: list[BrainResultRecord],
    attribute_name: str,
    *,
    limit: int = 10,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        for name in parse_names_json(getattr(result, attribute_name, "[]")):
            counts[name] += 1
    return dict(counts.most_common(limit))


def _run_to_dict(run: RunRecord | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "run_id": run.run_id,
        "status": run.status,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "profile_name": run.profile_name,
        "region": run.region,
        "selected_timeframe": run.selected_timeframe,
        "regime_key": run.regime_key,
        "global_regime_key": run.global_regime_key,
        "market_regime_key": run.market_regime_key,
        "effective_regime_key": run.effective_regime_key,
        "regime_label": run.regime_label,
        "regime_confidence": run.regime_confidence,
        "entry_command": run.entry_command,
    }


def _batch_to_dict(batch: SubmissionBatchRecord | None) -> dict[str, Any] | None:
    if batch is None:
        return None
    return {
        "batch_id": batch.batch_id,
        "run_id": batch.run_id,
        "round_index": batch.round_index,
        "backend": batch.backend,
        "status": batch.status,
        "candidate_count": batch.candidate_count,
        "created_at": batch.created_at,
        "updated_at": batch.updated_at,
        "service_status_reason": batch.service_status_reason,
        "last_polled_at": batch.last_polled_at,
        "quarantined_at": batch.quarantined_at,
    }


def _submission_to_dict(submission: SubmissionRecord) -> dict[str, Any]:
    return {
        "job_id": submission.job_id,
        "batch_id": submission.batch_id,
        "candidate_id": submission.candidate_id,
        "expression": submission.expression,
        "status": submission.status,
        "submitted_at": submission.submitted_at,
        "updated_at": submission.updated_at,
        "completed_at": submission.completed_at,
        "retry_count": submission.retry_count,
        "error_message": submission.error_message,
        "service_failure_reason": submission.service_failure_reason,
    }


def _queue_item_to_dict(item: ServiceDispatchQueueRecord) -> dict[str, Any]:
    return {
        "queue_item_id": item.queue_item_id,
        "candidate_id": item.candidate_id,
        "source_round_index": item.source_round_index,
        "queue_position": item.queue_position,
        "status": item.status,
        "batch_id": item.batch_id,
        "job_id": item.job_id,
        "failure_reason": item.failure_reason,
        "updated_at": item.updated_at,
    }


def _result_to_dict(result: BrainResultRecord) -> dict[str, Any]:
    return {
        "job_id": result.job_id,
        "batch_id": result.batch_id,
        "candidate_id": result.candidate_id,
        "expression": result.expression,
        "status": result.status,
        "fitness": result.fitness,
        "sharpe": result.sharpe,
        "turnover": result.turnover,
        "submission_eligible": result.submission_eligible,
        "derived_submit_ready": result.derived_submit_ready,
        "hard_fail_checks": result.hard_fail_checks_json,
        "warning_checks": result.warning_checks_json,
        "blocking_warning_checks": result.blocking_warning_checks_json,
        "rejection_reason": result.rejection_reason,
        "simulated_at": result.simulated_at,
        "created_at": result.created_at,
    }
