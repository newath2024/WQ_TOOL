from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from storage.models import (
    BrainResultRecord,
    RunRecord,
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
    recent_batches: list[SubmissionBatchRecord]
    recent_submissions: list[SubmissionRecord]
    recent_results: list[BrainResultRecord]
    batch_counts: dict[str, int]
    submission_counts: dict[str, int]
    result_counts: dict[str, int]
    active_batch_submission_counts: dict[str, int]


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
            recent_batches=[],
            recent_submissions=[],
            recent_results=[],
            batch_counts={},
            submission_counts={},
            result_counts={},
            active_batch_submission_counts={},
        )

    batches = repository.submissions.list_batches(resolved_run_id)
    submissions = repository.submissions.list_submissions(run_id=resolved_run_id)
    results = repository.brain_results.list_results(run_id=resolved_run_id)

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
        recent_batches=list(reversed(batches))[:limit],
        recent_submissions=list(reversed(submissions))[:limit],
        recent_results=list(reversed(results))[:limit],
        batch_counts=_count_statuses(batch.status for batch in batches),
        submission_counts=_count_statuses(submission.status for submission in submissions),
        result_counts=_count_statuses(result.status for result in results),
        active_batch_submission_counts=_count_statuses(
            submission.status for submission in active_batch_submissions
        ),
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
        },
        "active_batch": _batch_to_dict(snapshot.active_batch),
        "active_batch_submissions": [_submission_to_dict(row) for row in snapshot.active_batch_submissions],
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
        "rejection_reason": result.rejection_reason,
        "simulated_at": result.simulated_at,
        "created_at": result.created_at,
    }
