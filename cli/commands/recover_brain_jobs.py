from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime

from adapters.brain_api_adapter import ApiEndpointConfig, BrainApiAdapter
from core.config import AppConfig
from services.brain_service import BrainService
from services.models import CommandEnvironment
from services.session_manager import SessionManager
from storage.models import ServiceRuntimeRecord, SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository

_PENDING_STATUSES = {"submitted", "running"}


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "recover-brain-jobs",
        help="Poll specific stale BRAIN jobs directly without applying local timeout deadlines.",
        parents=[common],
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--batch-id",
        default=None,
        help="Recover all pending jobs that belong to the specified submission batch.",
    )
    target_group.add_argument(
        "--job-id",
        action="append",
        default=None,
        help="Recover one pending BRAIN job by id. Repeat the flag to recover multiple jobs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON document instead of a human-readable summary.",
    )
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    if config.brain.backend != "api":
        raise ValueError("recover-brain-jobs requires `brain.backend: api` in the active config.")

    run_id = environment.context.run_id
    target_submissions = _resolve_target_submissions(repository, run_id=run_id, args=args)
    if isinstance(target_submissions, str):
        if args.json:
            print(json.dumps({"run_id": run_id, "error": target_submissions}, ensure_ascii=False, indent=2))
        else:
            print(target_submissions)
        return 1

    adapter = _build_adapter(config)
    session_manager = SessionManager(
        adapter,
        persona_retry_interval_seconds=config.service.persona_retry_interval_seconds,
    )
    runtime = repository.service_runtime.get_state(config.service.lock_name) or _ephemeral_runtime(
        service_name=config.service.lock_name,
        run_id=run_id,
    )
    session_state = session_manager.ensure_session(runtime=runtime, allow_new_login=False)
    if session_state.status != "ready":
        payload = {
            "run_id": run_id,
            "status": "auth_not_ready",
            "auth_status": session_state.status,
            "detail": session_state.detail,
            "persona_url": session_state.persona_url,
            "job_ids": [submission.job_id for submission in target_submissions],
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"run_id: {run_id}")
            print("status: auth_not_ready")
            print(f"auth_status: {session_state.status}")
            if session_state.detail:
                print(f"detail: {session_state.detail}")
            if session_state.persona_url:
                print(f"persona_url: {session_state.persona_url}")
        return 1

    service = BrainService(repository, config.brain, adapter=adapter)
    recovered = service.recover_jobs(
        [submission.job_id for submission in target_submissions],
        config=config,
        environment=environment,
    )
    batches = {
        batch_id: repository.submissions.get_batch(batch_id)
        for batch_id in sorted({submission.batch_id for submission in recovered})
    }
    payload = {
        "run_id": run_id,
        "status": "recovered",
        "job_count": len(recovered),
        "submission_counts": dict(sorted(Counter(submission.status for submission in recovered).items())),
        "batches": [
            _batch_to_dict(batch)
            for batch in batches.values()
            if batch is not None
        ],
        "jobs": [asdict(submission) for submission in recovered],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human(payload)
    return 0


def _resolve_target_submissions(
    repository: SQLiteRepository,
    *,
    run_id: str,
    args: argparse.Namespace,
) -> list[SubmissionRecord] | str:
    if args.batch_id:
        batch = repository.submissions.get_batch(str(args.batch_id))
        if batch is None:
            return f"Unknown batch_id: {args.batch_id}"
        if batch.run_id != run_id:
            return f"Batch {batch.batch_id} belongs to run_id={batch.run_id}, not active run_id={run_id}."
        submissions = repository.submissions.list_submissions(
            run_id=run_id,
            batch_id=batch.batch_id,
            statuses=tuple(sorted(_PENDING_STATUSES)),
        )
        if not submissions:
            return f"Batch {batch.batch_id} has no pending submitted/running jobs to recover."
        return submissions

    requested_job_ids = list(dict.fromkeys(args.job_id or []))
    submissions: list[SubmissionRecord] = []
    for job_id in requested_job_ids:
        submission = repository.submissions.get_submission(job_id)
        if submission is None:
            return f"Unknown job_id: {job_id}"
        if submission.run_id != run_id:
            return f"Job {job_id} belongs to run_id={submission.run_id}, not active run_id={run_id}."
        if submission.status not in _PENDING_STATUSES:
            return f"Job {job_id} is not pending (status={submission.status})."
        submissions.append(submission)
    if not submissions:
        return "No job ids were provided."
    return submissions


def _build_adapter(config: AppConfig) -> BrainApiAdapter:
    return BrainApiAdapter(
        base_url=config.brain.api_base_url or "https://api.worldquantbrain.com",
        auth_env=config.brain.api_auth_env,
        auth_token=os.getenv(config.brain.api_auth_env),
        email_env=config.brain.email_env,
        password_env=config.brain.password_env,
        credentials_file=config.brain.credentials_file,
        session_path=config.brain.session_path,
        auth_expiry_seconds=config.brain.auth_expiry_seconds,
        open_browser_for_persona=config.brain.open_browser_for_persona,
        persona_poll_interval_seconds=config.brain.persona_poll_interval_seconds,
        persona_timeout_seconds=config.brain.persona_timeout_seconds,
        endpoints=ApiEndpointConfig(),
        max_retries=config.brain.max_retries,
        rate_limit_per_minute=config.brain.rate_limit_per_minute,
    )


def _ephemeral_runtime(*, service_name: str, run_id: str) -> ServiceRuntimeRecord:
    timestamp = datetime.now(UTC).isoformat()
    return ServiceRuntimeRecord(
        service_name=service_name,
        service_run_id=run_id,
        owner_token="recover-brain-jobs",
        pid=0,
        hostname="local",
        status="running",
        tick_id=0,
        active_batch_id=None,
        pending_job_count=0,
        consecutive_failures=0,
        cooldown_until=None,
        last_heartbeat_at=timestamp,
        last_success_at=None,
        last_error=None,
        persona_url=None,
        persona_wait_started_at=None,
        persona_last_notification_at=None,
        counters_json="{}",
        lock_expires_at=None,
        started_at=timestamp,
        updated_at=timestamp,
    )


def _batch_to_dict(batch: SubmissionBatchRecord) -> dict[str, object]:
    return {
        "batch_id": batch.batch_id,
        "status": batch.status,
        "candidate_count": batch.candidate_count,
        "service_status_reason": batch.service_status_reason,
        "updated_at": batch.updated_at,
    }


def _print_human(payload: dict[str, object]) -> None:
    print(f"run_id: {payload['run_id']}")
    print(f"status: {payload['status']}")
    print(f"job_count: {payload['job_count']}")
    submission_counts = payload.get("submission_counts") or {}
    if submission_counts:
        print("submission_counts: " + " ".join(f"{key}={value}" for key, value in submission_counts.items()))
    batches = payload.get("batches") or []
    if batches:
        print("batches:")
        for batch in batches:
            print(
                f"  batch_id={batch['batch_id']} status={batch['status']} "
                f"candidate_count={batch['candidate_count']}"
            )
    jobs = payload.get("jobs") or []
    if jobs:
        print("jobs:")
        for job in jobs:
            print(
                f"  job_id={job['job_id']} batch={job['batch_id']} status={job['status']} "
                f"updated_at={job['updated_at']}"
            )
