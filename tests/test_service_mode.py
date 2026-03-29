from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from adapters.brain_api_adapter import BrainApiAdapter, PersonaVerificationRequired
from core.config import load_config
from core.run_context import RunContext
from generator.engine import AlphaCandidate
from services.brain_service import BrainService
from services.models import CommandEnvironment
from services.notification_manager import NotificationManager
from services.runtime_lock import RuntimeLock
from services.service_runner import ServiceRunner
from services.service_worker import ServiceWorker
from services.session_manager import SessionManager
from storage.models import ServiceRuntimeRecord, SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository


class FakeApiAdapter(BrainApiAdapter):
    def __init__(
        self,
        *,
        auth_plan: list[dict | Exception] | None = None,
        status_plan: dict[str, list[dict | Exception]] | None = None,
        result_plan: dict[str, dict] | None = None,
    ) -> None:
        self.auth_plan = list(auth_plan or [{"mode": "session_cookie", "status": "ready", "session_path": "session.json"}])
        self.status_plan = {job_id: list(items) for job_id, items in (status_plan or {}).items()}
        self.result_plan = dict(result_plan or {})
        self.submit_calls: list[tuple[str, dict]] = []
        self.status_calls: list[str] = []
        self.result_calls: list[str] = []
        self.persona_notifications: list[str] = []

    def ensure_authenticated(self, **kwargs) -> dict:
        del kwargs
        item = self.auth_plan.pop(0) if self.auth_plan else {"mode": "session_cookie", "status": "ready"}
        if isinstance(item, Exception):
            raise item
        return {"status": "ready", "session_path": "session.json", **item}

    def send_persona_notification(self, persona_url: str) -> bool:
        self.persona_notifications.append(persona_url)
        return True

    def submit_simulation(self, expression: str, sim_config: dict) -> dict:
        self.submit_calls.append((expression, sim_config))
        payload = list(sim_config.get("candidate_payloads") or [{}])[0]
        return {
            "job_id": str(payload.get("job_id") or f"job-{len(self.submit_calls)}"),
            "status": "submitted",
            "expression": expression,
            "raw_submission": {"queued": True},
        }

    def get_simulation_status(self, job_id: str) -> dict:
        self.status_calls.append(job_id)
        plan = self.status_plan.get(job_id) or []
        item = plan.pop(0) if plan else {"job_id": job_id, "status": "completed"}
        self.status_plan[job_id] = plan
        if isinstance(item, Exception):
            raise item
        return item

    def get_simulation_result(self, job_id: str) -> dict:
        self.result_calls.append(job_id)
        return self.result_plan.get(
            job_id,
            {
                "job_id": job_id,
                "status": "completed",
                "raw_result": {
                    "metrics": {
                        "sharpe": 1.25,
                        "fitness": 1.05,
                        "turnover": 0.45,
                        "drawdown": 0.12,
                        "returns": 0.08,
                        "margin": 0.05,
                    },
                    "submission_eligible": True,
                },
            },
        )


def test_runtime_lock_prevents_duplicate_instances_and_reclaims_stale_lease() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        lock1 = RuntimeLock.create(repository.service_runtime, service_name="brain-service", service_run_id="run-1", lease_seconds=60)
        lock2 = RuntimeLock.create(repository.service_runtime, service_name="brain-service", service_run_id="run-2", lease_seconds=60)

        assert lock1.acquire(status="running") is True
        assert lock2.acquire(status="running") is False

        repository.service_runtime.update_state(
            "brain-service",
            lock_expires_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
            updated_at=datetime.now(UTC).isoformat(),
        )
        assert lock2.acquire(status="running") is True
    finally:
        repository.close()


def test_session_manager_reauthenticates_when_adapter_requires_new_login() -> None:
    adapter = FakeApiAdapter(
        auth_plan=[
            {"mode": "session_cookie", "status": "ready"},
            {"mode": "non_interactive", "status": "ready"},
        ]
    )
    manager = SessionManager(adapter)
    runtime = _runtime_record(run_id="run-auth")

    first = manager.ensure_session(runtime=runtime)
    second = manager.ensure_session(runtime=runtime)

    assert first.status == "ready"
    assert first.detail == "session_cookie"
    assert second.status == "ready"
    assert second.detail == "non_interactive"


def test_service_worker_waits_for_persona_and_throttles_notifications() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        environment = _environment("run-persona")
        runtime = _runtime_record(run_id="run-persona")
        repository.service_runtime.upsert_state(runtime)
        adapter = FakeApiAdapter(
            auth_plan=[
                PersonaVerificationRequired("https://persona.example/scan"),
                PersonaVerificationRequired("https://persona.example/scan"),
            ]
        )
        worker = ServiceWorker(
            repository,
            config=config,
            environment=environment,
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=999999),
        )

        first = worker.run_tick(runtime=runtime, tick_id=1)
        refreshed = repository.service_runtime.get_state(config.service.lock_name)
        assert refreshed is not None
        second = worker.run_tick(runtime=refreshed, tick_id=2)
    finally:
        repository.close()

    assert first.status == "waiting_persona"
    assert second.status == "waiting_persona"
    assert adapter.persona_notifications == ["https://persona.example/scan"]


def test_service_runner_resumes_pending_jobs_without_duplicate_submission() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        run_id = "run-resume"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(run_id, [_candidate("alpha-1", "rank(close)")])
        _seed_pending_batch(repository, run_id=run_id, batch_id="batch-1", job_id="job-1", status="submitted")
        repository.service_runtime.upsert_state(_runtime_record(run_id=run_id))
        adapter = FakeApiAdapter(status_plan={"job-1": [{"job_id": "job-1", "status": "completed"}]})
        runner = ServiceRunner(
            repository,
            config=config,
            environment=_environment("fresh-run-id"),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            sleep_fn=lambda seconds: None,
            install_signal_handlers=False,
        )

        summary = runner.run(max_ticks=1)
        result = repository.brain_results.get_result("job-1")
        submission = repository.submissions.get_submission("job-1")
    finally:
        repository.close()

    assert summary.run_id == run_id
    assert len(adapter.submit_calls) == 0
    assert result is not None and result.status == "completed"
    assert submission is not None and submission.status == "completed"


def test_service_worker_applies_backoff_after_transient_poll_failure() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        run_id = "run-backoff"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(run_id, [_candidate("alpha-1", "rank(close)")])
        _seed_pending_batch(repository, run_id=run_id, batch_id="batch-1", job_id="job-1", status="submitted")
        adapter = FakeApiAdapter(status_plan={"job-1": [RuntimeError("temporary network issue")]})
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1)
        submission = repository.submissions.get_submission("job-1")
    finally:
        repository.close()

    assert outcome.status == "running"
    assert submission is not None
    assert submission.retry_count == 1
    assert submission.next_poll_after is not None
    assert submission.service_failure_reason == "temporary network issue"


def test_service_runner_releases_lock_on_graceful_shutdown() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        run_id = "run-shutdown"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(run_id, [_candidate("alpha-1", "rank(close)")])
        _seed_pending_batch(repository, run_id=run_id, batch_id="batch-1", job_id="job-1", status="submitted")
        adapter = FakeApiAdapter(status_plan={"job-1": [{"job_id": "job-1", "status": "running"}]})
        runner = ServiceRunner(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            sleep_fn=lambda seconds: runner.request_shutdown(),
            install_signal_handlers=False,
        )

        summary = runner.run(max_ticks=2)
        runtime = repository.service_runtime.get_state(config.service.lock_name)
    finally:
        repository.close()

    assert summary.status == "service_stopped"
    assert runtime is not None
    assert runtime.owner_token == ""
    assert runtime.status == "service_stopped"


def test_service_quarantines_ambiguous_submitting_batch_without_duplicate_submission() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        run_id = "run-quarantine"
        _seed_run(repository, run_id)
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-ambiguous",
                run_id=run_id,
                round_index=1,
                backend="api",
                status="submitting",
                candidate_count=2,
                sim_config_snapshot=json.dumps({"region": "USA"}),
                export_path=None,
                notes_json="{}",
                created_at=_timestamp(),
                updated_at=_timestamp(),
            )
        )
        repository.submissions.upsert_submissions(
            [
                SubmissionRecord(
                    job_id="job-1",
                    batch_id="batch-ambiguous",
                    run_id=run_id,
                    round_index=1,
                    candidate_id="alpha-1",
                    expression="rank(close)",
                    backend="api",
                    status="submitted",
                    sim_config_snapshot=json.dumps({"region": "USA"}),
                    submitted_at=_timestamp(),
                    updated_at=_timestamp(),
                    completed_at=None,
                    export_path=None,
                    raw_submission_json="{}",
                    error_message=None,
                )
            ]
        )
        repository.service_runtime.upsert_state(_runtime_record(run_id=run_id))
        adapter = FakeApiAdapter()
        runner = ServiceRunner(
            repository,
            config=config,
            environment=_environment("fresh-run-id"),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            sleep_fn=lambda seconds: None,
            install_signal_handlers=False,
        )

        summary = runner.run(max_ticks=1)
        batch = repository.submissions.get_batch("batch-ambiguous")
    finally:
        repository.close()

    assert summary.status == "service_paused_quarantine"
    assert batch is not None
    assert batch.status == "paused_quarantine"
    assert batch.service_status_reason == "ambiguous_submission"
    assert len(adapter.submit_calls) == 0


def _service_config():
    config = load_config("config/dev.yaml")
    config.storage.path = ":memory:"
    config.brain.backend = "api"
    config.service.resume_incomplete_jobs = True
    config.service.max_consecutive_failures = 2
    config.service.cooldown_seconds = 60
    return config


def _environment(run_id: str) -> CommandEnvironment:
    return CommandEnvironment(
        config_path="config/dev.yaml",
        command_name="run-service",
        context=RunContext.create(seed=7, config_path="config/dev.yaml", run_id=run_id),
    )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _runtime_record(run_id: str) -> ServiceRuntimeRecord:
    now = _timestamp()
    return ServiceRuntimeRecord(
        service_name="brain-service",
        service_run_id=run_id,
        owner_token="",
        pid=1234,
        hostname="test-host",
        status="running",
        tick_id=0,
        active_batch_id=None,
        pending_job_count=0,
        consecutive_failures=0,
        cooldown_until=None,
        last_heartbeat_at=now,
        last_success_at=None,
        last_error=None,
        persona_url=None,
        persona_wait_started_at=None,
        persona_last_notification_at=None,
        counters_json="{}",
        lock_expires_at=None,
        started_at=now,
        updated_at=now,
    )


def _seed_run(repository: SQLiteRepository, run_id: str) -> None:
    repository.upsert_run(
        run_id=run_id,
        seed=7,
        config_path="config/dev.yaml",
        config_snapshot="{}",
        status="running_service",
        started_at=_timestamp(),
        entry_command="run-service",
    )
    repository.save_dataset_summary(run_id, summary={}, regime_key="service-regime")


def _seed_pending_batch(
    repository: SQLiteRepository,
    *,
    run_id: str,
    batch_id: str,
    job_id: str,
    status: str,
) -> None:
    repository.submissions.upsert_batch(
        SubmissionBatchRecord(
            batch_id=batch_id,
            run_id=run_id,
            round_index=1,
            backend="api",
            status="submitted",
            candidate_count=1,
            sim_config_snapshot=json.dumps(
                {
                    "region": "USA",
                    "universe": "TOP3000",
                    "delay": 1,
                    "neutralization": "sector",
                    "decay": 0,
                }
            ),
            export_path=None,
            notes_json="{}",
            created_at=_timestamp(),
            updated_at=_timestamp(),
        )
    )
    repository.submissions.upsert_submissions(
        [
            SubmissionRecord(
                job_id=job_id,
                batch_id=batch_id,
                run_id=run_id,
                round_index=1,
                candidate_id="alpha-1",
                expression="rank(close)",
                backend="api",
                status=status,
                sim_config_snapshot=json.dumps(
                    {
                        "region": "USA",
                        "universe": "TOP3000",
                        "delay": 1,
                        "neutralization": "sector",
                        "decay": 0,
                    }
                ),
                submitted_at=_timestamp(),
                updated_at=_timestamp(),
                completed_at=None,
                export_path=None,
                raw_submission_json="{}",
                error_message=None,
            )
        ]
    )


def _candidate(alpha_id: str, expression: str) -> AlphaCandidate:
    return AlphaCandidate(
        alpha_id=alpha_id,
        expression=expression,
        normalized_expression=expression,
        generation_mode="template",
        parent_ids=(),
        complexity=3,
        created_at=_timestamp(),
        template_name="momentum",
        fields_used=("close",),
        operators_used=("rank",),
        depth=2,
        generation_metadata={},
    )
