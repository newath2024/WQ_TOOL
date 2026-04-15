from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from adapters.brain_api_adapter import (
    BiometricsThrottled,
    BrainApiAdapter,
    ConcurrentSimulationLimitExceeded,
    PersonaVerificationRequired,
)
from core.config import load_config
from core.run_context import RunContext
from generator.engine import AlphaCandidate
from memory.case_memory import ObjectiveVector
from memory.pattern_memory import PatternMemoryService
from services.brain_service import BrainService
from services.models import BatchPreparationResult, CandidateScore, CommandEnvironment, ServiceTickOutcome
from services.notification_manager import NotificationManager
from services.runtime_lock import RuntimeLock
from services.service_scheduler import ServiceScheduler
from services.service_runner import ServiceRunner
from services.service_worker import ServiceWorker
from services.session_manager import SessionManager
from storage.models import BrainResultRecord, ServiceRuntimeRecord, SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository


class FakeApiAdapter(BrainApiAdapter):
    def __init__(
        self,
        *,
        auth_plan: list[dict | Exception] | None = None,
        persona_resume_plan: list[dict | Exception] | None = None,
        persona_confirmation_supported: bool = True,
        persona_confirmation_plan: list[dict | Exception] | None = None,
        status_plan: dict[str, list[dict | Exception]] | None = None,
        result_plan: dict[str, dict] | None = None,
    ) -> None:
        self.auth_plan = list(auth_plan or [{"mode": "session_cookie", "status": "ready", "session_path": "session.json"}])
        self.persona_resume_plan = list(persona_resume_plan or [])
        self.persona_confirmation_supported = persona_confirmation_supported
        self.persona_confirmation_plan = list(persona_confirmation_plan or [])
        self.status_plan = {job_id: list(items) for job_id, items in (status_plan or {}).items()}
        self.result_plan = dict(result_plan or {})
        self.submit_calls: list[tuple[str, dict]] = []
        self.status_calls: list[str] = []
        self.result_calls: list[str] = []
        self.persona_notifications: list[str] = []
        self.persona_confirmation_prompts: list[tuple[str, str]] = []
        self.persona_confirmation_polls: list[tuple[str, int | None]] = []
        self.auth_calls = 0
        self.persona_resume_calls: list[str] = []
        self.persona_timeout_seconds = 1800

    def ensure_authenticated(self, **kwargs) -> dict:
        del kwargs
        self.auth_calls += 1
        item = self.auth_plan.pop(0) if self.auth_plan else {"mode": "session_cookie", "status": "ready"}
        if isinstance(item, Exception):
            raise item
        return {"status": "ready", "session_path": "session.json", **item}

    def probe_authenticated_session(self) -> dict:
        return {"authenticated": False, "mode": "not_authenticated", "session_path": "session.json"}

    def resume_persona_authentication(self, persona_url: str) -> dict:
        self.persona_resume_calls.append(persona_url)
        item = (
            self.persona_resume_plan.pop(0)
            if self.persona_resume_plan
            else {"status": "waiting_persona", "persona_url": persona_url, "detail": "pending", "expired": False}
        )
        if isinstance(item, Exception):
            raise item
        payload = dict(item)
        payload.setdefault("status", "waiting_persona")
        payload.setdefault("persona_url", persona_url)
        payload.setdefault("detail", "pending")
        payload.setdefault("expired", False)
        return payload

    def send_persona_notification(self, persona_url: str) -> bool:
        self.persona_notifications.append(persona_url)
        return True

    def supports_persona_confirmation(self) -> bool:
        return self.persona_confirmation_supported

    def send_persona_confirmation_prompt(self, *, prompt_token: str, service_name: str) -> bool:
        self.persona_confirmation_prompts.append((prompt_token, service_name))
        return self.persona_confirmation_supported

    def poll_persona_confirmation(self, *, prompt_token: str, last_update_id: int | None = None) -> dict:
        self.persona_confirmation_polls.append((prompt_token, last_update_id))
        item = (
            self.persona_confirmation_plan.pop(0)
            if self.persona_confirmation_plan
            else {"supported": self.persona_confirmation_supported, "approved": False, "declined": False}
        )
        if isinstance(item, Exception):
            raise item
        payload = dict(item)
        payload.setdefault("supported", self.persona_confirmation_supported)
        payload.setdefault("approved", False)
        payload.setdefault("declined", False)
        payload.setdefault("last_update_id", last_update_id)
        return payload

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


class FakeBatchPreparationService:
    def __init__(self, result: BatchPreparationResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def prepare_service_batch(self, **kwargs) -> BatchPreparationResult:
        self.calls.append(dict(kwargs))
        return self.result


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
    manager = SessionManager(adapter, persona_retry_interval_seconds=30)
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
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
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


def test_service_worker_waits_for_telegram_confirmation_before_requesting_persona_link() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.persona_confirmation_required = True
        config.service.persona_confirmation_prompt_cooldown_seconds = 999999
        environment = _environment("run-persona-confirmation")
        runtime = _runtime_record(run_id="run-persona-confirmation")
        repository.service_runtime.upsert_state(runtime)
        adapter = FakeApiAdapter(
            auth_plan=[PersonaVerificationRequired("https://persona.example/scan-confirmed")],
            persona_confirmation_plan=[{"approved": False, "last_update_id": 101}],
        )
        worker = ServiceWorker(
            repository,
            config=config,
            environment=environment,
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(
                adapter,
                persona_email_cooldown_seconds=999999,
                persona_confirmation_required=True,
            ),
        )

        outcome = worker.run_tick(runtime=runtime, tick_id=1)
        refreshed = repository.service_runtime.get_state(config.service.lock_name)
    finally:
        repository.close()

    assert outcome.status == "waiting_persona_confirmation"
    assert adapter.auth_calls == 0
    assert len(adapter.persona_confirmation_prompts) == 1
    assert adapter.persona_notifications == []
    assert refreshed is not None
    assert refreshed.persona_confirmation_nonce
    assert refreshed.persona_url is None


def test_service_worker_requests_persona_link_only_after_telegram_confirmation() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.persona_confirmation_required = True
        config.service.persona_confirmation_prompt_cooldown_seconds = 999999
        environment = _environment("run-persona-approved")
        runtime = _runtime_record(run_id="run-persona-approved")
        repository.service_runtime.upsert_state(runtime)
        adapter = FakeApiAdapter(
            auth_plan=[PersonaVerificationRequired("https://persona.example/scan-confirmed")],
            persona_confirmation_plan=[
                {"approved": False, "last_update_id": 201},
                {"approved": True, "last_update_id": 202},
            ],
        )
        worker = ServiceWorker(
            repository,
            config=config,
            environment=environment,
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(
                adapter,
                persona_email_cooldown_seconds=999999,
                persona_confirmation_required=True,
            ),
        )

        first = worker.run_tick(runtime=runtime, tick_id=1)
        refreshed = repository.service_runtime.get_state(config.service.lock_name)
        assert refreshed is not None
        second = worker.run_tick(runtime=refreshed, tick_id=2)
        final_runtime = repository.service_runtime.get_state(config.service.lock_name)
    finally:
        repository.close()

    assert first.status == "waiting_persona_confirmation"
    assert second.status == "waiting_persona"
    assert adapter.auth_calls == 1
    assert len(adapter.persona_confirmation_prompts) == 1
    assert adapter.persona_notifications == ["https://persona.example/scan-confirmed"]
    assert final_runtime is not None
    assert final_runtime.persona_url == "https://persona.example/scan-confirmed"
    assert final_runtime.persona_confirmation_nonce is None
    assert final_runtime.persona_confirmation_granted_at is None


def test_notification_manager_does_not_resend_same_persona_url_once_delivered() -> None:
    adapter = FakeApiAdapter()
    manager = NotificationManager(adapter, persona_email_cooldown_seconds=1)
    runtime = replace(
        _runtime_record(run_id="run-persona-no-resend"),
        persona_url="https://persona.example/scan-same",
        persona_last_notification_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
    )

    sent, _ = manager.notify_persona_required(
        runtime=runtime,
        persona_url="https://persona.example/scan-same",
    )

    assert sent is False
    assert adapter.persona_notifications == []


def test_notification_manager_requires_fresh_confirmation_for_new_persona_url_without_active_grant() -> None:
    adapter = FakeApiAdapter()
    manager = NotificationManager(
        adapter,
        persona_email_cooldown_seconds=999999,
        persona_confirmation_required=True,
    )
    runtime = replace(
        _runtime_record(run_id="run-persona-gate"),
        persona_url="https://persona.example/scan-old",
    )

    assert manager.requires_fresh_persona_confirmation_for_url(
        runtime=runtime,
        persona_url="https://persona.example/scan-new",
        now=datetime.now(UTC).isoformat(),
    )
    assert not manager.requires_fresh_persona_confirmation_for_url(
        runtime=runtime,
        persona_url="https://persona.example/scan-old",
        now=datetime.now(UTC).isoformat(),
    )


def test_service_worker_resends_notification_when_persona_url_changes() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        environment = _environment("run-persona-refresh")
        runtime = _runtime_record(run_id="run-persona-refresh")
        runtime = replace(
            runtime,
            persona_url="https://persona.example/scan-old",
            persona_last_notification_at=_timestamp(),
        )
        repository.service_runtime.upsert_state(runtime)
        adapter = FakeApiAdapter(
            auth_plan=[
                PersonaVerificationRequired("https://persona.example/scan-new"),
            ]
        )
        worker = ServiceWorker(
            repository,
            config=config,
            environment=environment,
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=999999),
        )

        outcome = worker.run_tick(runtime=runtime, tick_id=1)
        refreshed = repository.service_runtime.get_state(config.service.lock_name)
    finally:
        repository.close()

    assert outcome.status == "waiting_persona"
    assert adapter.persona_notifications == ["https://persona.example/scan-new"]
    assert refreshed is not None
    assert refreshed.persona_url == "https://persona.example/scan-new"


def test_service_worker_reuses_existing_persona_inquiry_before_requesting_new_one() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        environment = _environment("run-persona-reuse")
        runtime = replace(
            _runtime_record(run_id="run-persona-reuse"),
            status="waiting_persona",
            persona_url="https://persona.example/scan-existing",
            persona_wait_started_at=(datetime.now(UTC) - timedelta(seconds=5)).isoformat(),
            persona_last_notification_at=_timestamp(),
        )
        repository.service_runtime.upsert_state(runtime)
        adapter = FakeApiAdapter(
            auth_plan=[PersonaVerificationRequired("https://persona.example/scan-new")],
        )
        worker = ServiceWorker(
            repository,
            config=config,
            environment=environment,
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=999999),
        )

        outcome = worker.run_tick(runtime=runtime, tick_id=1)
    finally:
        repository.close()

    assert outcome.status == "waiting_persona"
    assert outcome.persona_url == "https://persona.example/scan-existing"
    assert adapter.auth_calls == 0
    assert adapter.persona_resume_calls == []
    assert adapter.persona_notifications == []


def test_service_worker_resumes_existing_persona_inquiry_after_face_scan() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        run_id = "run-persona-complete"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(run_id, [_candidate("alpha-1", "rank(close)")])
        runtime = replace(
            _runtime_record(run_id=run_id),
            status="waiting_persona",
            persona_url="https://persona.example/scan-existing",
            persona_wait_started_at=(datetime.now(UTC) - timedelta(seconds=45)).isoformat(),
            persona_last_notification_at=_timestamp(),
        )
        repository.service_runtime.upsert_state(runtime)
        adapter = FakeApiAdapter(
            persona_resume_plan=[{"status": "ready", "mode": "session_cookie", "session_path": "session.json"}]
        )
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=999999),
        )

        outcome = worker.run_tick(runtime=runtime, tick_id=1)
        refreshed = repository.service_runtime.get_state(config.service.lock_name)
    finally:
        repository.close()

    assert outcome.status in {"idle", "no_candidates", "running", "submitting"}
    assert adapter.auth_calls == 0
    assert adapter.persona_resume_calls == ["https://persona.example/scan-existing"]
    assert refreshed is not None
    assert refreshed.persona_url is None
    assert refreshed.persona_wait_started_at is None
    assert refreshed.persona_last_notification_at is None


def test_session_manager_reports_biometrics_throttle_without_failing_service() -> None:
    adapter = FakeApiAdapter(auth_plan=[BiometricsThrottled("BIOMETRICS_THROTTLED", retry_after_seconds=45)])
    manager = SessionManager(adapter, persona_retry_interval_seconds=30)

    state = manager.ensure_session(runtime=_runtime_record(run_id="run-throttled"))

    assert state.status == "auth_throttled"
    assert state.retry_after_seconds == 45


def test_session_manager_reports_auth_transport_error_without_failing_service() -> None:
    adapter = FakeApiAdapter(
        auth_plan=[requests.ConnectionError("Remote end closed connection without response")]
    )
    manager = SessionManager(adapter, persona_retry_interval_seconds=30)

    state = manager.ensure_session(runtime=_runtime_record(run_id="run-auth-unavailable"))

    assert state.status == "auth_unavailable"
    assert state.detail is not None
    assert "Remote end closed connection without response" in state.detail


def test_service_worker_waits_when_biometrics_are_throttled() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        environment = _environment("run-throttled")
        runtime = replace(
            _runtime_record(run_id="run-throttled"),
            persona_url="https://persona.example/stale",
            persona_wait_started_at=_timestamp(),
            persona_last_notification_at=_timestamp(),
        )
        repository.service_runtime.upsert_state(runtime)
        adapter = FakeApiAdapter(auth_plan=[BiometricsThrottled("BIOMETRICS_THROTTLED", retry_after_seconds=45)])
        worker = ServiceWorker(
            repository,
            config=config,
            environment=environment,
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=runtime, tick_id=1)
        refreshed = repository.service_runtime.get_state(config.service.lock_name)
    finally:
        repository.close()

    assert outcome.status == "auth_throttled"
    assert outcome.next_sleep_seconds == 45
    assert outcome.persona_url is None
    assert refreshed is not None
    assert refreshed.status == "auth_throttled"
    assert refreshed.persona_url is None
    assert refreshed.persona_wait_started_at is not None


def test_service_worker_retries_when_auth_transport_error_occurs() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        environment = _environment("run-auth-unavailable")
        runtime = replace(
            _runtime_record(run_id="run-auth-unavailable"),
            persona_url="https://persona.example/stale",
            persona_wait_started_at=_timestamp(),
            persona_last_notification_at=_timestamp(),
        )
        repository.service_runtime.upsert_state(runtime)
        adapter = FakeApiAdapter(
            auth_plan=[requests.ConnectionError("Remote end closed connection without response")]
        )
        worker = ServiceWorker(
            repository,
            config=config,
            environment=environment,
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=runtime, tick_id=1)
        refreshed = repository.service_runtime.get_state(config.service.lock_name)
    finally:
        repository.close()

    assert outcome.status == "auth_unavailable"
    assert outcome.next_sleep_seconds == config.service.tick_interval_seconds
    assert outcome.persona_url is None
    assert refreshed is not None
    assert refreshed.status == "auth_unavailable"
    assert refreshed.persona_url is None
    assert refreshed.last_error is not None
    assert "Remote end closed connection without response" in refreshed.last_error


def test_service_worker_rechecks_auth_during_auth_related_cooldown() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.max_pending_jobs = 0
        run_id = "run-auth-cooldown"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(run_id, [_candidate("alpha-1", "rank(close)")])
        _seed_pending_batch(repository, run_id=run_id, batch_id="batch-1", job_id="job-1", status="submitted")
        adapter = FakeApiAdapter(
            auth_plan=[{"mode": "session_cookie", "status": "ready"}],
            status_plan={"job-1": [{"job_id": "job-1", "status": "completed"}]},
        )
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )
        now = datetime.now(UTC)
        runtime = replace(
            _runtime_record(run_id=run_id),
            status="cooldown",
            cooldown_until=(now + timedelta(seconds=60)).isoformat(),
            last_error='BRAIN authentication failed with status 429: {"detail":"BIOMETRICS_THROTTLED"}',
        )

        outcome = worker.run_tick(runtime=runtime, tick_id=1)
    finally:
        repository.close()

    assert outcome.status == "idle"
    assert outcome.completed_count == 1


def test_service_worker_polls_submitting_batch_during_submission_cooldown() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        run_id = "run-submit-cooldown-poll"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(run_id, [_candidate("alpha-1", "rank(close)")])
        _seed_pending_batch(repository, run_id=run_id, batch_id="batch-1", job_id="job-1", status="submitted")
        repository.submissions.update_batch_status(
            "batch-1",
            status="submitting",
            updated_at=_timestamp(),
            service_status_reason="submission_failed:ConcurrentSimulationLimitExceeded",
        )
        adapter = FakeApiAdapter(status_plan={"job-1": [{"job_id": "job-1", "status": "completed"}]})
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )
        now = datetime.now(UTC)
        runtime = replace(
            _runtime_record(run_id=run_id),
            status="cooldown",
            cooldown_until=(now + timedelta(seconds=60)).isoformat(),
            last_error="BRAIN concurrent simulation limit exceeded: CONCURRENT_SIMULATION_LIMIT_EXCEEDED",
        )

        outcome = worker.run_tick(runtime=runtime, tick_id=1)
        batch = repository.submissions.get_batch("batch-1")
        result = repository.brain_results.get_result("job-1")
    finally:
        repository.close()

    assert outcome.status == "cooldown"
    assert outcome.completed_count == 1
    assert outcome.pending_job_count == 0
    assert outcome.cooldown_until == runtime.cooldown_until
    assert batch is not None
    assert batch.status == "completed"
    assert result is not None
    assert result.status == "completed"


def test_service_runner_resumes_pending_jobs_without_duplicate_submission() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.max_pending_jobs = 0
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


def test_service_runner_writes_progress_log_jsonl(tmp_path: Path) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.max_pending_jobs = 0
        config.runtime.progress_log_dir = str(tmp_path / "progress")
        run_id = "run-progress-log"
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
        log_path = tmp_path / "progress" / f"{run_id}.jsonl"
        progress_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    finally:
        repository.close()

    assert summary.progress_log_path == str(log_path)
    assert any(row["event"] == "service_run_started" for row in progress_rows)
    assert any(row["event"] == "service_tick_started" for row in progress_rows)
    assert any(row["event"] == "batch_polled" and row["batch_id"] == "batch-1" for row in progress_rows)
    assert any(row["event"] == "service_tick_completed" and row["status"] == "idle" for row in progress_rows)
    assert any(row["event"] == "service_run_finished" for row in progress_rows)


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
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
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


def test_service_worker_uses_normal_submission_count_without_failure_history() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 3
        config.service.max_pending_jobs = 5
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-adaptive-normal"
        _seed_run(repository, run_id)
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=FakeBatchPreparationService(_batch_result(_service_candidates(3))),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker._submit_new_batch(runtime=_runtime_record(run_id=run_id), tick_id=1)
    finally:
        repository.close()

    assert outcome.submitted_count == 3
    assert len(adapter.submit_calls) == 3


def test_service_worker_limits_submission_count_to_available_pending_slots() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 4
        config.service.max_pending_jobs = 5
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-available-slots"
        _seed_run(repository, run_id)
        for index in range(1, 5):
            _seed_pending_batch(
                repository,
                run_id=run_id,
                batch_id=f"batch-{index}",
                job_id=f"job-{index}",
                status="submitted",
            )
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=FakeBatchPreparationService(_batch_result(_service_candidates(4))),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker._submit_new_batch(runtime=_runtime_record(run_id=run_id), tick_id=1)
    finally:
        repository.close()

    assert outcome.submitted_count == 1
    assert outcome.pending_job_count == 5
    assert len(adapter.submit_calls) == 1


def test_service_worker_respects_available_slots_even_with_extreme_fail_history() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 4
        config.service.max_pending_jobs = 5
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-available-slots-fail-history"
        _seed_run(repository, run_id)
        for index in range(1, 5):
            _seed_pending_batch(
                repository,
                run_id=run_id,
                batch_id=f"batch-{index}",
                job_id=f"job-{index}",
                status="submitted",
            )
        _seed_completed_batch(
            repository,
            run_id=run_id,
            batch_id="batch-a",
            statuses=("failed", "failed", "failed"),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        _seed_completed_batch(
            repository,
            run_id=run_id,
            batch_id="batch-b",
            statuses=("failed", "failed", "timeout"),
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=FakeBatchPreparationService(_batch_result(_service_candidates(4))),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker._submit_new_batch(runtime=_runtime_record(run_id=run_id), tick_id=1)
    finally:
        repository.close()

    assert outcome.submitted_count == 1
    assert outcome.pending_job_count == 5
    assert len(adapter.submit_calls) == 1


def test_service_worker_caps_submission_count_when_recent_fail_rate_exceeds_half() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 4
        config.service.max_pending_jobs = 5
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-adaptive-half"
        _seed_run(repository, run_id)
        _seed_completed_batch(
            repository,
            run_id=run_id,
            batch_id="batch-a",
            statuses=("failed", "failed", "failed"),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        _seed_completed_batch(
            repository,
            run_id=run_id,
            batch_id="batch-b",
            statuses=("completed", "completed", "failed"),
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=FakeBatchPreparationService(_batch_result(_service_candidates(4))),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker._submit_new_batch(runtime=_runtime_record(run_id=run_id), tick_id=1)
    finally:
        repository.close()

    assert outcome.submitted_count == 3
    assert len(adapter.submit_calls) == 3


def test_service_worker_caps_submission_count_at_one_when_recent_fail_rate_is_extreme() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 4
        config.service.max_pending_jobs = 5
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-adaptive-high"
        _seed_run(repository, run_id)
        _seed_completed_batch(
            repository,
            run_id=run_id,
            batch_id="batch-a",
            statuses=("failed", "failed", "failed"),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        _seed_completed_batch(
            repository,
            run_id=run_id,
            batch_id="batch-b",
            statuses=("failed", "failed", "timeout"),
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=FakeBatchPreparationService(_batch_result(_service_candidates(4))),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker._submit_new_batch(runtime=_runtime_record(run_id=run_id), tick_id=1)
    finally:
        repository.close()

    assert outcome.submitted_count == 3
    assert len(adapter.submit_calls) == 3


def test_service_worker_keeps_existing_normal_batch_cap_when_history_is_clean() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 2
        config.service.max_pending_jobs = 5
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-adaptive-cap"
        _seed_run(repository, run_id)
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=FakeBatchPreparationService(_batch_result(_service_candidates(4))),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker._submit_new_batch(runtime=_runtime_record(run_id=run_id), tick_id=1)
    finally:
        repository.close()

    assert outcome.submitted_count == 2
    assert len(adapter.submit_calls) == 2


def test_service_worker_skips_new_submission_when_pending_capacity_is_full() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 4
        config.service.max_pending_jobs = 2
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-capacity-full"
        _seed_run(repository, run_id)
        for index in range(1, 3):
            _seed_pending_batch(
                repository,
                run_id=run_id,
                batch_id=f"batch-{index}",
                job_id=f"job-{index}",
                status="submitted",
            )
        batch_service = FakeBatchPreparationService(_batch_result(_service_candidates(4)))
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=batch_service,
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker._submit_new_batch(runtime=_runtime_record(run_id=run_id), tick_id=1)
    finally:
        repository.close()

    assert outcome.status == "running"
    assert outcome.submitted_count == 0
    assert outcome.pending_job_count == 2
    assert outcome.active_batch_id == "batch-1"
    assert len(batch_service.calls) == 0
    assert len(adapter.submit_calls) == 0


def test_service_worker_does_not_top_up_while_pending_batches_exist() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 2
        config.service.max_pending_jobs = 2
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-top-up-after-poll"
        _seed_run(repository, run_id)
        _seed_pending_batch(repository, run_id=run_id, batch_id="batch-1", job_id="job-1", status="submitted")
        adapter = FakeApiAdapter(status_plan={"job-1": [{"job_id": "job-1", "status": "running"}]})
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=FakeBatchPreparationService(_batch_result(_service_candidates(2))),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1)
        pending = repository.submissions.list_pending_submissions(run_id)
    finally:
        repository.close()

    assert outcome.status == "running"
    assert outcome.new_result_count == 0
    assert outcome.submitted_count == 0
    assert outcome.pending_job_count == 1
    assert len(pending) == 1
    assert len(adapter.submit_calls) == 0


def test_service_worker_submits_on_next_idle_tick_after_pending_batch_completes() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 2
        config.service.max_pending_jobs = 2
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-idle-after-poll"
        _seed_run(repository, run_id)
        _seed_pending_batch(repository, run_id=run_id, batch_id="batch-1", job_id="job-1", status="submitted")
        batch_service = FakeBatchPreparationService(_batch_result(_service_candidates(2)))
        adapter = FakeApiAdapter(status_plan={"job-1": [{"job_id": "job-1", "status": "completed"}]})
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=batch_service,
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        first = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1)
        second = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=2)
        pending = repository.submissions.list_pending_submissions(run_id)
    finally:
        repository.close()

    assert first.status == "idle"
    assert first.new_result_count == 1
    assert first.submitted_count == 0
    assert len(batch_service.calls) == 1
    assert second.status == "running"
    assert second.submitted_count == 2
    assert second.pending_job_count == 2
    assert len(pending) == 2
    assert len(adapter.submit_calls) == 2


def test_service_worker_syncs_closed_loop_rounds_for_service_batches() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 1
        config.service.max_pending_jobs = 1
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-service-round-sync"
        _seed_run(repository, run_id)
        batch_service = FakeBatchPreparationService(_batch_result(_service_candidates(1)))
        adapter = FakeApiAdapter(status_plan={"job-1": [{"job_id": "job-1", "status": "completed"}]})
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=batch_service,
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        submitted = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1)
        submitted_round = repository.brain_results.get_closed_loop_round(run_id, 1)
        polled = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=2)
        completed_round = repository.brain_results.get_closed_loop_round(run_id, 1)
    finally:
        repository.close()

    assert submitted.status == "running"
    assert submitted_round is not None
    assert submitted_round.status == "submitted"
    assert submitted_round.generated_count == 1
    assert submitted_round.validated_count == 1
    assert submitted_round.submitted_count == 1
    assert submitted_round.completed_count == 0
    assert json.loads(submitted_round.summary_json)["source"] == "service"
    assert polled.status == "idle"
    assert completed_round is not None
    assert completed_round.status == "completed"
    assert completed_round.completed_count == 1
    assert json.loads(completed_round.summary_json)["terminal_submission_count"] == 1


def test_service_worker_skips_new_submission_when_stop_requested() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 2
        config.service.max_pending_jobs = 2
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-stop-requested"
        _seed_run(repository, run_id)
        batch_service = FakeBatchPreparationService(_batch_result(_service_candidates(2)))
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=batch_service,
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1, stop_requested=True)
        pending = repository.submissions.list_pending_submissions(run_id)
    finally:
        repository.close()

    assert outcome.status == "idle"
    assert outcome.submitted_count == 0
    assert outcome.pending_job_count == 0
    assert len(pending) == 0
    assert len(batch_service.calls) == 0
    assert len(adapter.submit_calls) == 0


def test_service_worker_extends_pending_timeout_deadline_while_waiting_for_confirmation() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.persona_confirmation_required = True
        config.service.persona_confirmation_poll_interval_seconds = 45
        run_id = "run-wait-extends-timeout"
        _seed_run(repository, run_id)
        _seed_pending_batch(repository, run_id=run_id, batch_id="batch-1", job_id="job-1", status="submitted")
        adapter = FakeApiAdapter(persona_confirmation_plan=[{"approved": False, "last_update_id": 101}])
        brain_service = BrainService(repository, config.brain, adapter=adapter)
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=brain_service,
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(
                adapter,
                persona_email_cooldown_seconds=999999,
                persona_confirmation_required=True,
            ),
        )
        before = repository.submissions.get_submission("job-1")
        assert before is not None

        outcome = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1)
        after = repository.submissions.get_submission("job-1")
    finally:
        repository.close()

    assert outcome.status == "waiting_persona_confirmation"
    assert after is not None
    assert before.timeout_deadline_at is not None
    assert after.timeout_deadline_at is not None
    assert datetime.fromisoformat(after.timeout_deadline_at) > datetime.fromisoformat(before.timeout_deadline_at)


def test_service_worker_respects_learned_safe_cap_when_submitting_new_batch() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 10
        config.service.max_pending_jobs = 10
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-learned-safe-cap"
        _seed_run(repository, run_id)
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=FakeBatchPreparationService(_batch_result(_service_candidates(10))),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        runtime = replace(_runtime_record(run_id=run_id), counters_json=json.dumps({"learned_safe_cap": 4}))
        outcome = worker.run_tick(runtime=runtime, tick_id=1)
    finally:
        repository.close()

    assert outcome.status == "running"
    assert outcome.submitted_count == 4
    assert outcome.pending_job_count == 4
    assert len(adapter.submit_calls) == 4


def test_service_worker_pauses_three_minutes_after_concurrent_submission_limit() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        run_id = "run-submit-limit"
        _seed_run(repository, run_id)
        adapter = FakeApiAdapter()

        def fail_submit(expression: str, sim_config: dict) -> dict:
            adapter.submit_calls.append((expression, sim_config))
            raise ConcurrentSimulationLimitExceeded("CONCURRENT_SIMULATION_LIMIT_EXCEEDED")

        adapter.submit_simulation = fail_submit  # type: ignore[method-assign]
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=FakeBatchPreparationService(_batch_result(_service_candidates(1))),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1)
    finally:
        repository.close()

    assert outcome.status == "cooldown"
    assert outcome.next_sleep_seconds == 180
    assert outcome.cooldown_until is not None
    assert outcome.last_error is not None
    assert "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" in outcome.last_error
    assert len(adapter.submit_calls) == 1


def test_service_worker_persists_learned_safe_cap_after_partial_limit_hit() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.loop.simulation_batch_size = 3
        config.service.max_pending_jobs = 10
        config.service.max_consecutive_batch_failures_before_auth_check = 999
        run_id = "run-submit-limit-partial"
        _seed_run(repository, run_id)
        runtime = _runtime_record(run_id=run_id)
        repository.service_runtime.upsert_state(runtime)
        adapter = FakeApiAdapter()
        original_submit = adapter.submit_simulation

        def fail_after_partial(expression: str, sim_config: dict) -> dict:
            if len(adapter.submit_calls) >= 2:
                adapter.submit_calls.append((expression, sim_config))
                raise ConcurrentSimulationLimitExceeded("CONCURRENT_SIMULATION_LIMIT_EXCEEDED")
            return original_submit(expression, sim_config)

        adapter.submit_simulation = fail_after_partial  # type: ignore[method-assign]
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            batch_service=FakeBatchPreparationService(_batch_result(_service_candidates(3))),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=runtime, tick_id=1)
        refreshed = repository.service_runtime.get_state(config.service.lock_name)
    finally:
        repository.close()

    assert outcome.status == "cooldown"
    assert outcome.pending_job_count == 2
    assert len(adapter.submit_calls) == 3
    assert refreshed is not None
    counters = json.loads(refreshed.counters_json)
    assert counters["learned_safe_cap"] == 3
    assert refreshed.cooldown_until is not None
    assert refreshed.last_error is not None


def test_service_worker_empty_poll_skips_progress_event_and_returns_zero_new_results(tmp_path: Path) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.max_pending_jobs = 1
        config.runtime.progress_log_dir = str(tmp_path / "progress")
        run_id = "run-empty-poll"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(run_id, [_candidate("alpha-1", "rank(close)")])
        _seed_pending_batch(repository, run_id=run_id, batch_id="batch-1", job_id="job-1", status="submitted")
        adapter = FakeApiAdapter(status_plan={"job-1": [{"job_id": "job-1", "status": "running"}]})
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1)
        log_path = tmp_path / "progress" / f"{run_id}.jsonl"
    finally:
        repository.close()

    assert outcome.status == "running"
    assert outcome.new_result_count == 0
    assert not log_path.exists()


def test_service_worker_logs_completed_batch_poll_even_without_new_results(tmp_path: Path) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.runtime.progress_log_dir = str(tmp_path / "progress")
        run_id = "run-completed-no-results"
        _seed_run(repository, run_id)
        _seed_pending_batch(repository, run_id=run_id, batch_id="batch-1", job_id="job-1", status="completed")
        repository.submissions.update_batch_status("batch-1", status="completed", updated_at=_timestamp())
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker._poll_pending_batches(
            runtime=_runtime_record(run_id=run_id),
            tick_id=1,
            batch_ids=["batch-1"],
        )
        log_path = tmp_path / "progress" / f"{run_id}.jsonl"
        progress_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    finally:
        repository.close()

    assert outcome.status == "idle"
    assert outcome.new_result_count == 0
    assert any(row["event"] == "batch_polled" and row["status"] == "completed" for row in progress_rows)


def test_service_scheduler_exponential_backoff_resets_after_results() -> None:
    config = _service_config()
    config.service.poll_interval_seconds = 5
    scheduler = ServiceScheduler(config.service)
    runtime = _runtime_record(run_id="run-scheduler")
    outcome = ServiceTickOutcome(status="running", pending_job_count=1)

    assert scheduler.next_sleep_seconds(runtime=runtime, outcome=outcome) == 5

    scheduler.record_poll_result(0)
    assert scheduler.next_sleep_seconds(runtime=runtime, outcome=outcome) == 10

    scheduler.record_poll_result(0)
    assert scheduler.next_sleep_seconds(runtime=runtime, outcome=outcome) == 15

    scheduler.record_poll_result(0)
    assert scheduler.next_sleep_seconds(runtime=runtime, outcome=outcome) == 15

    scheduler.record_poll_result(0)
    assert scheduler.next_sleep_seconds(runtime=runtime, outcome=outcome) == 15

    scheduler.record_poll_result(2)
    assert scheduler.next_sleep_seconds(runtime=runtime, outcome=outcome) == 5


def test_service_scheduler_uses_one_second_sleep_for_fractional_cooldown_tail() -> None:
    config = _service_config()
    scheduler = ServiceScheduler(config.service)
    runtime = _runtime_record(run_id="run-scheduler-cooldown-tail")
    outcome = ServiceTickOutcome(
        status="cooldown",
        pending_job_count=10,
        cooldown_until="2026-01-01T00:00:00.300000+00:00",
    )

    assert scheduler.next_sleep_seconds(
        runtime=runtime,
        outcome=outcome,
        now="2026-01-01T00:00:00+00:00",
    ) == 1


def test_service_runner_suppresses_noop_progress_events_after_first_empty_poll(tmp_path: Path) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.max_pending_jobs = 1
        config.runtime.progress_log_dir = str(tmp_path / "progress")
        run_id = "run-noop-progress"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(run_id, [_candidate("alpha-1", "rank(close)")])
        _seed_pending_batch(repository, run_id=run_id, batch_id="batch-1", job_id="job-1", status="submitted")
        repository.service_runtime.upsert_state(_runtime_record(run_id=run_id))
        adapter = FakeApiAdapter(
            status_plan={
                "job-1": [
                    {"job_id": "job-1", "status": "running"},
                    {"job_id": "job-1", "status": "running"},
                ]
            }
        )
        runner = ServiceRunner(
            repository,
            config=config,
            environment=_environment("fresh-run-id"),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            sleep_fn=lambda seconds: None,
            install_signal_handlers=False,
        )

        summary = runner.run(max_ticks=2)
        log_path = tmp_path / "progress" / f"{run_id}.jsonl"
        progress_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    finally:
        repository.close()

    assert summary.status == "service_running"
    assert sum(1 for row in progress_rows if row["event"] == "service_tick_started") == 1
    assert sum(1 for row in progress_rows if row["event"] == "service_sleeping") == 1
    assert sum(1 for row in progress_rows if row["event"] == "service_tick_completed") == 2
    assert not any(row["event"] == "batch_polled" for row in progress_rows)


def test_service_runner_releases_lock_on_graceful_shutdown() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        run_id = "run-shutdown"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(run_id, [_candidate("alpha-1", "rank(close)")])
        _seed_pending_batch(repository, run_id=run_id, batch_id="batch-1", job_id="job-1", status="submitted")
        repository.service_runtime.upsert_state(
            replace(
                _runtime_record(run_id=run_id),
                persona_url="https://persona.example/stale",
                persona_wait_started_at=_timestamp(),
                persona_last_notification_at=_timestamp(),
            )
        )
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
        run = repository.get_run(run_id)
    finally:
        repository.close()

    assert summary.status == "service_stopped_pending"
    assert runtime is not None
    assert runtime.owner_token == ""
    assert runtime.status == "service_stopped_pending"
    assert runtime.pending_job_count == 1
    assert runtime.persona_url is None
    assert runtime.persona_wait_started_at is None
    assert runtime.persona_last_notification_at is None
    assert run is not None
    assert run.finished_at is None


def test_service_runner_interruptible_sleep_stops_without_waiting_full_interval() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        runner = ServiceRunner(
            repository,
            config=config,
            environment=_environment("run-sleep-stop"),
            brain_service=BrainService(repository, config.brain, adapter=FakeApiAdapter()),
            sleep_fn=lambda seconds: sleep_calls.append(seconds) or runner.request_shutdown(),
            install_signal_handlers=False,
        )
        sleep_calls: list[float] = []

        runner._sleep_interruptibly(120.0)
    finally:
        repository.close()

    assert sleep_calls
    assert sleep_calls[0] <= 1.0
    assert len(sleep_calls) == 1


def test_service_marks_ambiguous_submitting_batch_failed_when_policy_is_fail() -> None:
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
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        summary = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1, stop_requested=True)
        batch = repository.submissions.get_batch("batch-ambiguous")
        submission = repository.submissions.get_submission("job-1")
    finally:
        repository.close()

    assert summary.status == "idle"
    assert batch is not None
    assert batch.status == "failed"
    assert batch.service_status_reason == "ambiguous_submission_assumed_failed"
    assert submission is not None
    assert submission.status == "failed"
    assert len(adapter.submit_calls) == 0


def test_service_resubmits_ambiguous_batch_when_policy_is_resubmit() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.ambiguous_submission_policy = "resubmit"
        config.service.max_pending_jobs = 0
        run_id = "run-resubmit"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(
            run_id,
            [
                _candidate("alpha-1", "rank(close)"),
                _candidate("alpha-2", "rank(open)"),
            ],
        )
        timestamp = _timestamp()
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-ambiguous",
                run_id=run_id,
                round_index=1,
                backend="api",
                status="paused_quarantine",
                candidate_count=2,
                sim_config_snapshot=json.dumps(
                    {
                        "region": "USA",
                        "candidate_payloads": [
                            {"candidate_id": "alpha-1", "expression": "rank(close)", "job_id": "old-job-1"},
                            {"candidate_id": "alpha-2", "expression": "rank(open)", "job_id": "old-job-2"},
                        ],
                    }
                ),
                export_path=None,
                notes_json=json.dumps({"candidate_ids": ["alpha-1", "alpha-2"]}),
                service_status_reason="ambiguous_submission",
                quarantined_at=timestamp,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        repository.service_runtime.upsert_state(_runtime_record(run_id=run_id))
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1)
        batches = repository.submissions.list_batches(run_id)
        old_batch = repository.submissions.get_batch("batch-ambiguous")
        new_batches = [batch for batch in batches if batch.batch_id != "batch-ambiguous"]
    finally:
        repository.close()

    assert outcome.status in {"idle", "running"}
    assert old_batch is not None
    assert old_batch.status == "failed"
    assert old_batch.service_status_reason.startswith("ambiguous_submission_resubmitted:")
    assert len(adapter.submit_calls) == 2
    assert len(new_batches) == 1
    assert new_batches[0].status in {"submitted", "running", "completed"}


def test_service_replay_guard_skips_candidates_with_active_submission_during_ambiguous_resubmit() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.ambiguous_submission_policy = "resubmit"
        run_id = "run-resubmit-replay-active"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(
            run_id,
            [
                _candidate("alpha-1", "rank(close)"),
                _candidate("alpha-2", "rank(open)"),
            ],
        )
        _seed_pending_batch(
            repository,
            run_id=run_id,
            batch_id="batch-active",
            job_id="job-active",
            status="submitted",
            candidate_id="alpha-1",
            expression="rank(close)",
        )
        timestamp = _timestamp()
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-ambiguous",
                run_id=run_id,
                round_index=2,
                backend="api",
                status="paused_quarantine",
                candidate_count=2,
                sim_config_snapshot=json.dumps(
                    {
                        "region": "USA",
                        "candidate_payloads": [
                            {"candidate_id": "alpha-1", "expression": "rank(close)", "job_id": "old-job-1"},
                            {"candidate_id": "alpha-2", "expression": "rank(open)", "job_id": "old-job-2"},
                        ],
                    }
                ),
                export_path=None,
                notes_json=json.dumps({"candidate_ids": ["alpha-1", "alpha-2"]}),
                service_status_reason="ambiguous_submission",
                quarantined_at=timestamp,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        _, _, resubmitted, _ = worker._recover_submitting_batches(  # noqa: SLF001
            service_name=config.service.lock_name,
            run_id=run_id,
            now=_timestamp(),
            allow_resubmit=True,
        )
        source_batch = repository.submissions.get_batch("batch-ambiguous")
        new_batch = repository.submissions.get_batch(resubmitted[0]) if resubmitted else None
        pending_alpha_1 = [
            submission
            for submission in repository.submissions.list_pending_submissions(run_id)
            if submission.candidate_id == "alpha-1"
        ]
    finally:
        repository.close()

    assert len(resubmitted) == 1
    assert len(adapter.submit_calls) == 1
    assert source_batch is not None
    assert source_batch.status == "failed"
    assert source_batch.service_status_reason.startswith("ambiguous_submission_resubmitted:")
    source_notes = json.loads(source_batch.notes_json)
    assert source_notes["recovery_source_candidate_ids"] == ["alpha-1", "alpha-2"]
    assert source_notes["recovery_skipped_candidates"] == [
        {
            "candidate_id": "alpha-1",
            "reason": "ambiguous_replay_guard_active_submission",
            "blocking_batch_id": "batch-active",
            "blocking_job_id": "job-active",
            "blocking_status": "submitted",
        }
    ]
    assert new_batch is not None
    assert new_batch.candidate_count == 1
    new_notes = json.loads(new_batch.notes_json)
    assert new_notes["candidate_ids"] == ["alpha-2"]
    assert new_notes["resubmitted_from_batch_id"] == "batch-ambiguous"
    assert new_notes["recovery_skipped_candidates"] == source_notes["recovery_skipped_candidates"]
    assert len(pending_alpha_1) == 1
    assert pending_alpha_1[0].batch_id == "batch-active"


def test_service_replay_guard_skips_candidates_with_terminal_result_during_ambiguous_resubmit() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.ambiguous_submission_policy = "resubmit"
        run_id = "run-resubmit-replay-terminal"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(
            run_id,
            [
                _candidate("alpha-1", "rank(close)"),
                _candidate("alpha-2", "rank(open)"),
            ],
        )
        _seed_terminal_result(
            repository,
            run_id=run_id,
            batch_id="batch-terminal",
            job_id="job-terminal",
            candidate_id="alpha-1",
            expression="rank(close)",
            status="failed",
        )
        timestamp = _timestamp()
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-ambiguous",
                run_id=run_id,
                round_index=2,
                backend="api",
                status="paused_quarantine",
                candidate_count=2,
                sim_config_snapshot=json.dumps(
                    {
                        "region": "USA",
                        "candidate_payloads": [
                            {"candidate_id": "alpha-1", "expression": "rank(close)", "job_id": "old-job-1"},
                            {"candidate_id": "alpha-2", "expression": "rank(open)", "job_id": "old-job-2"},
                        ],
                    }
                ),
                export_path=None,
                notes_json=json.dumps({"candidate_ids": ["alpha-1", "alpha-2"]}),
                service_status_reason="ambiguous_submission",
                quarantined_at=timestamp,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        _, _, resubmitted, _ = worker._recover_submitting_batches(  # noqa: SLF001
            service_name=config.service.lock_name,
            run_id=run_id,
            now=_timestamp(),
            allow_resubmit=True,
        )
        source_batch = repository.submissions.get_batch("batch-ambiguous")
        new_batch = repository.submissions.get_batch(resubmitted[0]) if resubmitted else None
    finally:
        repository.close()

    assert len(resubmitted) == 1
    assert len(adapter.submit_calls) == 1
    assert source_batch is not None
    source_notes = json.loads(source_batch.notes_json)
    assert source_notes["recovery_skipped_candidates"] == [
        {
            "candidate_id": "alpha-1",
            "reason": "ambiguous_replay_guard_terminal_result",
            "blocking_batch_id": "batch-terminal",
            "blocking_job_id": "job-terminal",
            "blocking_status": "failed",
        }
    ]
    assert new_batch is not None
    new_notes = json.loads(new_batch.notes_json)
    assert new_notes["candidate_ids"] == ["alpha-2"]
    assert "alpha-1" not in new_notes["candidate_ids"]


def test_service_replay_guard_fails_ambiguous_batch_when_no_candidates_remain_resubmittable() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.ambiguous_submission_policy = "resubmit"
        run_id = "run-resubmit-replay-empty"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(
            run_id,
            [
                _candidate("alpha-1", "rank(close)"),
                _candidate("alpha-2", "rank(open)"),
            ],
        )
        _seed_pending_batch(
            repository,
            run_id=run_id,
            batch_id="batch-active",
            job_id="job-active",
            status="running",
            candidate_id="alpha-1",
            expression="rank(close)",
        )
        _seed_terminal_result(
            repository,
            run_id=run_id,
            batch_id="batch-terminal",
            job_id="job-terminal",
            candidate_id="alpha-2",
            expression="rank(open)",
            status="completed",
        )
        timestamp = _timestamp()
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-ambiguous",
                run_id=run_id,
                round_index=2,
                backend="api",
                status="paused_quarantine",
                candidate_count=2,
                sim_config_snapshot=json.dumps(
                    {
                        "region": "USA",
                        "candidate_payloads": [
                            {"candidate_id": "alpha-1", "expression": "rank(close)", "job_id": "old-job-1"},
                            {"candidate_id": "alpha-2", "expression": "rank(open)", "job_id": "old-job-2"},
                        ],
                    }
                ),
                export_path=None,
                notes_json=json.dumps({"candidate_ids": ["alpha-1", "alpha-2"]}),
                service_status_reason="ambiguous_submission",
                quarantined_at=timestamp,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        _, failed, resubmitted, _ = worker._recover_submitting_batches(  # noqa: SLF001
            service_name=config.service.lock_name,
            run_id=run_id,
            now=_timestamp(),
            allow_resubmit=True,
        )
        source_batch = repository.submissions.get_batch("batch-ambiguous")
        batches = repository.submissions.list_batches(run_id)
    finally:
        repository.close()

    assert resubmitted == []
    assert failed == ["batch-ambiguous"]
    assert len(adapter.submit_calls) == 0
    assert len([batch for batch in batches if batch.batch_id not in {"batch-active", "batch-ambiguous", "batch-terminal"}]) == 0
    assert source_batch is not None
    assert source_batch.status == "failed"
    assert source_batch.service_status_reason == "ambiguous_submission_no_resubmittable_candidates"
    source_notes = json.loads(source_batch.notes_json)
    assert {item["reason"] for item in source_notes["recovery_skipped_candidates"]} == {
        "ambiguous_replay_guard_active_submission",
        "ambiguous_replay_guard_terminal_result",
    }


def test_service_replay_guard_skips_missing_candidate_payload_without_copying_it_to_new_batch() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.ambiguous_submission_policy = "resubmit"
        run_id = "run-resubmit-replay-missing"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(run_id, [_candidate("alpha-2", "rank(open)")])
        timestamp = _timestamp()
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-ambiguous",
                run_id=run_id,
                round_index=2,
                backend="api",
                status="paused_quarantine",
                candidate_count=2,
                sim_config_snapshot=json.dumps(
                    {
                        "region": "USA",
                        "candidate_payloads": [
                            {"candidate_id": "alpha-1", "job_id": "old-job-1"},
                            {"candidate_id": "alpha-2", "expression": "rank(open)", "job_id": "old-job-2"},
                        ],
                    }
                ),
                export_path=None,
                notes_json=json.dumps({"candidate_ids": ["alpha-1", "alpha-2"]}),
                service_status_reason="ambiguous_submission",
                quarantined_at=timestamp,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        _, _, resubmitted, _ = worker._recover_submitting_batches(  # noqa: SLF001
            service_name=config.service.lock_name,
            run_id=run_id,
            now=_timestamp(),
            allow_resubmit=True,
        )
        source_batch = repository.submissions.get_batch("batch-ambiguous")
        new_batch = repository.submissions.get_batch(resubmitted[0]) if resubmitted else None
    finally:
        repository.close()

    assert len(resubmitted) == 1
    assert len(adapter.submit_calls) == 1
    assert source_batch is not None
    source_notes = json.loads(source_batch.notes_json)
    assert source_notes["recovery_skipped_candidates"] == [
        {
            "candidate_id": "alpha-1",
            "reason": "ambiguous_submission_missing_candidate",
        }
    ]
    assert new_batch is not None
    assert new_batch.candidate_count == 1
    new_notes = json.loads(new_batch.notes_json)
    assert new_notes["candidate_ids"] == ["alpha-2"]
    snapshot = json.loads(new_batch.sim_config_snapshot)
    assert [payload["candidate_id"] for payload in snapshot["candidate_payloads"]] == ["alpha-2"]


def test_service_recovers_partial_ambiguous_batch_without_duplicate_resubmit() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.ambiguous_submission_policy = "resubmit"
        config.service.max_pending_jobs = 0
        run_id = "run-resubmit-partial"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(
            run_id,
            [
                _candidate("alpha-1", "rank(close)"),
                _candidate("alpha-2", "rank(open)"),
            ],
        )
        timestamp = _timestamp()
        sim_snapshot = json.dumps(
            {
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
                "neutralization": "sector",
                "decay": 0,
                "candidate_payloads": [
                    {"candidate_id": "alpha-1", "expression": "rank(close)", "job_id": "job-1"},
                    {"candidate_id": "alpha-2", "expression": "rank(open)", "job_id": "job-2"},
                ],
            }
        )
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-ambiguous",
                run_id=run_id,
                round_index=1,
                backend="api",
                status="paused_quarantine",
                candidate_count=2,
                sim_config_snapshot=sim_snapshot,
                export_path=None,
                notes_json=json.dumps({"candidate_ids": ["alpha-1", "alpha-2"]}),
                service_status_reason="ambiguous_submission",
                quarantined_at=timestamp,
                created_at=timestamp,
                updated_at=timestamp,
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
                    sim_config_snapshot=sim_snapshot,
                    submitted_at=timestamp,
                    updated_at=timestamp,
                    completed_at=None,
                    export_path=None,
                    raw_submission_json="{}",
                    error_message=None,
                )
            ]
        )
        repository.service_runtime.upsert_state(_runtime_record(run_id=run_id))
        adapter = FakeApiAdapter(status_plan={"job-1": [{"job_id": "job-1", "status": "completed"}]})
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1)
        batch = repository.submissions.get_batch("batch-ambiguous")
        submission = repository.submissions.get_submission("job-1")
        result = repository.brain_results.get_result("job-1")
    finally:
        repository.close()

    assert outcome.status == "idle"
    assert outcome.completed_count == 1
    assert batch is not None
    assert batch.status == "completed"
    assert batch.service_status_reason is None
    assert submission is not None
    assert submission.status == "completed"
    assert result is not None
    assert result.status == "completed"
    assert len(adapter.submit_calls) == 0


def test_service_resubmit_pauses_after_concurrent_submission_limit() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.ambiguous_submission_policy = "resubmit"
        run_id = "run-resubmit-limit"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(
            run_id,
            [
                _candidate("alpha-1", "rank(close)"),
                _candidate("alpha-2", "rank(open)"),
            ],
        )
        timestamp = _timestamp()
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-ambiguous",
                run_id=run_id,
                round_index=1,
                backend="api",
                status="paused_quarantine",
                candidate_count=2,
                sim_config_snapshot=json.dumps(
                    {
                        "region": "USA",
                        "candidate_payloads": [
                            {"candidate_id": "alpha-1", "expression": "rank(close)", "job_id": "old-job-1"},
                            {"candidate_id": "alpha-2", "expression": "rank(open)", "job_id": "old-job-2"},
                        ],
                    }
                ),
                export_path=None,
                notes_json=json.dumps({"candidate_ids": ["alpha-1", "alpha-2"]}),
                service_status_reason="ambiguous_submission",
                quarantined_at=timestamp,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        repository.service_runtime.upsert_state(_runtime_record(run_id=run_id))
        adapter = FakeApiAdapter()

        def fail_submit(expression: str, sim_config: dict) -> dict:
            adapter.submit_calls.append((expression, sim_config))
            raise ConcurrentSimulationLimitExceeded("CONCURRENT_SIMULATION_LIMIT_EXCEEDED")

        adapter.submit_simulation = fail_submit  # type: ignore[method-assign]
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1)
        batch = repository.submissions.get_batch("batch-ambiguous")
    finally:
        repository.close()

    assert outcome.status == "cooldown"
    assert outcome.next_sleep_seconds == 180
    assert outcome.cooldown_until is not None
    assert batch is not None
    assert batch.status == "paused_quarantine"
    assert len(adapter.submit_calls) == 1


def test_service_only_resubmits_latest_ambiguous_leaf_batch() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.ambiguous_submission_policy = "resubmit"
        config.service.max_pending_jobs = 0
        run_id = "run-resubmit-latest-only"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(
            run_id,
            [
                _candidate("alpha-1", "rank(close)"),
                _candidate("alpha-2", "rank(open)"),
            ],
        )
        first = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
        second = datetime(2026, 1, 1, 0, 5, tzinfo=UTC).isoformat()
        third = datetime(2026, 1, 1, 0, 10, tzinfo=UTC).isoformat()
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-root",
                run_id=run_id,
                round_index=1,
                backend="api",
                status="paused_quarantine",
                candidate_count=2,
                sim_config_snapshot=json.dumps(
                    {
                        "region": "USA",
                        "candidate_payloads": [
                            {"candidate_id": "alpha-1", "expression": "rank(close)", "job_id": "old-job-1"},
                            {"candidate_id": "alpha-2", "expression": "rank(open)", "job_id": "old-job-2"},
                        ],
                    }
                ),
                export_path=None,
                notes_json=json.dumps({"candidate_ids": ["alpha-1", "alpha-2"]}),
                service_status_reason="ambiguous_submission",
                quarantined_at=first,
                created_at=first,
                updated_at=first,
            )
        )
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-middle",
                run_id=run_id,
                round_index=1,
                backend="api",
                status="completed",
                candidate_count=2,
                sim_config_snapshot=json.dumps({"region": "USA"}),
                export_path=None,
                notes_json=json.dumps(
                    {
                        "candidate_ids": ["alpha-1", "alpha-2"],
                        "resubmitted_from_batch_id": "batch-root",
                    }
                ),
                service_status_reason="submission_failed:ConcurrentSimulationLimitExceeded",
                created_at=second,
                updated_at=second,
            )
        )
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-leaf",
                run_id=run_id,
                round_index=1,
                backend="api",
                status="paused_quarantine",
                candidate_count=2,
                sim_config_snapshot=json.dumps(
                    {
                        "region": "USA",
                        "candidate_payloads": [
                            {"candidate_id": "alpha-1", "expression": "rank(close)", "job_id": "leaf-job-1"},
                            {"candidate_id": "alpha-2", "expression": "rank(open)", "job_id": "leaf-job-2"},
                        ],
                    }
                ),
                export_path=None,
                notes_json=json.dumps(
                    {
                        "candidate_ids": ["alpha-1", "alpha-2"],
                        "resubmitted_from_batch_id": "batch-middle",
                    }
                ),
                service_status_reason="ambiguous_submission",
                quarantined_at=third,
                created_at=third,
                updated_at=third,
            )
        )
        repository.service_runtime.upsert_state(_runtime_record(run_id=run_id))
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1)
        root_batch = repository.submissions.get_batch("batch-root")
        leaf_batch = repository.submissions.get_batch("batch-leaf")
        batches = repository.submissions.list_batches(run_id)
        new_batches = [batch for batch in batches if batch.batch_id not in {"batch-root", "batch-middle", "batch-leaf"}]
    finally:
        repository.close()

    assert outcome.status in {"idle", "running"}
    assert root_batch is not None
    assert root_batch.status == "failed"
    assert root_batch.service_status_reason == "ambiguous_submission_superseded"
    assert leaf_batch is not None
    assert leaf_batch.status == "failed"
    assert leaf_batch.service_status_reason.startswith("ambiguous_submission_resubmitted:")
    assert len(adapter.submit_calls) == 2
    assert len(new_batches) == 1


def test_service_only_resubmits_newest_ambiguous_batch() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _service_config()
        config.service.ambiguous_submission_policy = "resubmit"
        config.service.max_pending_jobs = 0
        run_id = "run-resubmit-newest-only"
        _seed_run(repository, run_id)
        repository.save_alpha_candidates(
            run_id,
            [
                _candidate("alpha-1", "rank(close)"),
                _candidate("alpha-2", "rank(open)"),
            ],
        )
        older = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
        newer = datetime(2026, 1, 1, 0, 10, tzinfo=UTC).isoformat()
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-older",
                run_id=run_id,
                round_index=1,
                backend="api",
                status="paused_quarantine",
                candidate_count=2,
                sim_config_snapshot=json.dumps(
                    {
                        "region": "USA",
                        "candidate_payloads": [
                            {"candidate_id": "alpha-1", "expression": "rank(close)", "job_id": "old-job-1"},
                            {"candidate_id": "alpha-2", "expression": "rank(open)", "job_id": "old-job-2"},
                        ],
                    }
                ),
                export_path=None,
                notes_json=json.dumps({"candidate_ids": ["alpha-1", "alpha-2"]}),
                service_status_reason="ambiguous_submission",
                quarantined_at=older,
                created_at=older,
                updated_at=older,
            )
        )
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-newer",
                run_id=run_id,
                round_index=2,
                backend="api",
                status="paused_quarantine",
                candidate_count=2,
                sim_config_snapshot=json.dumps(
                    {
                        "region": "USA",
                        "candidate_payloads": [
                            {"candidate_id": "alpha-1", "expression": "rank(close)", "job_id": "new-job-1"},
                            {"candidate_id": "alpha-2", "expression": "rank(open)", "job_id": "new-job-2"},
                        ],
                    }
                ),
                export_path=None,
                notes_json=json.dumps({"candidate_ids": ["alpha-1", "alpha-2"]}),
                service_status_reason="ambiguous_submission",
                quarantined_at=newer,
                created_at=newer,
                updated_at=newer,
            )
        )
        repository.service_runtime.upsert_state(_runtime_record(run_id=run_id))
        adapter = FakeApiAdapter()
        worker = ServiceWorker(
            repository,
            config=config,
            environment=_environment(run_id),
            brain_service=BrainService(repository, config.brain, adapter=adapter),
            session_manager=SessionManager(adapter, persona_retry_interval_seconds=config.service.persona_retry_interval_seconds),
            notification_manager=NotificationManager(adapter, persona_email_cooldown_seconds=900),
        )

        outcome = worker.run_tick(runtime=_runtime_record(run_id=run_id), tick_id=1)
        older_batch = repository.submissions.get_batch("batch-older")
        newer_batch = repository.submissions.get_batch("batch-newer")
        batches = repository.submissions.list_batches(run_id)
        new_batches = [batch for batch in batches if batch.batch_id not in {"batch-older", "batch-newer"}]
    finally:
        repository.close()

    assert outcome.status in {"idle", "running"}
    assert older_batch is not None
    assert older_batch.status == "failed"
    assert older_batch.service_status_reason == "ambiguous_submission_stale"
    assert newer_batch is not None
    assert newer_batch.status == "failed"
    assert newer_batch.service_status_reason.startswith("ambiguous_submission_resubmitted:")
    assert len(adapter.submit_calls) == 2
    assert len(new_batches) == 1


def _service_config():
    config = load_config("config/dev.yaml")
    config.storage.path = ":memory:"
    config.brain.backend = "api"
    config.service.resume_incomplete_jobs = True
    config.service.max_consecutive_failures = 2
    config.service.cooldown_seconds = 60
    config.service.persona_confirmation_required = False
    config.service.ambiguous_submission_policy = "fail"
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
    candidate_id: str = "alpha-1",
    expression: str = "rank(close)",
) -> None:
    timestamp = _timestamp()
    sim_config_snapshot = json.dumps(
        {
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "neutralization": "sector",
            "decay": 0,
        }
    )
    repository.submissions.upsert_batch(
        SubmissionBatchRecord(
            batch_id=batch_id,
            run_id=run_id,
            round_index=1,
            backend="api",
            status="submitted",
            candidate_count=1,
            sim_config_snapshot=sim_config_snapshot,
            export_path=None,
            notes_json="{}",
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    repository.submissions.upsert_submissions(
        [
            SubmissionRecord(
                job_id=job_id,
                batch_id=batch_id,
                run_id=run_id,
                round_index=1,
                candidate_id=candidate_id,
                expression=expression,
                backend="api",
                status=status,
                sim_config_snapshot=sim_config_snapshot,
                submitted_at=timestamp,
                updated_at=timestamp,
                completed_at=None,
                export_path=None,
                raw_submission_json="{}",
                error_message=None,
            )
        ]
    )


def _seed_terminal_result(
    repository: SQLiteRepository,
    *,
    run_id: str,
    batch_id: str,
    job_id: str,
    candidate_id: str,
    expression: str,
    status: str,
) -> None:
    timestamp = _timestamp()
    sim_config_snapshot = json.dumps(
        {
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "neutralization": "sector",
            "decay": 0,
        }
    )
    repository.submissions.upsert_batch(
        SubmissionBatchRecord(
            batch_id=batch_id,
            run_id=run_id,
            round_index=1,
            backend="api",
            status="completed",
            candidate_count=1,
            sim_config_snapshot=sim_config_snapshot,
            export_path=None,
            notes_json=json.dumps({"candidate_ids": [candidate_id]}),
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    repository.submissions.upsert_submissions(
        [
            SubmissionRecord(
                job_id=job_id,
                batch_id=batch_id,
                run_id=run_id,
                round_index=1,
                candidate_id=candidate_id,
                expression=expression,
                backend="api",
                status=status,
                sim_config_snapshot=sim_config_snapshot,
                submitted_at=timestamp,
                updated_at=timestamp,
                completed_at=timestamp,
                export_path=None,
                raw_submission_json="{}",
                error_message=None,
            )
        ]
    )
    repository.brain_results.save_results(
        [
            BrainResultRecord(
                job_id=job_id,
                run_id=run_id,
                round_index=1,
                batch_id=batch_id,
                candidate_id=candidate_id,
                expression=expression,
                status=status,
                region="USA",
                universe="TOP3000",
                delay=1,
                neutralization="sector",
                decay=0,
                sharpe=1.2 if status == "completed" else None,
                fitness=0.9 if status == "completed" else None,
                turnover=0.5 if status == "completed" else None,
                drawdown=0.2 if status == "completed" else None,
                returns=0.07 if status == "completed" else None,
                margin=0.04 if status == "completed" else None,
                submission_eligible=(status == "completed"),
                rejection_reason=None,
                raw_result_json="{}",
                metric_source="external_brain",
                simulated_at=timestamp,
                created_at=timestamp,
            )
        ]
    )


def _seed_completed_batch(
    repository: SQLiteRepository,
    *,
    run_id: str,
    batch_id: str,
    statuses: tuple[str, ...],
    created_at: datetime,
) -> None:
    created_at_iso = created_at.isoformat()
    repository.submissions.upsert_batch(
        SubmissionBatchRecord(
            batch_id=batch_id,
            run_id=run_id,
            round_index=1,
            backend="api",
            status="completed",
            candidate_count=len(statuses),
            sim_config_snapshot=json.dumps({"region": "USA"}),
            export_path=None,
            notes_json="{}",
            created_at=created_at_iso,
            updated_at=created_at_iso,
        )
    )
    repository.submissions.upsert_submissions(
        [
            SubmissionRecord(
                job_id=f"{batch_id}-job-{index}",
                batch_id=batch_id,
                run_id=run_id,
                round_index=1,
                candidate_id=f"{batch_id}-alpha-{index}",
                expression=f"rank(close_{index})",
                backend="api",
                status=status,
                sim_config_snapshot=json.dumps({"region": "USA"}),
                submitted_at=created_at_iso,
                updated_at=created_at_iso,
                completed_at=created_at_iso,
                export_path=None,
                raw_submission_json="{}",
                error_message=None,
            )
            for index, status in enumerate(statuses, start=1)
        ]
    )


def _service_candidates(count: int) -> list[AlphaCandidate]:
    return [
        AlphaCandidate(
            alpha_id=f"alpha-{index}",
            expression=f"rank(ts_mean(close, {index + 1}))",
            normalized_expression=f"rank(ts_mean(close, {index + 1}))",
            generation_mode="template",
            parent_ids=(),
            complexity=3,
            created_at=_timestamp(),
            template_name="momentum",
            fields_used=("close",),
            operators_used=("rank", "ts_mean"),
            depth=2,
            generation_metadata={"field_families": ["price"]},
        )
        for index in range(1, count + 1)
    ]


def _batch_result(candidates: list[AlphaCandidate]) -> BatchPreparationResult:
    memory_service = PatternMemoryService()
    selected = []
    for candidate in candidates:
        signature = memory_service.extract_signature(
            candidate.expression,
            generation_metadata=candidate.generation_metadata,
        )
        selected.append(
            CandidateScore(
                candidate=candidate,
                objective_vector=ObjectiveVector(),
                local_heuristic_score=0.0,
                novelty_score=0.0,
                family_score=0.0,
                diversity_score=0.0,
                duplicate_risk=0.0,
                crowding_penalty=0.0,
                regime_fit=0.0,
                composite_score=0.0,
                structural_signature=signature,
            )
        )
    return BatchPreparationResult(
        candidates=tuple(candidates),
        selected=tuple(selected),
        regime_key="service-regime",
        validated_count=len(candidates),
        archived_count=0,
        generation_stage_metrics={},
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
