from __future__ import annotations

import signal
import time
from datetime import UTC, datetime, timedelta
from types import FrameType

from adapters.brain_api_adapter import BrainApiAdapter
from core.config import AppConfig
from core.logging import get_logger
from core.run_context import RunContext
from services.brain_service import BrainService
from services.heartbeat_reporter import HeartbeatReporter
from services.models import CommandEnvironment, ServiceRunSummary
from services.notification_manager import NotificationManager
from services.progress_log import append_progress_event, resolve_progress_log_path
from services.runtime_lock import RuntimeLock
from services.runtime_service import init_run
from services.service_scheduler import ServiceScheduler
from services.service_worker import ServiceWorker
from services.session_manager import SessionManager
from storage.models import ServiceRuntimeRecord
from storage.repository import SQLiteRepository


class ServiceRunner:
    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        brain_service: BrainService | None = None,
        sleep_fn=time.sleep,
        install_signal_handlers: bool = True,
    ) -> None:
        self.repository = repository
        self.config = config
        self.environment = environment
        self.sleep_fn = sleep_fn
        self.install_signal_handlers = install_signal_handlers
        self.stop_requested = False
        self.brain_service = brain_service or BrainService(repository, config.brain)
        if not isinstance(self.brain_service.adapter, BrainApiAdapter):
            raise ValueError("run-service requires `brain.backend: api` and a BrainApiAdapter backend.")
        self.session_manager = SessionManager(
            self.brain_service.adapter,
            persona_retry_interval_seconds=config.service.persona_retry_interval_seconds,
        )
        self.notification_manager = NotificationManager(
            self.brain_service.adapter,
            persona_email_cooldown_seconds=config.service.persona_email_cooldown_seconds,
            persona_confirmation_required=config.service.persona_confirmation_required,
            persona_confirmation_prompt_cooldown_seconds=(
                config.service.persona_confirmation_prompt_cooldown_seconds
            ),
            persona_confirmation_granted_ttl_seconds=config.service.persona_confirmation_granted_ttl_seconds,
        )
        self.scheduler = ServiceScheduler(config.service)
        self.heartbeat = HeartbeatReporter(repository.service_runtime)
        self.worker: ServiceWorker | None = None
        self._last_tick_was_noop = False

    def run(self, *, max_ticks: int | None = None) -> ServiceRunSummary:
        environment = self._resolve_service_environment()
        self.environment = environment
        logger = get_logger(__name__, run_id=environment.context.run_id, stage="service")
        progress_log_path = resolve_progress_log_path(self.config, environment)
        progress_log_path_str = str(progress_log_path) if progress_log_path is not None else None
        init_run(self.repository, self.config, environment, status="running_service")
        if progress_log_path_str:
            logger.info("Progress log path=%s", progress_log_path_str)
        append_progress_event(
            self.config,
            environment,
            event="service_run_started",
            stage="service",
            status="running_service",
            payload={
                "service_name": self.config.service.lock_name,
                "max_ticks": max_ticks,
                "pending_job_limit": self.config.service.max_pending_jobs,
            },
        )

        runtime_lock = RuntimeLock.create(
            self.repository.service_runtime,
            service_name=self.config.service.lock_name,
            service_run_id=environment.context.run_id,
            lease_seconds=self.config.service.lock_lease_seconds,
        )
        if not runtime_lock.acquire(status="starting"):
            logger.error("Service already active for lock=%s", self.config.service.lock_name)
            append_progress_event(
                self.config,
                environment,
                event="service_run_lock_denied",
                stage="service",
                status="lock_denied",
                payload={"service_name": self.config.service.lock_name},
            )
            return ServiceRunSummary(
                run_id=environment.context.run_id,
                service_name=self.config.service.lock_name,
                status="lock_denied",
                ticks_executed=0,
                pending_job_count=0,
                progress_log_path=progress_log_path_str,
            )

        now = _utcnow()
        runtime = self._ensure_runtime_row(
            service_name=self.config.service.lock_name,
            service_run_id=environment.context.run_id,
            owner_token=runtime_lock.owner_token,
            pid=runtime_lock.pid,
            hostname=runtime_lock.hostname,
            started_at=environment.context.started_at,
            now=now,
        )
        self.worker = ServiceWorker(
            self.repository,
            config=self.config,
            environment=environment,
            brain_service=self.brain_service,
            session_manager=self.session_manager,
            notification_manager=self.notification_manager,
        )
        if self.install_signal_handlers:
            self._install_signal_handlers()

        ticks_executed = 0
        final_status = "service_running"
        try:
            while not self.stop_requested:
                if max_ticks is not None and ticks_executed >= max_ticks:
                    final_status = runtime.status if runtime.status.startswith("service_") else f"service_{runtime.status}"
                    break
                if ticks_executed > 0 and not runtime_lock.renew():
                    logger.error("Lost runtime lease for service lock=%s", self.config.service.lock_name)
                    final_status = "lock_lost"
                    break

                runtime = self.repository.service_runtime.get_state(self.config.service.lock_name) or runtime
                tick_id = runtime.tick_id + 1
                suppress_noop_progress = self._last_tick_was_noop

                # --- Persona wait timeout auto-recovery ---
                if runtime.persona_wait_started_at and runtime.status in (
                    "waiting_persona",
                    "waiting_persona_confirmation",
                ):
                    wait_started = datetime.fromisoformat(runtime.persona_wait_started_at)
                    wait_elapsed = (datetime.now(UTC) - wait_started).total_seconds()
                    max_wait = self.config.service.max_persona_wait_seconds
                    if wait_elapsed > max_wait:
                        logger.warning(
                            "Persona wait exceeded max_persona_wait_seconds=%s (elapsed=%.0fs). "
                            "Resetting persona state and forcing re-login attempt.",
                            max_wait,
                            wait_elapsed,
                        )
                        self.repository.service_runtime.update_state(
                            runtime.service_name,
                            status="running",
                            persona_url=None,
                            persona_wait_started_at=None,
                            persona_last_notification_at=None,
                            persona_confirmation_nonce=None,
                            persona_confirmation_last_prompt_at=None,
                            persona_confirmation_granted_at=None,
                            updated_at=_utcnow(),
                        )
                        runtime = self.repository.service_runtime.get_state(self.config.service.lock_name) or runtime

                logger.info("Service tick starting active_batch=%s pending=%s", runtime.active_batch_id, runtime.pending_job_count)
                if not suppress_noop_progress:
                    append_progress_event(
                        self.config,
                        environment,
                        event="service_tick_started",
                        stage="service",
                        status=runtime.status,
                        tick_id=tick_id,
                        batch_id=runtime.active_batch_id,
                        payload={
                            "pending_job_count": runtime.pending_job_count,
                        },
                    )
                try:
                    outcome = self.worker.run_tick(
                        runtime=runtime,
                        tick_id=tick_id,
                        stop_requested=self.stop_requested,
                    )
                    self._last_tick_was_noop = self._is_noop_tick(outcome)
                    runtime = self.heartbeat.record_tick(runtime=runtime, outcome=outcome, tick_id=tick_id)
                    self.repository.update_run_status(environment.context.run_id, f"service_{outcome.status}")
                    final_status = f"service_{outcome.status}"
                    append_progress_event(
                        self.config,
                        environment,
                        event="service_tick_completed",
                        stage="service",
                        status=outcome.status,
                        tick_id=tick_id,
                        batch_id=outcome.active_batch_id,
                        payload={
                            "pending_job_count": outcome.pending_job_count,
                            "queue_depth": outcome.queue_depth,
                            "queue_counts": dict(outcome.queue_counts),
                            "generated_count": outcome.generated_count,
                            "submitted_count": outcome.submitted_count,
                            "completed_count": outcome.completed_count,
                            "failed_count": outcome.failed_count,
                            "quarantined_count": outcome.quarantined_count,
                            "poll_pending_ms": outcome.poll_pending_ms,
                            "prepare_batch_ms": outcome.prepare_batch_ms,
                            "submit_batch_ms": outcome.submit_batch_ms,
                            "pre_prepare_pending_job_count": outcome.pre_prepare_pending_job_count,
                            "next_sleep_seconds": outcome.next_sleep_seconds,
                            "cooldown_until": outcome.cooldown_until,
                            "last_error": outcome.last_error,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    self._last_tick_was_noop = False
                    now = _utcnow()
                    consecutive_failures = runtime.consecutive_failures + 1
                    cooldown_until = None
                    if consecutive_failures >= self.config.service.max_consecutive_failures:
                        cooldown_until = _shift(now, self.config.service.cooldown_seconds)
                        final_status = "service_cooldown"
                    else:
                        final_status = "service_error"
                    runtime = self.heartbeat.record_failure(
                        runtime=runtime,
                        tick_id=tick_id,
                        last_error=str(exc),
                        consecutive_failures=consecutive_failures,
                        cooldown_until=cooldown_until,
                    )
                    self.repository.update_run_status(environment.context.run_id, final_status)
                    logger.exception("Service tick failed")
                    append_progress_event(
                        self.config,
                        environment,
                        event="service_tick_failed",
                        stage="service",
                        status=final_status,
                        tick_id=tick_id,
                        batch_id=runtime.active_batch_id,
                        payload={
                            "error": str(exc),
                            "consecutive_failures": consecutive_failures,
                            "cooldown_until": cooldown_until,
                        },
                    )

                ticks_executed += 1
                if self.stop_requested:
                    break
                self.scheduler.record_poll_result(outcome.new_result_count)
                sleep_seconds = self.scheduler.next_sleep_seconds(runtime=runtime, outcome=self._runtime_to_outcome(runtime))
                logger.info("Service sleeping for %ss", sleep_seconds)
                if not suppress_noop_progress:
                    append_progress_event(
                        self.config,
                        environment,
                        event="service_sleeping",
                        stage="service",
                        status=runtime.status,
                        tick_id=runtime.tick_id,
                        batch_id=runtime.active_batch_id,
                        payload={"sleep_seconds": sleep_seconds},
                    )
                self._sleep_interruptibly(float(sleep_seconds))
        finally:
            runtime = self.repository.service_runtime.get_state(self.config.service.lock_name) or runtime
            pending_on_shutdown = int(runtime.pending_job_count or 0)
            if pending_on_shutdown > 0:
                stopped_status = "service_stopped_pending"
                finished = False
            elif self.stop_requested:
                stopped_status = "service_stopped"
                finished = True
            else:
                stopped_status = final_status
                finished = True
            self.heartbeat.record_shutdown(runtime=runtime, status=stopped_status)
            runtime_lock.release(status=stopped_status)
            self.repository.update_run_status(
                environment.context.run_id,
                stopped_status,
                finished=finished,
            )
            append_progress_event(
                self.config,
                environment,
                event="service_run_finished",
                stage="service",
                status=stopped_status,
                tick_id=runtime.tick_id,
                batch_id=runtime.active_batch_id,
                payload={
                    "ticks_executed": ticks_executed,
                    "pending_job_count": runtime.pending_job_count,
                },
            )

        refreshed = self.repository.service_runtime.get_state(self.config.service.lock_name) or runtime
        return ServiceRunSummary(
            run_id=environment.context.run_id,
            service_name=self.config.service.lock_name,
            status=refreshed.status,
            ticks_executed=ticks_executed,
            pending_job_count=refreshed.pending_job_count,
            progress_log_path=progress_log_path_str,
        )

    def request_shutdown(self, signum: int | None = None, frame: FrameType | None = None) -> None:
        del frame
        logger = get_logger(__name__, run_id=self.environment.context.run_id, stage="service")
        if not self.stop_requested:
            logger.warning("Shutdown requested for service via signal=%s", signum)
        self.stop_requested = True

    def _resolve_service_environment(self) -> CommandEnvironment:
        existing = self.repository.service_runtime.get_state(self.config.service.lock_name)
        if existing and self.config.service.resume_incomplete_jobs and existing.service_run_id:
            run = self.repository.get_run(existing.service_run_id)
            started_at = run.started_at if run is not None else existing.started_at
            context = RunContext(
                run_id=existing.service_run_id,
                seed=self.environment.context.seed,
                started_at=started_at,
                config_path=self.environment.context.config_path,
            )
            return CommandEnvironment(
                config_path=self.environment.config_path,
                command_name=self.environment.command_name,
                context=context,
            )
        return self.environment

    def _ensure_runtime_row(
        self,
        *,
        service_name: str,
        service_run_id: str,
        owner_token: str,
        pid: int,
        hostname: str,
        started_at: str,
        now: str,
    ) -> ServiceRuntimeRecord:
        current = self.repository.service_runtime.get_state(service_name)
        if current is not None:
            self.repository.service_runtime.update_state(
                service_name,
                service_run_id=service_run_id,
                owner_token=owner_token,
                pid=pid,
                hostname=hostname,
                status="running",
                updated_at=now,
            )
            refreshed = self.repository.service_runtime.get_state(service_name)
            if refreshed is None:
                raise RuntimeError(f"Unable to refresh runtime row for service={service_name}")
            return refreshed

        record = ServiceRuntimeRecord(
            service_name=service_name,
            service_run_id=service_run_id,
            owner_token=owner_token,
            pid=pid,
            hostname=hostname,
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
            persona_confirmation_nonce=None,
            persona_confirmation_last_prompt_at=None,
            persona_confirmation_granted_at=None,
            persona_confirmation_last_update_id=None,
            counters_json="{}",
            lock_expires_at=_shift(now, self.config.service.lock_lease_seconds),
            started_at=started_at,
            updated_at=now,
        )
        self.repository.service_runtime.upsert_state(record)
        refreshed = self.repository.service_runtime.get_state(service_name)
        if refreshed is None:
            raise RuntimeError(f"Unable to create runtime row for service={service_name}")
        return refreshed

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.request_shutdown)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self.request_shutdown)

    def _sleep_interruptibly(self, seconds: float) -> None:
        remaining = max(float(seconds), 0.0)
        if remaining <= 0:
            return
        deadline = time.monotonic() + remaining
        while not self.stop_requested:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                return
            self.sleep_fn(min(remaining, 1.0))

    @staticmethod
    def _runtime_to_outcome(runtime: ServiceRuntimeRecord):
        from services.models import ServiceTickOutcome

        return ServiceTickOutcome(
            status=runtime.status,
            pending_job_count=runtime.pending_job_count,
            active_batch_id=runtime.active_batch_id,
            cooldown_until=runtime.cooldown_until,
            persona_url=runtime.persona_url,
        )

    @staticmethod
    def _is_noop_tick(outcome) -> bool:
        return (
            outcome.generated_count == 0
            and outcome.completed_count == 0
            and outcome.failed_count == 0
            and outcome.submitted_count == 0
        )


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _shift(timestamp: str, seconds: int) -> str:
    return (datetime.fromisoformat(timestamp) + timedelta(seconds=seconds)).isoformat()
