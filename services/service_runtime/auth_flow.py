from __future__ import annotations

from datetime import UTC, datetime

from services.models import ServiceTickOutcome
from storage.models import (
    ServiceRuntimeRecord,
)

class AuthFlow:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def prepare_session(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        now: str,
        logger,
    ) -> tuple[ServiceRuntimeRecord, bool, ServiceTickOutcome | None]:
        should_probe_auth = self._should_probe_session_during_cooldown(runtime)
        submission_cooldown_active = bool(runtime.cooldown_until and runtime.cooldown_until > now and not should_probe_auth)
        if submission_cooldown_active:
            logger.info(
                "Service is cooling down until %s; polling existing work but skipping new submissions.",
                runtime.cooldown_until,
            )
        if runtime.cooldown_until and runtime.cooldown_until > now and should_probe_auth:
            logger.info("Auth-related cooldown detected; probing session state before waiting out cooldown.")

        session_state = self.session_manager.ensure_session(runtime=runtime, allow_new_login=False)
        if session_state.status == "authentication_required":
            confirmation = self.notification_manager.request_persona_confirmation(
                runtime=runtime,
                service_name=runtime.service_name,
            )
            self._update_persona_confirmation_state(
                service_name=runtime.service_name,
                now=now,
                nonce=confirmation.nonce,
                last_prompt_at=confirmation.last_prompt_at,
                granted_at=confirmation.granted_at,
                last_update_id=confirmation.last_update_id,
            )
            runtime = self.repository.service_runtime.get_state(runtime.service_name) or runtime
            if confirmation.status == "pending":
                pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
                logger.warning(
                    "Waiting for Telegram confirmation before requesting Persona link; pending_jobs=%s",
                    pending_jobs,
                )
                return (
                    runtime,
                    submission_cooldown_active,
                    self._defer_pending_timeouts_for_wait(
                        run_id=runtime.service_run_id,
                        updated_at=now,
                        outcome=ServiceTickOutcome(
                            status="waiting_persona_confirmation",
                            pending_job_count=pending_jobs,
                            active_batch_id=runtime.active_batch_id,
                            persona_url=None,
                            last_error=confirmation.detail,
                            next_sleep_seconds=self.config.service.persona_confirmation_poll_interval_seconds,
                        ),
                    ),
                )
            session_state = self.session_manager.ensure_session(runtime=runtime, allow_new_login=True)
        if session_state.status == "waiting_persona":
            if self.notification_manager.requires_fresh_persona_confirmation_for_url(
                runtime=runtime,
                persona_url=str(session_state.persona_url or ""),
                now=now,
            ):
                confirmation = self.notification_manager.request_persona_confirmation(
                    runtime=runtime,
                    service_name=runtime.service_name,
                )
                self._update_persona_confirmation_state(
                    service_name=runtime.service_name,
                    now=now,
                    nonce=confirmation.nonce,
                    last_prompt_at=confirmation.last_prompt_at,
                    granted_at=confirmation.granted_at,
                    last_update_id=confirmation.last_update_id,
                )
                runtime = self.repository.service_runtime.get_state(runtime.service_name) or runtime
                if confirmation.status == "pending":
                    pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
                    logger.warning(
                        "Waiting for Telegram confirmation before delivering Persona link; pending_jobs=%s",
                        pending_jobs,
                    )
                    return (
                        runtime,
                        submission_cooldown_active,
                        self._defer_pending_timeouts_for_wait(
                            run_id=runtime.service_run_id,
                            updated_at=now,
                            outcome=ServiceTickOutcome(
                                status="waiting_persona_confirmation",
                                pending_job_count=pending_jobs,
                                active_batch_id=runtime.active_batch_id,
                                persona_url=None,
                                last_error=confirmation.detail,
                                next_sleep_seconds=self.config.service.persona_confirmation_poll_interval_seconds,
                            ),
                        ),
                    )
            persona_wait_started_at = runtime.persona_wait_started_at or now
            if session_state.persona_url and session_state.persona_url != runtime.persona_url:
                persona_wait_started_at = now
            sent, notification_at = self.notification_manager.notify_persona_required(
                runtime=runtime,
                persona_url=str(session_state.persona_url or ""),
            )
            self.repository.service_runtime.update_state(
                runtime.service_name,
                status="waiting_persona",
                persona_url=session_state.persona_url,
                persona_wait_started_at=persona_wait_started_at,
                persona_last_notification_at=notification_at if sent else runtime.persona_last_notification_at,
                persona_confirmation_nonce=None,
                persona_confirmation_last_prompt_at=None,
                persona_confirmation_granted_at=None,
                updated_at=now,
            )
            pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
            logger.warning("Persona verification required; pending_jobs=%s", pending_jobs)
            return (
                runtime,
                submission_cooldown_active,
                self._defer_pending_timeouts_for_wait(
                    run_id=runtime.service_run_id,
                    updated_at=now,
                    outcome=ServiceTickOutcome(
                        status="waiting_persona",
                        pending_job_count=pending_jobs,
                        active_batch_id=runtime.active_batch_id,
                        persona_url=session_state.persona_url,
                        last_error=session_state.detail,
                        next_sleep_seconds=(
                            session_state.retry_after_seconds or self.config.service.persona_retry_interval_seconds
                        ),
                    ),
                ),
            )
        if session_state.status == "auth_throttled":
            self.repository.service_runtime.update_state(
                runtime.service_name,
                status="auth_throttled",
                persona_url=None,
                persona_wait_started_at=runtime.persona_wait_started_at or now,
                updated_at=now,
            )
            pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
            logger.warning(
                "BRAIN biometrics throttled; pending_jobs=%s retry_after=%s",
                pending_jobs,
                session_state.retry_after_seconds,
            )
            return (
                runtime,
                submission_cooldown_active,
                self._defer_pending_timeouts_for_wait(
                    run_id=runtime.service_run_id,
                    updated_at=now,
                    outcome=ServiceTickOutcome(
                        status="auth_throttled",
                        pending_job_count=pending_jobs,
                        active_batch_id=runtime.active_batch_id,
                        persona_url=None,
                        last_error=session_state.detail,
                        next_sleep_seconds=(
                            session_state.retry_after_seconds or self.config.service.persona_retry_interval_seconds
                        ),
                    ),
                ),
            )
        if session_state.status == "auth_unavailable":
            self.repository.service_runtime.update_state(
                runtime.service_name,
                status="auth_unavailable",
                persona_url=None,
                persona_wait_started_at=runtime.persona_wait_started_at,
                last_error=session_state.detail,
                updated_at=now,
            )
            pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
            logger.warning(
                "BRAIN auth probe failed; pending_jobs=%s detail=%s",
                pending_jobs,
                session_state.detail,
            )
            return (
                runtime,
                submission_cooldown_active,
                self._defer_pending_timeouts_for_wait(
                    run_id=runtime.service_run_id,
                    updated_at=now,
                    outcome=ServiceTickOutcome(
                        status="auth_unavailable",
                        pending_job_count=pending_jobs,
                        active_batch_id=runtime.active_batch_id,
                        persona_url=None,
                        last_error=session_state.detail,
                        next_sleep_seconds=self.config.service.tick_interval_seconds,
                    ),
                ),
            )

        if runtime.cooldown_until and runtime.cooldown_until > now and should_probe_auth:
            logger.info("Authentication recovered during cooldown; resuming normal service work.")
        if (
            runtime.persona_url
            or runtime.persona_wait_started_at
            or runtime.persona_confirmation_nonce
            or runtime.persona_confirmation_granted_at
        ):
            self.repository.service_runtime.update_state(
                runtime.service_name,
                status="running",
                persona_url=None,
                persona_wait_started_at=None,
                persona_last_notification_at=None,
                persona_confirmation_nonce=None,
                persona_confirmation_last_prompt_at=None,
                persona_confirmation_granted_at=None,
                persona_confirmation_last_update_id=None,
                updated_at=now,
            )
        return runtime, submission_cooldown_active, None

    def _probe_auth_after_batch_failures(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        logger,
    ) -> ServiceTickOutcome | None:
        threshold = self.config.service.max_consecutive_batch_failures_before_auth_check
        recent_all_failed = self.repository.submissions.count_recent_all_failed_batches(
            runtime.service_run_id,
            lookback=threshold + 1,
        )
        if recent_all_failed < threshold:
            return None
        logger.warning(
            "Detected %s consecutive batches with 100%% job failures. Probing auth session before preparing more work.",
            recent_all_failed,
        )
        session_state = self.session_manager.ensure_session(runtime=runtime, allow_new_login=False)
        if session_state.status == "ready":
            return None
        logger.warning(
            "Auth probe after batch failures detected session status=%s. Triggering re-authentication flow.",
            session_state.status,
        )
        now = datetime.now(UTC).isoformat()
        self.repository.service_runtime.update_state(
            runtime.service_name,
            status="waiting_persona",
            persona_wait_started_at=now,
            updated_at=now,
        )
        pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
        return self._with_queue_metrics(
            self._defer_pending_timeouts_for_wait(
                run_id=runtime.service_run_id,
                updated_at=now,
                outcome=ServiceTickOutcome(
                    status="waiting_persona",
                    pending_job_count=pending_jobs,
                    active_batch_id=runtime.active_batch_id,
                    last_error=f"Auth re-check triggered after {recent_all_failed} consecutive all-failed batches.",
                    next_sleep_seconds=self.config.service.persona_retry_interval_seconds,
                ),
            ),
            service_name=runtime.service_name,
            run_id=runtime.service_run_id,
        )

    @staticmethod
    def _should_probe_session_during_cooldown(runtime: ServiceRuntimeRecord) -> bool:
        if runtime.status in {
            "waiting_persona",
            "waiting_persona_confirmation",
            "auth_throttled",
            "auth_unavailable",
        }:
            return True
        if runtime.persona_url or runtime.persona_wait_started_at or runtime.persona_confirmation_nonce:
            return True
        last_error = str(runtime.last_error or "").upper()
        return any(
            token in last_error
            for token in (
                "BRAIN AUTHENTICATION FAILED",
                "BRAIN AUTH TRANSPORT ERROR",
                "BIOMETRICS_THROTTLED",
                "PERSONA",
            )
        )

    def _update_persona_confirmation_state(
        self,
        *,
        service_name: str,
        now: str,
        nonce: str | None,
        last_prompt_at: str | None,
        granted_at: str | None,
        last_update_id: int | None,
    ) -> None:
        self.repository.service_runtime.update_state(
            service_name,
            persona_confirmation_nonce=nonce,
            persona_confirmation_last_prompt_at=last_prompt_at,
            persona_confirmation_granted_at=granted_at,
            persona_confirmation_last_update_id=last_update_id,
            updated_at=now,
        )
