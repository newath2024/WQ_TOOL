from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ServiceConfig:
    enabled: bool = False
    tick_interval_seconds: int = 5
    idle_sleep_seconds: int = 30
    poll_interval_seconds: int = 15
    max_pending_jobs: int = 20
    max_consecutive_failures: int = 5
    cooldown_seconds: int = 300
    heartbeat_interval_seconds: int = 30
    lock_name: str = "brain-service"
    lock_lease_seconds: int = 60
    resume_incomplete_jobs: bool = True
    shutdown_grace_period_seconds: int = 30
    stuck_job_after_seconds: int = 1800
    persona_retry_interval_seconds: int = 300
    persona_email_cooldown_seconds: int = 900
    persona_confirmation_required: bool = True
    persona_confirmation_poll_interval_seconds: int = 30
    persona_confirmation_prompt_cooldown_seconds: int = 1800
    persona_confirmation_granted_ttl_seconds: int = 300
    max_persona_wait_seconds: int = 1800
    max_consecutive_batch_failures_before_auth_check: int = 2
    ambiguous_submission_policy: str = "fail"
    research_context_cache_enabled: bool = True
    research_context_cache_ttl_seconds: int = 0
    observed_limit_ttl_seconds: int = 1800
    observed_limit_probe_interval_seconds: int = 300

    def __post_init__(self) -> None:
        self.ambiguous_submission_policy = (
            str(self.ambiguous_submission_policy or "fail").strip().lower()
        )
        if self.tick_interval_seconds <= 0:
            raise ValueError("service.tick_interval_seconds must be > 0")
        if self.idle_sleep_seconds <= 0:
            raise ValueError("service.idle_sleep_seconds must be > 0")
        if self.poll_interval_seconds <= 0:
            raise ValueError("service.poll_interval_seconds must be > 0")
        if self.max_pending_jobs <= 0:
            raise ValueError("service.max_pending_jobs must be > 0")
        if self.max_consecutive_failures <= 0:
            raise ValueError("service.max_consecutive_failures must be > 0")
        if self.cooldown_seconds <= 0:
            raise ValueError("service.cooldown_seconds must be > 0")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("service.heartbeat_interval_seconds must be > 0")
        if not self.lock_name.strip():
            raise ValueError("service.lock_name must not be empty")
        if self.lock_lease_seconds <= 0:
            raise ValueError("service.lock_lease_seconds must be > 0")
        if self.shutdown_grace_period_seconds <= 0:
            raise ValueError("service.shutdown_grace_period_seconds must be > 0")
        if self.stuck_job_after_seconds <= 0:
            raise ValueError("service.stuck_job_after_seconds must be > 0")
        if self.persona_retry_interval_seconds <= 0:
            raise ValueError("service.persona_retry_interval_seconds must be > 0")
        if self.persona_email_cooldown_seconds <= 0:
            raise ValueError("service.persona_email_cooldown_seconds must be > 0")
        if self.persona_confirmation_poll_interval_seconds <= 0:
            raise ValueError("service.persona_confirmation_poll_interval_seconds must be > 0")
        if self.persona_confirmation_prompt_cooldown_seconds <= 0:
            raise ValueError("service.persona_confirmation_prompt_cooldown_seconds must be > 0")
        if self.persona_confirmation_granted_ttl_seconds <= 0:
            raise ValueError("service.persona_confirmation_granted_ttl_seconds must be > 0")
        if self.max_persona_wait_seconds <= 0:
            raise ValueError("service.max_persona_wait_seconds must be > 0")
        if self.max_consecutive_batch_failures_before_auth_check <= 0:
            raise ValueError("service.max_consecutive_batch_failures_before_auth_check must be > 0")
        if self.ambiguous_submission_policy not in {"quarantine", "fail", "resubmit"}:
            raise ValueError(
                "service.ambiguous_submission_policy must be one of: quarantine, fail, resubmit"
            )
        if self.research_context_cache_ttl_seconds < 0:
            raise ValueError("service.research_context_cache_ttl_seconds must be >= 0")
        if self.observed_limit_ttl_seconds <= 0:
            raise ValueError("service.observed_limit_ttl_seconds must be > 0")
        if self.observed_limit_probe_interval_seconds <= 0:
            raise ValueError("service.observed_limit_probe_interval_seconds must be > 0")


__all__ = ["ServiceConfig"]
