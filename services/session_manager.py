from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import requests

from adapters.brain_api_adapter import BrainApiAdapter, BiometricsThrottled, PersonaVerificationRequired
from storage.models import ServiceRuntimeRecord


@dataclass(slots=True, frozen=True)
class SessionState:
    status: str
    persona_url: str | None = None
    session_path: str | None = None
    detail: str | None = None
    retry_after_seconds: int | None = None


class SessionManager:
    def __init__(self, adapter: BrainApiAdapter, *, persona_retry_interval_seconds: int) -> None:
        self.adapter = adapter
        self.persona_retry_interval_seconds = persona_retry_interval_seconds

    def ensure_session(self, *, runtime: ServiceRuntimeRecord, force: bool = False) -> SessionState:
        if not force:
            resumed = self._resume_pending_persona(runtime)
            if resumed is not None:
                return resumed
        try:
            result = self.adapter.ensure_authenticated(force=force, interactive=False)
        except PersonaVerificationRequired as exc:
            return SessionState(status="waiting_persona", persona_url=exc.persona_url, detail=str(exc))
        except BiometricsThrottled as exc:
            return SessionState(
                status="auth_throttled",
                detail=str(exc),
                retry_after_seconds=exc.retry_after_seconds,
            )
        except requests.RequestException as exc:
            return SessionState(
                status="auth_unavailable",
                detail=f"BRAIN auth transport error: {exc}",
            )
        return SessionState(
            status="ready",
            persona_url=None,
            session_path=str(result.get("session_path") or "") or None,
            detail=str(result.get("mode") or "non_interactive"),
        )

    def _resume_pending_persona(self, runtime: ServiceRuntimeRecord) -> SessionState | None:
        persona_url = str(runtime.persona_url or "").strip()
        if not persona_url:
            return None
        if runtime.status != "waiting_persona":
            return None
        if self._persona_wait_timed_out(runtime):
            return None
        retry_after = self._persona_retry_after_seconds(runtime)
        if retry_after > 0:
            return SessionState(
                status="waiting_persona",
                persona_url=persona_url,
                detail="Waiting for the current BRAIN Persona inquiry to complete.",
                retry_after_seconds=retry_after,
            )
        try:
            result = self.adapter.resume_persona_authentication(persona_url)
        except BiometricsThrottled as exc:
            return SessionState(
                status="auth_throttled",
                detail=str(exc),
                retry_after_seconds=exc.retry_after_seconds,
            )
        except requests.RequestException as exc:
            return SessionState(
                status="auth_unavailable",
                detail=f"BRAIN auth transport error: {exc}",
            )
        if str(result.get("status") or "") == "ready":
            return SessionState(
                status="ready",
                persona_url=None,
                session_path=str(result.get("session_path") or "") or None,
                detail=str(result.get("mode") or "session_cookie"),
            )
        if bool(result.get("expired")):
            return None
        return SessionState(
            status="waiting_persona",
            persona_url=persona_url,
            detail=str(result.get("detail") or "Waiting for BRAIN Persona verification to complete."),
            retry_after_seconds=self.persona_retry_interval_seconds,
        )

    def _persona_retry_after_seconds(self, runtime: ServiceRuntimeRecord) -> int:
        reference = runtime.persona_wait_started_at or runtime.updated_at
        if not reference:
            return 0
        try:
            started_at = datetime.fromisoformat(reference)
        except ValueError:
            return 0
        elapsed = (datetime.now(UTC) - started_at).total_seconds()
        remaining = int(self.persona_retry_interval_seconds - elapsed)
        return max(remaining, 0)

    def _persona_wait_timed_out(self, runtime: ServiceRuntimeRecord) -> bool:
        if not runtime.persona_wait_started_at:
            return False
        try:
            started_at = datetime.fromisoformat(runtime.persona_wait_started_at)
        except ValueError:
            return False
        elapsed = (datetime.now(UTC) - started_at).total_seconds()
        return elapsed >= float(self.adapter.persona_timeout_seconds)
