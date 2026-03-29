from __future__ import annotations

from dataclasses import dataclass

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
    def __init__(self, adapter: BrainApiAdapter) -> None:
        self.adapter = adapter

    def ensure_session(self, *, runtime: ServiceRuntimeRecord, force: bool = False) -> SessionState:
        del runtime
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
