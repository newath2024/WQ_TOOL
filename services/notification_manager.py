from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

from adapters.brain_api_adapter import BrainApiAdapter
from storage.models import ServiceRuntimeRecord


@dataclass(slots=True, frozen=True)
class PersonaConfirmationDecision:
    status: str
    nonce: str | None = None
    last_prompt_at: str | None = None
    granted_at: str | None = None
    last_update_id: int | None = None
    detail: str | None = None


class NotificationManager:
    def __init__(
        self,
        adapter: BrainApiAdapter,
        *,
        persona_email_cooldown_seconds: int,
        persona_confirmation_required: bool = False,
        persona_confirmation_prompt_cooldown_seconds: int = 1800,
        persona_confirmation_granted_ttl_seconds: int = 300,
    ) -> None:
        self.adapter = adapter
        self.persona_email_cooldown_seconds = persona_email_cooldown_seconds
        self.persona_confirmation_required = persona_confirmation_required
        self.persona_confirmation_prompt_cooldown_seconds = persona_confirmation_prompt_cooldown_seconds
        self.persona_confirmation_granted_ttl_seconds = persona_confirmation_granted_ttl_seconds

    def notify_persona_required(self, *, runtime: ServiceRuntimeRecord, persona_url: str) -> tuple[bool, str]:
        now = datetime.now(UTC).isoformat()
        print("BRAIN requires biometric authentication for service mode.")
        print(f"Open this URL to complete face scan: {persona_url}")
        if not self._should_send(runtime, now, persona_url):
            return False, now
        sent = self.adapter.send_persona_notification(persona_url)
        if sent:
            print("Sent Persona verification link via Telegram notifier.")
        else:
            print("Persona notification is not configured; skipping alert.")
        return sent, now

    def request_persona_confirmation(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        service_name: str,
    ) -> PersonaConfirmationDecision:
        now = datetime.now(UTC).isoformat()
        if not self.persona_confirmation_required:
            return PersonaConfirmationDecision(
                status="bypass",
                detail="Persona confirmation gate is disabled by config.",
            )
        if not self.adapter.supports_persona_confirmation():
            return PersonaConfirmationDecision(
                status="bypass",
                detail="Telegram confirmation is not configured; falling back to direct auth request.",
            )

        granted_at = self._active_granted_at(runtime=runtime, now=now)
        if granted_at is not None:
            return PersonaConfirmationDecision(
                status="approved",
                nonce=runtime.persona_confirmation_nonce,
                last_prompt_at=runtime.persona_confirmation_last_prompt_at,
                granted_at=granted_at,
                last_update_id=runtime.persona_confirmation_last_update_id,
                detail="Telegram confirmation already received; proceeding to request Persona link.",
            )

        nonce = self._current_prompt_nonce(runtime=runtime, now=now)
        last_update_id = runtime.persona_confirmation_last_update_id
        try:
            poll_result = self.adapter.poll_persona_confirmation(
                prompt_token=nonce,
                last_update_id=last_update_id,
            )
        except Exception as exc:  # noqa: BLE001
            return PersonaConfirmationDecision(
                status="pending",
                nonce=nonce,
                last_prompt_at=runtime.persona_confirmation_last_prompt_at,
                granted_at=None,
                last_update_id=last_update_id,
                detail=f"Telegram confirmation check failed: {exc}",
            )

        last_update_id = poll_result.get("last_update_id")
        if last_update_id is None:
            last_update_id = runtime.persona_confirmation_last_update_id
        if bool(poll_result.get("approved")):
            return PersonaConfirmationDecision(
                status="approved",
                nonce=nonce,
                last_prompt_at=runtime.persona_confirmation_last_prompt_at,
                granted_at=now,
                last_update_id=last_update_id,
                detail="Telegram confirmation received; requesting Persona link now.",
            )

        last_prompt_at = runtime.persona_confirmation_last_prompt_at
        detail = "Waiting for Telegram confirmation before requesting a new Persona link."
        if self._should_send_confirmation_prompt(runtime=runtime, now=now, nonce=nonce):
            try:
                sent = self.adapter.send_persona_confirmation_prompt(
                    prompt_token=nonce,
                    service_name=service_name,
                )
            except Exception as exc:  # noqa: BLE001
                sent = False
                detail = f"Telegram confirmation prompt failed: {exc}"
            if sent:
                last_prompt_at = now
                detail = (
                    "Sent Telegram readiness prompt. "
                    "No Persona link will be requested until confirmation arrives."
                )

        return PersonaConfirmationDecision(
            status="pending",
            nonce=nonce,
            last_prompt_at=last_prompt_at,
            granted_at=None,
            last_update_id=last_update_id,
            detail=detail,
        )

    def _should_send(self, runtime: ServiceRuntimeRecord, now: str, persona_url: str) -> bool:
        current_url = str(runtime.persona_url or "").strip()
        next_url = str(persona_url or "").strip()
        if next_url and next_url != current_url:
            return True
        if not runtime.persona_last_notification_at:
            return True
        elapsed = datetime.fromisoformat(now) - datetime.fromisoformat(runtime.persona_last_notification_at)
        return elapsed.total_seconds() >= float(self.persona_email_cooldown_seconds)

    def _should_send_confirmation_prompt(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        now: str,
        nonce: str,
    ) -> bool:
        if nonce != str(runtime.persona_confirmation_nonce or ""):
            return True
        if not runtime.persona_confirmation_last_prompt_at:
            return True
        try:
            last_prompt_at = datetime.fromisoformat(runtime.persona_confirmation_last_prompt_at)
        except ValueError:
            return True
        elapsed = datetime.fromisoformat(now) - last_prompt_at
        return elapsed.total_seconds() >= float(self.persona_confirmation_prompt_cooldown_seconds)

    def _active_granted_at(self, *, runtime: ServiceRuntimeRecord, now: str) -> str | None:
        granted_at = str(runtime.persona_confirmation_granted_at or "").strip()
        if not granted_at:
            return None
        try:
            granted_ts = datetime.fromisoformat(granted_at)
        except ValueError:
            return None
        elapsed = datetime.fromisoformat(now) - granted_ts
        if elapsed.total_seconds() >= float(self.persona_confirmation_granted_ttl_seconds):
            return None
        return granted_at

    @staticmethod
    def _current_prompt_nonce(*, runtime: ServiceRuntimeRecord, now: str) -> str:
        existing_nonce = str(runtime.persona_confirmation_nonce or "").strip()
        if not existing_nonce:
            return secrets.token_hex(4)
        granted_at = str(runtime.persona_confirmation_granted_at or "").strip()
        if not granted_at:
            return existing_nonce
        try:
            granted_ts = datetime.fromisoformat(granted_at)
        except ValueError:
            return secrets.token_hex(4)
        if (datetime.fromisoformat(now) - granted_ts).total_seconds() < 0:
            return secrets.token_hex(4)
        return secrets.token_hex(4)
