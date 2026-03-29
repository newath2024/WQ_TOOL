from __future__ import annotations

from datetime import UTC, datetime

from adapters.brain_api_adapter import BrainApiAdapter
from storage.models import ServiceRuntimeRecord


class NotificationManager:
    def __init__(
        self,
        adapter: BrainApiAdapter,
        *,
        persona_email_cooldown_seconds: int,
    ) -> None:
        self.adapter = adapter
        self.persona_email_cooldown_seconds = persona_email_cooldown_seconds

    def notify_persona_required(self, *, runtime: ServiceRuntimeRecord, persona_url: str) -> tuple[bool, str]:
        now = datetime.now(UTC).isoformat()
        print("BRAIN requires biometric authentication for service mode.")
        print(f"Open this URL to complete face scan: {persona_url}")
        if not self._should_send(runtime, now):
            return False, now
        sent = self.adapter.send_persona_notification(persona_url)
        if sent:
            print("Sent Persona verification link via email.")
        else:
            print("Persona email notification is not configured; skipping email alert.")
        return sent, now

    def _should_send(self, runtime: ServiceRuntimeRecord, now: str) -> bool:
        if not runtime.persona_last_notification_at:
            return True
        elapsed = datetime.fromisoformat(now) - datetime.fromisoformat(runtime.persona_last_notification_at)
        return elapsed.total_seconds() >= float(self.persona_email_cooldown_seconds)
