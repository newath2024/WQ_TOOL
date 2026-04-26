from __future__ import annotations


class PersonaVerificationRequired(RuntimeError):
    def __init__(self, persona_url: str) -> None:
        super().__init__("BRAIN Persona verification is required before API work can continue.")
        self.persona_url = persona_url


class BiometricsThrottled(RuntimeError):
    def __init__(self, detail: str, *, retry_after_seconds: int | None = None) -> None:
        super().__init__(f"BRAIN biometrics throttled: {detail}")
        self.detail = detail
        self.retry_after_seconds = retry_after_seconds


class ConcurrentSimulationLimitExceeded(RuntimeError):
    def __init__(self, detail: str, *, cooldown_seconds: int = 180) -> None:
        super().__init__(f"BRAIN concurrent simulation limit exceeded: {detail}")
        self.detail = detail
        self.cooldown_seconds = cooldown_seconds
