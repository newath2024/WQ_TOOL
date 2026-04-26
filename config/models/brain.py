from __future__ import annotations

from dataclasses import dataclass, field

from config.defaults import DEFAULT_SIMULATION_PROFILES
from config.validators import _normalize_brain_enum


@dataclass(slots=True)
class SimulationProfile:
    name: str
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    neutralization: str = "SUBINDUSTRY"
    decay: int = 3
    truncation: float = 0.01
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.name = str(self.name or "").strip()
        if not self.name:
            raise ValueError("brain.simulation_profiles[].name must not be empty")
        self.region = str(self.region).strip().upper()
        self.universe = str(self.universe).strip().upper()
        self.neutralization = _normalize_brain_enum(
            self.neutralization, true_value="ON", false_value="OFF"
        )
        if self.delay < 0:
            raise ValueError("brain.simulation_profiles[].delay must be >= 0")
        if self.decay < 0:
            raise ValueError("brain.simulation_profiles[].decay must be >= 0")
        if self.truncation < 0:
            raise ValueError("brain.simulation_profiles[].truncation must be >= 0")
        if self.weight < 0:
            raise ValueError("brain.simulation_profiles[].weight must be >= 0")


def _default_simulation_profiles() -> list[SimulationProfile]:
    return [SimulationProfile(**profile) for profile in DEFAULT_SIMULATION_PROFILES]


@dataclass(slots=True)
class BrainConfig:
    backend: str = "manual"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    neutralization: str = "SUBINDUSTRY"
    decay: int = 3
    truncation: float = 0.01
    simulation_profiles: list[SimulationProfile] = field(
        default_factory=_default_simulation_profiles
    )
    pasteurization: bool = True
    unit_handling: str = "verify"
    nan_handling: str = "off"
    poll_interval_seconds: int = 10
    timeout_seconds: int = 600
    max_retries: int = 3
    batch_size: int = 20
    manual_export_dir: str = "outputs/brain_manual"
    api_base_url: str = ""
    api_auth_env: str = "BRAIN_API_TOKEN"
    email_env: str = "BRAIN_API_EMAIL"
    password_env: str = "BRAIN_API_PASSWORD"
    credentials_file: str = "secrets/brain_credentials.json"
    session_path: str = "outputs/brain_api_session.json"
    auth_expiry_seconds: int = 14400
    open_browser_for_persona: bool = True
    persona_poll_interval_seconds: int = 15
    persona_timeout_seconds: int = 1800
    rate_limit_per_minute: int = 60

    def __post_init__(self) -> None:
        self.backend = str(self.backend).strip().lower()
        self.region = str(self.region).strip().upper()
        self.universe = str(self.universe).strip().upper()
        self.neutralization = _normalize_brain_enum(
            self.neutralization, true_value="ON", false_value="OFF"
        )
        self.simulation_profiles = [
            item if isinstance(item, SimulationProfile) else SimulationProfile(**item)
            for item in self.simulation_profiles
        ]
        self.unit_handling = _normalize_brain_enum(
            self.unit_handling, true_value="VERIFY", false_value="IGNORE"
        )
        self.nan_handling = _normalize_brain_enum(
            self.nan_handling, true_value="ON", false_value="OFF"
        )
        allowed_backends = {"manual", "api"}
        if self.backend not in allowed_backends:
            raise ValueError(f"brain.backend must be one of {sorted(allowed_backends)}")
        if self.delay < 0:
            raise ValueError("brain.delay must be >= 0")
        if self.poll_interval_seconds <= 0:
            raise ValueError("brain.poll_interval_seconds must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("brain.timeout_seconds must be > 0")
        if self.max_retries < 0:
            raise ValueError("brain.max_retries must be >= 0")
        if self.batch_size <= 0:
            raise ValueError("brain.batch_size must be > 0")
        if not 1 <= self.auth_expiry_seconds <= 14400:
            raise ValueError("brain.auth_expiry_seconds must be between 1 and 14400")
        if self.persona_poll_interval_seconds <= 0:
            raise ValueError("brain.persona_poll_interval_seconds must be > 0")
        if self.persona_timeout_seconds <= 0:
            raise ValueError("brain.persona_timeout_seconds must be > 0")
        if self.rate_limit_per_minute <= 0:
            raise ValueError("brain.rate_limit_per_minute must be > 0")


__all__ = ["SimulationProfile", "BrainConfig", "_default_simulation_profiles"]
