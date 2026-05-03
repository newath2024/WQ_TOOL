from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QualityScoreConfig:
    check_penalty_weight: float = 1.0
    check_warning_weight: float = 0.5
    rejection_penalty_weight: float = 1.0
    base_rejection_penalty: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "check_penalty_weight",
            "check_warning_weight",
            "rejection_penalty_weight",
            "base_rejection_penalty",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"quality_score.{name} must be >= 0")
            setattr(self, name, value)


__all__ = ["QualityScoreConfig"]
