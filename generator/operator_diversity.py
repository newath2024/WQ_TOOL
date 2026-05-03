from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from random import Random
from typing import Iterable

from core.config import OperatorDiversityBoostConfig


@dataclass(slots=True)
class OperatorDiversityState:
    config: OperatorDiversityBoostConfig
    allowed_operators: set[str] = field(default_factory=set)
    operator_prior_multipliers: dict[str, float] = field(default_factory=dict)
    usage: Counter[str] = field(default_factory=Counter)
    total_ops: int = 0
    targeted_attempt_count: int = 0
    targeted_success_count: int = 0
    skip_reasons: Counter[str] = field(default_factory=Counter)

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "enabled", False))

    @property
    def dominant_operators(self) -> set[str]:
        return set(getattr(self.config, "dominant_operators", ()) or ())

    @property
    def underused_operators(self) -> set[str]:
        return set(getattr(self.config, "underused_operators", ()) or ())

    def adjusted_weight(self, operator: str, base_weight: float) -> float:
        weight = max(0.0, float(base_weight))
        if not self.enabled:
            return weight
        operator = str(operator or "").strip()
        if not operator:
            return weight
        if self.allowed_operators and operator not in self.allowed_operators:
            return 0.0

        weight *= max(0.0, float(self.operator_prior_multipliers.get(operator, 1.0) or 1.0))
        if operator in self.dominant_operators:
            usage_ratio = self.usage.get(operator, 0) / max(1, self.total_ops)
            decay_rate = max(0.0, float(getattr(self.config, "dominant_decay_rate", 2.0) or 0.0))
            floor = max(0.0, min(1.0, float(getattr(self.config, "dominant_min_multiplier", 0.30) or 0.0)))
            return weight * max(floor, 1.0 - usage_ratio * decay_rate)
        if operator in self.underused_operators:
            usage_count = self.usage.get(operator, 0)
            boost = max(0.0, float(getattr(self.config, "underused_boost", 3.0) or 0.0))
            decay = max(0.0, float(getattr(self.config, "underused_decay", 0.5) or 0.0))
            return weight * boost / (1.0 + usage_count * decay)
        return weight

    def record_operators(self, operators: Iterable[str]) -> None:
        if not self.enabled:
            return
        normalized = [str(operator).strip() for operator in operators if str(operator or "").strip()]
        if not normalized:
            return
        self.usage.update(normalized)
        self.total_ops += len(normalized)

    def record_targeted_attempt(self) -> None:
        if self.enabled:
            self.targeted_attempt_count += 1

    def record_targeted_success(self) -> None:
        if self.enabled:
            self.targeted_success_count += 1

    def record_skip(self, reason: str) -> None:
        if self.enabled:
            self.skip_reasons[str(reason or "unknown")] += 1

    def pick_target_operator(self, rng: Random, candidates: Iterable[str]) -> str | None:
        labels = [
            str(operator).strip()
            for operator in candidates
            if str(operator or "").strip()
            and str(operator).strip() in self.underused_operators
            and (not self.allowed_operators or str(operator).strip() in self.allowed_operators)
        ]
        if not self.enabled or not labels:
            return None
        weights = [max(1e-9, self.adjusted_weight(operator, 1.0)) for operator in labels]
        if sum(weights) <= 0.0:
            return None
        return rng.choices(labels, weights=weights, k=1)[0]

    def lowest_usage_allowed(self, candidates: Iterable[str]) -> str | None:
        labels = [
            str(operator).strip()
            for operator in candidates
            if str(operator or "").strip()
            and (not self.allowed_operators or str(operator).strip() in self.allowed_operators)
        ]
        if not labels:
            return None
        return min(labels, key=lambda operator: (self.usage.get(operator, 0), operator))

    def to_metrics(self) -> dict[str, object]:
        if not self.enabled:
            return {}
        dominant = self.dominant_operators
        underused = self.underused_operators
        dominant_count = sum(count for operator, count in self.usage.items() if operator in dominant)
        underused_count = sum(count for operator, count in self.usage.items() if operator in underused)
        total = max(1, self.total_ops)
        return {
            "enabled": True,
            "dominant_pct": round(100.0 * dominant_count / total, 3),
            "underused_pct": round(100.0 * underused_count / total, 3),
            "corr_count": int(self.usage.get("ts_corr", 0) + self.usage.get("ts_covariance", 0)),
            "days_from_last_change_count": int(self.usage.get("days_from_last_change", 0)),
            "targeted_attempt_count": int(self.targeted_attempt_count),
            "targeted_success_count": int(self.targeted_success_count),
            "skip_reasons": dict(self.skip_reasons),
            "operator_usage": dict(self.usage),
        }
