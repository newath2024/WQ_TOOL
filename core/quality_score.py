from __future__ import annotations

from typing import Any


NEUTRAL_OPERATIONAL_REJECTIONS = frozenset({"poll_timeout_after_downtime"})


class MultiObjectiveQualityScorer:
    @classmethod
    def score(
        cls,
        *,
        metrics: dict[str, Any],
        submission_eligible: bool | None = None,
        rejection_reason: str | None = None,
        status: str = "",
    ) -> float:
        fitness = _clip(_to_float(metrics.get("fitness")), -1.0, 2.0) / 2.0
        sharpe = _clip(_to_float(metrics.get("sharpe")), -1.0, 2.0) / 2.0
        returns = _clip(_to_float(metrics.get("returns")) / 0.10, -1.0, 1.0)
        margin = _clip(_to_float(metrics.get("margin")) / 0.10, -1.0, 1.0)
        turnover = _clip(_to_float(metrics.get("turnover")) / 1.50, 0.0, 1.0)
        drawdown = _clip(_to_float(metrics.get("drawdown")), 0.0, 1.0)
        eligible = _to_optional_bool(submission_eligible)
        eligible_bonus = 1.0 if eligible is True else -0.5 if eligible is False else 0.0
        rejection_penalty = cls.rejection_penalty(rejection_reason=rejection_reason, status=status)
        return float(
            0.45 * fitness
            + 0.30 * sharpe
            + 0.10 * returns
            + 0.05 * margin
            + 0.05 * eligible_bonus
            - 0.10 * turnover
            - 0.05 * drawdown
            - rejection_penalty
        )

    @classmethod
    def score_result(cls, result: Any) -> float:
        return cls.score(
            metrics=dict(getattr(result, "metrics", {}) or {}),
            submission_eligible=getattr(result, "submission_eligible", None),
            rejection_reason=getattr(result, "rejection_reason", None),
            status=str(getattr(result, "status", "") or ""),
        )

    @classmethod
    def score_record(cls, record: Any) -> float:
        return cls.score(
            metrics={
                "fitness": getattr(record, "fitness", None),
                "sharpe": getattr(record, "sharpe", None),
                "returns": getattr(record, "returns", None),
                "margin": getattr(record, "margin", None),
                "turnover": getattr(record, "turnover", None),
                "drawdown": getattr(record, "drawdown", None),
            },
            submission_eligible=getattr(record, "submission_eligible", None),
            rejection_reason=getattr(record, "rejection_reason", None),
            status=str(getattr(record, "status", "") or ""),
        )

    @classmethod
    def rejection_penalty(cls, *, rejection_reason: str | None, status: str = "") -> float:
        reason = str(rejection_reason or "").strip()
        if reason in NEUTRAL_OPERATIONAL_REJECTIONS:
            return 0.0
        if reason:
            return 0.25
        normalized_status = str(status or "").strip().lower()
        return 0.25 if normalized_status in {"failed", "rejected"} else 0.0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None
