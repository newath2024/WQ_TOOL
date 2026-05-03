from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from config.models.quality import QualityScoreConfig
from core.brain_checks import (
    OUTCOME_CHECK_NAMES,
    ROBUSTNESS_CHECK_NAMES,
    STRUCTURAL_RISK_BLOCKER_CHECK_NAMES,
    BrainCheckSummary,
    check_summary_from_record,
    is_check_based_rejection,
    summarize_brain_checks,
)


NEUTRAL_OPERATIONAL_REJECTIONS = frozenset({"poll_timeout_after_downtime"})
REVERSION_BLOCKER_CHECK = "REVERSION_COMPONENT"
DEFAULT_QUALITY_SCORE_CONFIG = QualityScoreConfig()


class MultiObjectiveQualityScorer:
    @classmethod
    def score(
        cls,
        *,
        metrics: dict[str, Any],
        submission_eligible: bool | None = None,
        rejection_reason: str | None = None,
        status: str = "",
        check_summary: BrainCheckSummary | dict[str, Any] | str | None = None,
        is_rejected: bool | None = None,
        quality_config: QualityScoreConfig | Mapping[str, Any] | None = None,
    ) -> float:
        config = _coerce_quality_score_config(quality_config)
        fitness = _clip(_to_float(metrics.get("fitness")), -1.0, 2.0) / 2.0
        sharpe = _clip(_to_float(metrics.get("sharpe")), -1.0, 2.0) / 2.0
        returns = _clip(_to_float(metrics.get("returns")) / 0.10, -1.0, 1.0)
        margin = _clip(_to_float(metrics.get("margin")) / 0.10, -1.0, 1.0)
        turnover = _clip(_to_float(metrics.get("turnover")) / 1.50, 0.0, 1.0)
        drawdown = _clip(_to_float(metrics.get("drawdown")), 0.0, 1.0)
        eligible = _to_optional_bool(submission_eligible)
        eligible_bonus = 1.0 if eligible is True else -0.5 if eligible is False else 0.0
        summary = _coerce_check_summary(
            check_summary,
            status=status,
            rejection_reason=rejection_reason,
        )
        base_check_penalty = cls.compute_check_penalty(summary)
        rejected = _is_rejected(
            status=status,
            rejection_reason=rejection_reason,
            is_rejected=is_rejected,
        )
        if rejected:
            if is_check_based_rejection(
                {
                    "is_rejected": True,
                    "status": status,
                    "rejection_reason": rejection_reason,
                    "check_summary": summary,
                }
            ):
                # Check-based rejects already encode severity in BrainChecks; adding
                # the generic rejection penalty would punish the same failure twice.
                check_penalty = config.check_penalty_weight * base_check_penalty
                rejection_penalty = 0.0
            else:
                # Non-check rejects (duplicate/syntax/API/profile/etc.) are scored by the
                # generic rejection penalty only, so check artifacts do not double-count.
                check_penalty = 0.0
                rejection_penalty = config.rejection_penalty_weight * cls.rejection_penalty(
                    rejection_reason=rejection_reason,
                    status=status,
                    quality_config=config,
                )
        else:
            # Completed alphas can still carry BRAIN check warnings. Apply the
            # warning weight only; no generic rejection penalty is involved.
            check_penalty = config.check_warning_weight * base_check_penalty
            rejection_penalty = 0.0
        return float(
            0.45 * fitness
            + 0.30 * sharpe
            + 0.10 * returns
            + 0.05 * margin
            + 0.05 * eligible_bonus
            - 0.10 * turnover
            - 0.05 * drawdown
            - rejection_penalty
            - check_penalty
        )

    @classmethod
    def score_result(
        cls,
        result: Any,
        *,
        quality_config: QualityScoreConfig | Mapping[str, Any] | None = None,
    ) -> float:
        return cls.score(
            metrics=dict(getattr(result, "metrics", {}) or {}),
            submission_eligible=getattr(result, "submission_eligible", None),
            rejection_reason=getattr(result, "rejection_reason", None),
            status=str(getattr(result, "status", "") or ""),
            check_summary=getattr(result, "check_summary", None),
            is_rejected=getattr(result, "is_rejected", None),
            quality_config=quality_config,
        )

    @classmethod
    def score_record(
        cls,
        record: Any,
        *,
        quality_config: QualityScoreConfig | Mapping[str, Any] | None = None,
    ) -> float:
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
            check_summary=check_summary_from_record(record),
            is_rejected=getattr(record, "is_rejected", None),
            quality_config=quality_config,
        )

    @classmethod
    def rejection_penalty(
        cls,
        *,
        rejection_reason: str | None,
        status: str = "",
        quality_config: QualityScoreConfig | Mapping[str, Any] | None = None,
    ) -> float:
        config = _coerce_quality_score_config(quality_config)
        reason = str(rejection_reason or "").strip()
        if reason in NEUTRAL_OPERATIONAL_REJECTIONS:
            return 0.0
        if reason:
            return config.base_rejection_penalty
        normalized_status = str(status or "").strip().lower()
        return config.base_rejection_penalty if normalized_status in {"failed", "rejected"} else 0.0

    @classmethod
    def check_penalty(
        cls,
        *,
        check_summary: BrainCheckSummary | dict[str, Any] | str | None,
        status: str = "",
        rejection_reason: str | None = None,
    ) -> float:
        summary = _coerce_check_summary(
            check_summary,
            status=status,
            rejection_reason=rejection_reason,
        )
        return cls.compute_check_penalty(summary)

    @classmethod
    def compute_check_penalty(cls, summary: BrainCheckSummary | None) -> float:
        if summary is None:
            return 0.0
        penalty = 0.0
        hard_checks = set(summary.hard_fail_checks)
        warning_checks = set(summary.warning_checks)
        blocking_warnings = set(summary.blocking_warning_checks)
        for name in hard_checks:
            if name in OUTCOME_CHECK_NAMES:
                continue
            if name == REVERSION_BLOCKER_CHECK:
                penalty += 0.35
            elif name in ROBUSTNESS_CHECK_NAMES:
                penalty += 0.05
            elif name in STRUCTURAL_RISK_BLOCKER_CHECK_NAMES:
                penalty += 0.10
            else:
                penalty += 0.05
        if REVERSION_BLOCKER_CHECK not in hard_checks and REVERSION_BLOCKER_CHECK in warning_checks:
            penalty += 0.35
        for name in blocking_warnings - hard_checks - {REVERSION_BLOCKER_CHECK}:
            if name in OUTCOME_CHECK_NAMES:
                continue
            if name in ROBUSTNESS_CHECK_NAMES:
                penalty += 0.05
            elif name in STRUCTURAL_RISK_BLOCKER_CHECK_NAMES:
                penalty += 0.10
            else:
                penalty += 0.05
        nonblocking_count = len(summary.nonblocking_warnings)
        if nonblocking_count > 0:
            penalty += min(0.05, 0.01 * float(nonblocking_count))
        return min(0.45, penalty)


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


def _is_rejected(
    *,
    status: str,
    rejection_reason: str | None,
    is_rejected: bool | None,
) -> bool:
    explicit = _to_optional_bool(is_rejected)
    if explicit is not None:
        return explicit
    if str(rejection_reason or "").strip():
        return True
    return str(status or "").strip().lower() in {"failed", "rejected"}


def _coerce_quality_score_config(
    value: QualityScoreConfig | Mapping[str, Any] | None,
) -> QualityScoreConfig:
    if value is None:
        return DEFAULT_QUALITY_SCORE_CONFIG
    if isinstance(value, QualityScoreConfig):
        return value
    if isinstance(value, Mapping):
        return QualityScoreConfig(**dict(value))
    return DEFAULT_QUALITY_SCORE_CONFIG


def _coerce_check_summary(
    value: BrainCheckSummary | dict[str, Any] | str | None,
    *,
    status: str,
    rejection_reason: str | None,
) -> BrainCheckSummary | None:
    if isinstance(value, BrainCheckSummary):
        return value
    if not value:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(value, dict):
        return None
    if "alpha" in value or "raw_result" in value:
        return summarize_brain_checks(value, status=status, rejection_reason=rejection_reason)
    return BrainCheckSummary(
        hard_fail_checks=_tuple_text(value.get("hard_fail_checks")),
        warning_checks=_tuple_text(value.get("warning_checks")),
        blocking_warning_checks=_tuple_text(value.get("blocking_warning_checks")),
        nonblocking_warnings=_tuple_text(value.get("nonblocking_warnings")),
    )


def _tuple_text(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item).strip().upper() for item in value if str(item).strip()))
