from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


BLOCKING_CHECK_NAMES = frozenset(
    {
        "LOW_SHARPE",
        "LOW_FITNESS",
        "LOW_2Y_SHARPE",
        "LOW_SUB_UNIVERSE_SHARPE",
        "IS_LADDER_SHARPE",
        "CONCENTRATED_WEIGHT",
        "LOW_TURNOVER",
        "HIGH_TURNOVER",
        "REVERSION_COMPONENT",
    }
)

OUTCOME_CHECK_NAMES = frozenset({"LOW_SHARPE", "LOW_FITNESS"})
ROBUSTNESS_CHECK_NAMES = frozenset(
    {
        "LOW_2Y_SHARPE",
        "LOW_SUB_UNIVERSE_SHARPE",
        "IS_LADDER_SHARPE",
    }
)
STRUCTURAL_RISK_BLOCKER_CHECK_NAMES = frozenset(
    {
        "REVERSION_COMPONENT",
        "CONCENTRATED_WEIGHT",
        "LOW_TURNOVER",
        "HIGH_TURNOVER",
    }
)
SYNTHETIC_REJECTION_CHECK_NAMES = frozenset({"REVERSION_COMPONENT"})
CHECK_BASED_REJECTION_CHECK_NAMES = frozenset(
    {
        *BLOCKING_CHECK_NAMES,
        "LOW_RETURNS",
        "HIGH_SELF_CORR",
        "HIGH_SELF_CORRELATION",
        "LOW_COVERAGE",
        "INSUFFICIENT_COVERAGE",
    }
)
CHECK_BASED_REJECTION_REASON_MARKERS = (
    "low sharpe",
    "sharpe too low",
    "2y sharpe",
    "2 year sharpe",
    "2-year sharpe",
    "low fitness",
    "fitness too low",
    "sub universe",
    "sub-universe",
    "ladder",
    "concentrated weight",
    "low turnover",
    "high turnover",
    "reversion",
    "low returns",
    "returns too low",
    "self corr",
    "self-corr",
    "self correlation",
    "self-correlation",
    "self_corr",
    "coverage too low",
    "insufficient coverage",
)
OPERATIONAL_TIMEOUT_MARKERS = (
    "rate_limit",
    "rate limit",
    "throttle",
    "throttled",
    "iqc",
    "concurrent_limit",
    "concurrent limit",
    "concurrentsimulationlimit",
    "server_error",
    "server error",
    "503",
    "429",
    "persona",
    "auth",
    "poll_timeout_after_downtime",
    "after_downtime",
    "downtime",
)
OPERATIONAL_RUNTIME_STATUSES = frozenset(
    {
        "auth_throttled",
        "auth_unavailable",
        "cooldown",
        "waiting_persona",
        "waiting_persona_confirmation",
    }
)
TimeoutCause = Literal["quality", "operational", "unknown"]


@dataclass(frozen=True, slots=True)
class BrainCheckSummary:
    checks: tuple[dict[str, Any], ...] = ()
    hard_fail_checks: tuple[str, ...] = ()
    warning_checks: tuple[str, ...] = ()
    blocking_warning_checks: tuple[str, ...] = ()
    nonblocking_warnings: tuple[str, ...] = ()
    pass_checks: tuple[str, ...] = ()
    long_count: int | None = None
    short_count: int | None = None
    derived_submit_ready: bool | None = None
    blocking_message: str | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [dict(item) for item in self.checks],
            "hard_fail_checks": list(self.hard_fail_checks),
            "warning_checks": list(self.warning_checks),
            "blocking_warning_checks": list(self.blocking_warning_checks),
            "nonblocking_warnings": list(self.nonblocking_warnings),
            "pass_checks": list(self.pass_checks),
            "long_count": self.long_count,
            "short_count": self.short_count,
            "derived_submit_ready": self.derived_submit_ready,
            "blocking_message": self.blocking_message,
            "metrics": dict(self.metrics),
        }


def summarize_brain_checks(
    raw_result: Mapping[str, Any] | str | None,
    *,
    status: str = "",
    rejection_reason: str | None = None,
) -> BrainCheckSummary:
    payload = _decode_mapping(raw_result)
    is_payload = _alpha_is_payload(payload)
    checks = tuple(_normalize_check(item) for item in _checks_payload(is_payload))
    hard_fail_checks = _names_for_results(checks, {"FAIL", "FAILED", "ERROR"})
    warning_checks = _names_for_results(checks, {"WARNING"})
    blocking_warning_checks = tuple(name for name in warning_checks if name in BLOCKING_CHECK_NAMES)
    nonblocking_warnings = tuple(name for name in warning_checks if name not in BLOCKING_CHECK_NAMES)
    pass_checks = _names_for_results(checks, {"PASS", "PASSED", "OK"})
    blocking_message = _first_blocking_message(checks, hard_fail_checks, blocking_warning_checks)
    normalized_status = str(status or "").strip().lower()
    normalized_rejection = str(rejection_reason or "").strip()
    derived_submit_ready = None
    if normalized_status:
        derived_submit_ready = (
            normalized_status == "completed"
            and not normalized_rejection
            and not hard_fail_checks
            and not blocking_warning_checks
        )
    return BrainCheckSummary(
        checks=checks,
        hard_fail_checks=hard_fail_checks,
        warning_checks=warning_checks,
        blocking_warning_checks=blocking_warning_checks,
        nonblocking_warnings=nonblocking_warnings,
        pass_checks=pass_checks,
        long_count=_optional_int(is_payload.get("longCount") or is_payload.get("long_count")),
        short_count=_optional_int(is_payload.get("shortCount") or is_payload.get("short_count")),
        derived_submit_ready=derived_submit_ready,
        blocking_message=blocking_message,
        metrics={
            "sharpe": _optional_float(is_payload.get("sharpe")),
            "fitness": _optional_float(is_payload.get("fitness")),
            "turnover": _optional_float(is_payload.get("turnover")),
            "drawdown": _optional_float(is_payload.get("drawdown")),
            "returns": _optional_float(is_payload.get("returns")),
            "margin": _optional_float(is_payload.get("margin")),
        },
    )


def check_summary_from_record(record: Any) -> BrainCheckSummary:
    raw_summary = getattr(record, "check_summary_json", None)
    if raw_summary:
        payload = _decode_mapping(raw_summary)
        return BrainCheckSummary(
            checks=tuple(_normalize_check(item) for item in payload.get("checks", []) if isinstance(item, Mapping)),
            hard_fail_checks=_tuple_text(payload.get("hard_fail_checks")),
            warning_checks=_tuple_text(payload.get("warning_checks")),
            blocking_warning_checks=_tuple_text(payload.get("blocking_warning_checks")),
            nonblocking_warnings=_tuple_text(payload.get("nonblocking_warnings")),
            pass_checks=_tuple_text(payload.get("pass_checks")),
            long_count=_optional_int(payload.get("long_count")),
            short_count=_optional_int(payload.get("short_count")),
            derived_submit_ready=_optional_bool(payload.get("derived_submit_ready")),
            blocking_message=str(payload.get("blocking_message") or "") or None,
            metrics=_decode_metrics(payload.get("metrics")),
        )
    return summarize_brain_checks(
        getattr(record, "raw_result_json", None),
        status=str(getattr(record, "status", "") or ""),
        rejection_reason=getattr(record, "rejection_reason", None),
    )


def checks_json(summary: BrainCheckSummary) -> str:
    return json.dumps(summary.to_dict(), sort_keys=True)


def names_json(names: tuple[str, ...] | list[str]) -> str:
    return json.dumps(list(names), sort_keys=True)


def parse_names_json(raw: Any) -> tuple[str, ...]:
    return _tuple_text(_decode_json(raw, default=[]))


def derive_submit_ready(
    *,
    status: str,
    rejection_reason: str | None,
    hard_fail_checks: tuple[str, ...] | list[str],
    blocking_warning_checks: tuple[str, ...] | list[str],
) -> bool:
    return (
        str(status or "").strip().lower() == "completed"
        and not str(rejection_reason or "").strip()
        and not tuple(hard_fail_checks)
        and not tuple(blocking_warning_checks)
    )


def classify_timeout_cause(result: Any) -> TimeoutCause:
    """Classify timeout evidence without requiring a concrete record type."""
    if result is None:
        return "unknown"
    status = str(_record_value(result, "status") or "").strip().lower()
    if status and status != "timeout":
        return "unknown"

    reason_text = " ".join(
        text
        for text in (
            _record_text(result, "rejection_reason"),
            _record_text(result, "error_message"),
            _record_text(result, "service_failure_reason"),
            _record_text(result, "submission_error_message"),
            _record_text(result, "submission_service_failure_reason"),
            _record_text(result, "batch_service_status_reason"),
            _record_text(result, "runtime_status"),
            _record_text(result, "runtime_last_error"),
            _record_text(result, "last_error"),
        )
        if text
    ).lower()
    compact_reason = reason_text.replace("-", "_")
    if any(marker in reason_text or marker in compact_reason for marker in OPERATIONAL_TIMEOUT_MARKERS):
        return "operational"

    runtime_status = _record_text(result, "runtime_status").strip().lower()
    if runtime_status in OPERATIONAL_RUNTIME_STATUSES:
        return "operational"
    if (
        _truthy_record_value(result, "auth_cooldown_active")
        or _truthy_record_value(result, "persona_confirmation_pending")
        or _truthy_record_value(result, "runtime_persona_confirmation_pending")
    ):
        return "operational"
    if _record_int(result, "batch_timeout_peer_count") >= 3:
        return "operational"
    if _record_int(result, "service_tick_timeout_peer_count") >= 3:
        return "operational"

    if (
        _record_int(result, "batch_timeout_peer_count") == 0
        and _record_int(result, "batch_completed_count") > 0
        and _record_int(result, "batch_failed_count") == 0
        and _record_int(result, "batch_rejected_count") == 0
    ):
        return "quality"
    if max(
        _record_int(result, "candidate_timeout_batch_count"),
        _record_int(result, "same_alpha_timeout_batch_count"),
        _record_int(result, "alpha_timeout_batch_count"),
    ) >= 2:
        return "quality"
    return "unknown"


def is_check_based_rejection(result: Any) -> bool:
    """Return True when rejection evidence points to a BRAIN check condition."""
    if result is None:
        return False

    reason = _record_text(result, "rejection_reason")
    if not reason:
        return False
    explicit_rejected = _optional_bool(_record_value(result, "is_rejected"))
    if explicit_rejected is False:
        return False

    summary = _summary_from_result(result, status=_record_text(result, "status"), rejection_reason=reason)
    if summary is not None and _reason_matches_check_summary(reason, summary):
        return True
    return _reason_matches_check_based_marker(reason)


def first_blocking_message(summary: BrainCheckSummary) -> str | None:
    return summary.blocking_message


def outcome_check_names(names: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    return _matching_names(names, OUTCOME_CHECK_NAMES)


def robustness_check_names(names: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    return _matching_names(names, ROBUSTNESS_CHECK_NAMES)


def structural_risk_check_names(names: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    return _matching_names(names, STRUCTURAL_RISK_BLOCKER_CHECK_NAMES)


def has_structural_risk_blocker(
    hard_fail_checks: tuple[str, ...] | list[str] | set[str],
    blocking_warning_checks: tuple[str, ...] | list[str] | set[str],
) -> bool:
    return bool(
        structural_risk_check_names(hard_fail_checks)
        or structural_risk_check_names(blocking_warning_checks)
    )


def first_synthetic_rejection_message(summary: BrainCheckSummary) -> str | None:
    rejection_checks = set(summary.hard_fail_checks) | set(summary.blocking_warning_checks)
    rejection_checks &= SYNTHETIC_REJECTION_CHECK_NAMES
    if not rejection_checks:
        return None
    for check in summary.checks:
        name = str(check.get("name") or "").strip().upper()
        if name not in rejection_checks:
            continue
        message = str(check.get("message") or "").strip()
        if message:
            return message
    return None


def _summary_from_result(
    result: Any,
    *,
    status: str,
    rejection_reason: str | None,
) -> BrainCheckSummary | None:
    raw_summary = _record_value(result, "check_summary")
    if raw_summary in (None, ""):
        raw_summary = _record_value(result, "check_summary_json")
    if isinstance(raw_summary, BrainCheckSummary):
        return raw_summary
    summary_payload = _decode_mapping(raw_summary)
    if summary_payload:
        if _looks_like_summary_payload(summary_payload):
            return BrainCheckSummary(
                checks=tuple(
                    _normalize_check(item)
                    for item in summary_payload.get("checks", [])
                    if isinstance(item, Mapping)
                ),
                hard_fail_checks=_tuple_text(summary_payload.get("hard_fail_checks")),
                warning_checks=_tuple_text(summary_payload.get("warning_checks")),
                blocking_warning_checks=_tuple_text(summary_payload.get("blocking_warning_checks")),
                nonblocking_warnings=_tuple_text(summary_payload.get("nonblocking_warnings")),
                pass_checks=_tuple_text(summary_payload.get("pass_checks")),
                long_count=_optional_int(summary_payload.get("long_count")),
                short_count=_optional_int(summary_payload.get("short_count")),
                derived_submit_ready=_optional_bool(summary_payload.get("derived_submit_ready")),
                blocking_message=str(summary_payload.get("blocking_message") or "") or None,
                metrics=_decode_metrics(summary_payload.get("metrics")),
            )
        return summarize_brain_checks(summary_payload, status=status, rejection_reason=rejection_reason)

    raw_result = _record_value(result, "raw_result")
    if raw_result in (None, ""):
        raw_result = _record_value(result, "raw_result_json")
    if raw_result not in (None, ""):
        return summarize_brain_checks(raw_result, status=status, rejection_reason=rejection_reason)

    hard_fail_checks = _names_from_record_field(result, "hard_fail_checks", "hard_fail_checks_json")
    blocking_warning_checks = _names_from_record_field(
        result,
        "blocking_warning_checks",
        "blocking_warning_checks_json",
    )
    warning_checks = _names_from_record_field(result, "warning_checks", "warning_checks_json")
    if hard_fail_checks or blocking_warning_checks or warning_checks:
        return BrainCheckSummary(
            hard_fail_checks=hard_fail_checks,
            warning_checks=warning_checks,
            blocking_warning_checks=blocking_warning_checks,
        )
    return None


def _looks_like_summary_payload(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload
        for key in (
            "checks",
            "hard_fail_checks",
            "warning_checks",
            "blocking_warning_checks",
            "nonblocking_warnings",
            "pass_checks",
            "blocking_message",
        )
    )


def _names_from_record_field(result: Any, tuple_key: str, json_key: str) -> tuple[str, ...]:
    raw_names = _record_value(result, tuple_key)
    names = _tuple_text(raw_names)
    if names:
        return names
    return _tuple_text(_decode_json(_record_value(result, json_key), default=[]))


def _reason_matches_check_summary(reason: str, summary: BrainCheckSummary) -> bool:
    rejection_checks = set(summary.hard_fail_checks) | set(summary.blocking_warning_checks)
    rejection_checks &= CHECK_BASED_REJECTION_CHECK_NAMES
    if not rejection_checks:
        return False

    if any(_reason_matches_check_name(reason, name) for name in rejection_checks):
        return True
    if _text_contains_equivalent(summary.blocking_message, reason):
        return True
    for check in summary.checks:
        name = str(check.get("name") or "").strip().upper()
        if name not in rejection_checks:
            continue
        if _text_contains_equivalent(check.get("message"), reason):
            return True
    return _reason_matches_check_based_marker(reason)


def _reason_matches_check_name(reason: str, check_name: str) -> bool:
    normalized_reason = _normalize_reason_text(reason)
    normalized_name = _normalize_reason_text(check_name)
    readable_name = normalized_name.replace("_", " ")
    return normalized_name in normalized_reason or readable_name in normalized_reason


def _reason_matches_check_based_marker(reason: str) -> bool:
    normalized = _normalize_reason_text(reason).replace("_", " ")
    return any(
        _normalize_reason_text(marker).replace("_", " ") in normalized
        for marker in CHECK_BASED_REJECTION_REASON_MARKERS
    )


def _text_contains_equivalent(left: Any, right: Any) -> bool:
    left_text = _normalize_reason_text(left)
    right_text = _normalize_reason_text(right)
    if not left_text or not right_text:
        return False
    return left_text == right_text or left_text in right_text or right_text in left_text


def _normalize_reason_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("-", " ").split())


def _alpha_is_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    alpha = payload.get("alpha")
    if not isinstance(alpha, Mapping):
        raw_result = payload.get("raw_result")
        raw_result_payload = _decode_mapping(raw_result)
        alpha = raw_result_payload.get("alpha") if raw_result_payload else None
    is_payload = alpha.get("is") if isinstance(alpha, Mapping) else None
    return dict(is_payload) if isinstance(is_payload, Mapping) else {}


def _checks_payload(is_payload: Mapping[str, Any]) -> list[Any]:
    checks = is_payload.get("checks")
    return list(checks) if isinstance(checks, list) else []


def _normalize_check(raw: Any) -> dict[str, Any]:
    check = dict(raw) if isinstance(raw, Mapping) else {}
    name = str(check.get("name") or check.get("test") or check.get("code") or "").strip().upper()
    result_value = check.get("result") or check.get("status")
    if result_value in (None, "") and isinstance(check.get("passed"), bool):
        result_value = "PASS" if check.get("passed") is True else "FAIL"
    elif result_value in (None, ""):
        result_value = check.get("passed")
    result = str(result_value or "").strip().upper()
    normalized = {
        "name": name,
        "result": result,
    }
    for key in ("value", "limit", "message", "date", "year", "startDate", "endDate"):
        if key in check:
            normalized[key] = check[key]
    return normalized


def _names_for_results(checks: tuple[dict[str, Any], ...], result_values: set[str]) -> tuple[str, ...]:
    names: list[str] = []
    for check in checks:
        name = str(check.get("name") or "").strip().upper()
        result = str(check.get("result") or "").strip().upper()
        if name and (result in result_values or (result == "FAIL" and "FAIL" in result_values)):
            names.append(name)
    return tuple(dict.fromkeys(names))


def _first_blocking_message(
    checks: tuple[dict[str, Any], ...],
    hard_fail_checks: tuple[str, ...],
    blocking_warning_checks: tuple[str, ...],
) -> str | None:
    blocking = set(hard_fail_checks) | set(blocking_warning_checks)
    for check in checks:
        name = str(check.get("name") or "").strip().upper()
        if name not in blocking:
            continue
        message = str(check.get("message") or "").strip()
        if message:
            return message
    return None


def _decode_mapping(raw: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    decoded = _decode_json(raw, default={})
    return decoded if isinstance(decoded, dict) else {}


def _decode_json(raw: Any, *, default: Any) -> Any:
    if raw in (None, ""):
        return default
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _decode_metrics(raw: Any) -> dict[str, float | None]:
    payload = raw if isinstance(raw, Mapping) else {}
    return {str(key): _optional_float(value) for key, value in payload.items()}


def _tuple_text(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item).strip().upper() for item in raw if str(item).strip()))


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
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


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(key, default)
    try:
        return record[key]
    except (KeyError, IndexError, TypeError):
        return getattr(record, key, default)


def _record_text(record: Any, key: str) -> str:
    value = _record_value(record, key)
    return str(value or "").strip()


def _record_int(record: Any, key: str) -> int:
    try:
        return int(_record_value(record, key) or 0)
    except (TypeError, ValueError):
        return 0


def _truthy_record_value(record: Any, key: str) -> bool:
    value = _record_value(record, key)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int):
        return value != 0
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "active", "pending"}


def _matching_names(names: tuple[str, ...] | list[str] | set[str], allowed: frozenset[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            normalized
            for item in names
            if (normalized := str(item).strip().upper()) and normalized in allowed
        )
    )
