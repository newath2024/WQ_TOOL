from __future__ import annotations

from core.brain_checks import (
    classify_timeout_cause,
    derive_submit_ready,
    has_structural_risk_blocker,
    outcome_check_names,
    robustness_check_names,
    structural_risk_check_names,
    summarize_brain_checks,
)


def test_summarize_brain_checks_extracts_failures_and_counts() -> None:
    summary = summarize_brain_checks(
        {
            "alpha": {
                "is": {
                    "longCount": 1200,
                    "shortCount": 1300,
                    "checks": [
                        {"name": "LOW_SHARPE", "result": "FAIL", "value": 0.7, "limit": 1.58},
                        {"name": "REVERSION_COMPONENT", "result": "WARNING", "message": "reversion"},
                        {"name": "MATCHES_THEMES", "result": "WARNING"},
                        {"name": "LOW_TURNOVER", "result": "PASS", "value": 0.03},
                    ],
                }
            }
        },
        status="completed",
        rejection_reason=None,
    )

    assert summary.long_count == 1200
    assert summary.short_count == 1300
    assert summary.hard_fail_checks == ("LOW_SHARPE",)
    assert summary.blocking_warning_checks == ("REVERSION_COMPONENT",)
    assert summary.nonblocking_warnings == ("MATCHES_THEMES",)
    assert summary.derived_submit_ready is False
    assert summary.blocking_message == "reversion"


def test_derive_submit_ready_requires_clean_completed_result() -> None:
    assert derive_submit_ready(
        status="completed",
        rejection_reason=None,
        hard_fail_checks=(),
        blocking_warning_checks=(),
    )
    assert not derive_submit_ready(
        status="completed",
        rejection_reason=None,
        hard_fail_checks=(),
        blocking_warning_checks=("REVERSION_COMPONENT",),
    )


def test_summarize_brain_checks_handles_boolean_passed_flag() -> None:
    summary = summarize_brain_checks(
        {
            "alpha": {
                "is": {
                    "checks": [
                        {"name": "LOW_FITNESS", "passed": False},
                        {"name": "LOW_SHARPE", "passed": True},
                    ]
                }
            }
        },
        status="completed",
    )

    assert summary.hard_fail_checks == ("LOW_FITNESS",)
    assert summary.pass_checks == ("LOW_SHARPE",)


def test_brain_check_classification_separates_outcome_robustness_and_structural() -> None:
    names = ("LOW_SHARPE", "LOW_2Y_SHARPE", "CONCENTRATED_WEIGHT")

    assert outcome_check_names(names) == ("LOW_SHARPE",)
    assert robustness_check_names(names) == ("LOW_2Y_SHARPE",)
    assert structural_risk_check_names(names) == ("CONCENTRATED_WEIGHT",)
    assert has_structural_risk_blocker(("LOW_FITNESS",), ("REVERSION_COMPONENT",))


def test_classify_timeout_cause_uses_operational_keywords_and_context() -> None:
    assert (
        classify_timeout_cause({"status": "timeout", "rejection_reason": "BRAIN 429 rate_limit"})
        == "operational"
    )
    assert classify_timeout_cause({"status": "timeout", "batch_timeout_peer_count": 3}) == "operational"
    assert classify_timeout_cause({"status": "timeout", "auth_cooldown_active": True}) == "operational"


def test_classify_timeout_cause_uses_quality_context_and_unknown_default() -> None:
    assert (
        classify_timeout_cause(
            {"status": "timeout", "batch_timeout_peer_count": 0, "batch_completed_count": 2}
        )
        == "quality"
    )
    assert classify_timeout_cause({"status": "timeout", "candidate_timeout_batch_count": 2}) == "quality"
    assert classify_timeout_cause({"status": "timeout"}) == "unknown"
