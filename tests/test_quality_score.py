from __future__ import annotations

import pytest

from config.models.quality import QualityScoreConfig
from core.quality_score import MultiObjectiveQualityScorer


TEST_QUALITY_CONFIG = QualityScoreConfig(
    check_penalty_weight=1.0,
    check_warning_weight=0.5,
    rejection_penalty_weight=1.0,
    base_rejection_penalty=0.25,
)


def test_multi_objective_quality_score_rewards_quality_and_penalizes_risk() -> None:
    strong = MultiObjectiveQualityScorer.score(
        metrics={"fitness": 0.50, "sharpe": 0.60, "returns": 0.08, "margin": 0.06, "turnover": 0.20, "drawdown": 0.05},
        submission_eligible=True,
        rejection_reason=None,
        status="completed",
    )
    weak = MultiObjectiveQualityScorer.score(
        metrics={"fitness": 0.02, "sharpe": 0.03, "returns": 0.00, "margin": 0.00, "turnover": 1.40, "drawdown": 0.70},
        submission_eligible=False,
        rejection_reason="invalid data field: foo",
        status="rejected",
    )

    assert strong > weak


def test_multi_objective_quality_score_does_not_penalize_after_downtime_timeout_as_alpha_signal() -> None:
    operational_timeout = MultiObjectiveQualityScorer.score(
        metrics={},
        submission_eligible=None,
        rejection_reason="poll_timeout_after_downtime",
        status="timeout",
    )
    rejected = MultiObjectiveQualityScorer.score(
        metrics={},
        submission_eligible=None,
        rejection_reason="invalid data field: foo",
        status="rejected",
    )

    assert operational_timeout == 0.0
    assert rejected < operational_timeout


def test_multi_objective_quality_score_does_not_sink_outcome_check_near_miss() -> None:
    metrics = {"fitness": 0.80, "sharpe": 1.00, "returns": 0.03, "margin": 0.03, "turnover": 0.10, "drawdown": 0.10}
    clean = MultiObjectiveQualityScorer.score(
        metrics=metrics,
        submission_eligible=None,
        rejection_reason=None,
        status="completed",
        check_summary={"hard_fail_checks": [], "warning_checks": [], "blocking_warning_checks": []},
        quality_config=TEST_QUALITY_CONFIG,
    )
    low_sharpe_fail = MultiObjectiveQualityScorer.score(
        metrics=metrics,
        submission_eligible=None,
        rejection_reason=None,
        status="completed",
        check_summary={
            "hard_fail_checks": ["LOW_SHARPE", "LOW_FITNESS"],
            "warning_checks": [],
            "blocking_warning_checks": [],
        },
    )
    weak_metrics = MultiObjectiveQualityScorer.score(
        metrics={"fitness": 0.10, "sharpe": 0.10, "returns": 0.00, "margin": 0.00, "turnover": 0.10, "drawdown": 0.10},
        submission_eligible=None,
        rejection_reason=None,
        status="completed",
        check_summary={"hard_fail_checks": [], "warning_checks": [], "blocking_warning_checks": []},
    )

    assert low_sharpe_fail == clean
    assert low_sharpe_fail > weak_metrics


def test_multi_objective_quality_score_still_penalizes_reversion_component() -> None:
    metrics = {"fitness": 0.80, "sharpe": 1.00, "returns": 0.03, "margin": 0.03, "turnover": 0.10, "drawdown": 0.10}
    clean = MultiObjectiveQualityScorer.score(
        metrics=metrics,
        submission_eligible=None,
        rejection_reason=None,
        status="completed",
        check_summary={"hard_fail_checks": [], "warning_checks": [], "blocking_warning_checks": []},
    )
    reversion = MultiObjectiveQualityScorer.score(
        metrics=metrics,
        submission_eligible=None,
        rejection_reason=None,
        status="completed",
        check_summary={
            "hard_fail_checks": [],
            "warning_checks": ["REVERSION_COMPONENT"],
            "blocking_warning_checks": ["REVERSION_COMPONENT"],
        },
        quality_config=TEST_QUALITY_CONFIG,
    )

    assert reversion == pytest.approx(clean - 0.5 * 0.35)


def test_check_based_rejection_no_double_penalty() -> None:
    metrics = {"fitness": 0.80, "sharpe": 0.80, "returns": 0.03, "margin": 0.03, "turnover": 0.10, "drawdown": 0.10}
    check_summary = {
        "hard_fail_checks": ["LOW_2Y_SHARPE"],
        "warning_checks": [],
        "blocking_warning_checks": [],
    }
    clean = MultiObjectiveQualityScorer.score(
        metrics=metrics,
        status="completed",
        check_summary={"hard_fail_checks": [], "warning_checks": [], "blocking_warning_checks": []},
        quality_config=TEST_QUALITY_CONFIG,
    )
    score = MultiObjectiveQualityScorer.score(
        metrics=metrics,
        rejection_reason="2Y Sharpe too low",
        status="rejected",
        check_summary=check_summary,
        quality_config=TEST_QUALITY_CONFIG,
    )
    check_penalty = MultiObjectiveQualityScorer.check_penalty(
        check_summary=check_summary,
        status="rejected",
        rejection_reason="2Y Sharpe too low",
    )

    assert score == pytest.approx(clean - TEST_QUALITY_CONFIG.check_penalty_weight * check_penalty)
    assert score > clean - check_penalty - TEST_QUALITY_CONFIG.base_rejection_penalty


def test_non_check_rejection_no_check_penalty() -> None:
    metrics = {"fitness": 0.80, "sharpe": 0.80, "returns": 0.03, "margin": 0.03, "turnover": 0.10, "drawdown": 0.10}
    check_summary = {
        "hard_fail_checks": ["LOW_2Y_SHARPE", "CONCENTRATED_WEIGHT"],
        "warning_checks": ["REVERSION_COMPONENT"],
        "blocking_warning_checks": ["REVERSION_COMPONENT"],
    }
    clean = MultiObjectiveQualityScorer.score(
        metrics=metrics,
        status="completed",
        check_summary={"hard_fail_checks": [], "warning_checks": [], "blocking_warning_checks": []},
        quality_config=TEST_QUALITY_CONFIG,
    )
    score = MultiObjectiveQualityScorer.score(
        metrics=metrics,
        rejection_reason="duplicate expression",
        status="rejected",
        check_summary=check_summary,
        quality_config=TEST_QUALITY_CONFIG,
    )

    assert score == pytest.approx(
        clean - TEST_QUALITY_CONFIG.rejection_penalty_weight * TEST_QUALITY_CONFIG.base_rejection_penalty
    )


def test_near_miss_scores_higher_than_full_fail() -> None:
    metrics = {"fitness": 0.80, "sharpe": 0.80, "returns": 0.03, "margin": 0.03, "turnover": 0.10, "drawdown": 0.10}
    near_miss = MultiObjectiveQualityScorer.score(
        metrics=metrics,
        rejection_reason="2Y Sharpe too low",
        status="rejected",
        check_summary={
            "hard_fail_checks": ["LOW_2Y_SHARPE"],
            "warning_checks": [],
            "blocking_warning_checks": [],
        },
        quality_config=TEST_QUALITY_CONFIG,
    )
    full_fail = MultiObjectiveQualityScorer.score(
        metrics=metrics,
        rejection_reason="Alpha expression includes a reversion component.",
        status="rejected",
        check_summary={
            "hard_fail_checks": [
                "LOW_2Y_SHARPE",
                "LOW_SUB_UNIVERSE_SHARPE",
                "IS_LADDER_SHARPE",
                "CONCENTRATED_WEIGHT",
                "HIGH_TURNOVER",
            ],
            "warning_checks": ["REVERSION_COMPONENT"],
            "blocking_warning_checks": ["REVERSION_COMPONENT"],
        },
        quality_config=TEST_QUALITY_CONFIG,
    )
    random_garbage = MultiObjectiveQualityScorer.score(
        metrics={},
        rejection_reason="duplicate expression",
        status="rejected",
        check_summary={
            "hard_fail_checks": [],
            "warning_checks": [],
            "blocking_warning_checks": [],
        },
        quality_config=TEST_QUALITY_CONFIG,
    )

    assert near_miss > full_fail > random_garbage


def test_completed_with_check_warnings() -> None:
    metrics = {"fitness": 0.80, "sharpe": 0.80, "returns": 0.03, "margin": 0.03, "turnover": 0.10, "drawdown": 0.10}
    check_summary = {
        "hard_fail_checks": [],
        "warning_checks": ["REVERSION_COMPONENT"],
        "blocking_warning_checks": ["REVERSION_COMPONENT"],
    }
    clean = MultiObjectiveQualityScorer.score(
        metrics=metrics,
        status="completed",
        check_summary={"hard_fail_checks": [], "warning_checks": [], "blocking_warning_checks": []},
        quality_config=TEST_QUALITY_CONFIG,
    )
    completed_with_warning = MultiObjectiveQualityScorer.score(
        metrics=metrics,
        status="completed",
        rejection_reason=None,
        check_summary=check_summary,
        quality_config=TEST_QUALITY_CONFIG,
    )
    check_penalty = MultiObjectiveQualityScorer.check_penalty(
        check_summary=check_summary,
        status="completed",
        rejection_reason=None,
    )

    assert completed_with_warning == pytest.approx(
        clean - TEST_QUALITY_CONFIG.check_warning_weight * check_penalty
    )
    assert completed_with_warning > clean - TEST_QUALITY_CONFIG.check_penalty_weight * check_penalty
