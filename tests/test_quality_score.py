from __future__ import annotations

from core.quality_score import MultiObjectiveQualityScorer


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
