from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BrainResultRecord:
    job_id: str
    run_id: str
    round_index: int
    batch_id: str
    candidate_id: str
    expression: str
    status: str
    region: str
    universe: str
    delay: int
    neutralization: str
    decay: int
    sharpe: float | None
    fitness: float | None
    turnover: float | None
    drawdown: float | None
    returns: float | None
    margin: float | None
    submission_eligible: bool | None
    rejection_reason: str | None
    raw_result_json: str
    metric_source: str
    simulated_at: str
    created_at: str
    quality_score: float = 0.0
