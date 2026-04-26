from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class SimulationJob:
    job_id: str
    candidate_id: str
    expression: str
    backend: str
    status: str
    submitted_at: str
    sim_config_snapshot: dict[str, object]
    run_id: str
    batch_id: str
    round_index: int = 0
    export_path: str | None = None
    raw_submission: dict[str, object] = field(default_factory=dict)
    error_message: str | None = None


@dataclass(slots=True, frozen=True)
class SimulationResult:
    expression: str
    job_id: str
    status: str
    region: str
    universe: str
    delay: int
    neutralization: str
    decay: int
    metrics: dict[str, float | None]
    submission_eligible: bool | None
    rejection_reason: str | None
    raw_result: dict[str, object]
    simulated_at: str
    candidate_id: str = ""
    batch_id: str = ""
    run_id: str = ""
    round_index: int = 0
    backend: str = ""
    metric_source: str = "external_brain"


@dataclass(slots=True, frozen=True)
class BrainSimulationBatch:
    batch_id: str
    backend: str
    status: str
    jobs: tuple[SimulationJob, ...] = ()
    results: tuple[SimulationResult, ...] = ()
    export_path: str | None = None

    @property
    def submitted_count(self) -> int:
        return len(self.jobs)

    @property
    def completed_count(self) -> int:
        return sum(1 for result in self.results if result.status == "completed")

    @property
    def pending_count(self) -> int:
        terminal = {"completed", "failed", "rejected", "timeout"}
        return sum(1 for job in self.jobs if job.status not in terminal)
