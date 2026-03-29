from __future__ import annotations

from dataclasses import dataclass, field

from core.run_context import RunContext
from data.field_registry import FieldRegistry
from data.schema import MarketDataBundle
from evaluation.filtering import EvaluatedAlpha
from features.transforms import ResearchMatrices
from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemoryService, StructuralSignature
from storage.models import MetricRecord, RunRecord, SelectionRecord


@dataclass(slots=True)
class CommandEnvironment:
    """Runtime objects shared across CLI commands."""

    config_path: str
    command_name: str
    context: RunContext


@dataclass(slots=True)
class ResearchContext:
    """Loaded dataset plus derived research matrices for one run."""

    bundle: MarketDataBundle
    matrices: ResearchMatrices
    regime_key: str
    memory_service: PatternMemoryService
    field_registry: FieldRegistry


@dataclass(slots=True)
class GenerationServiceResult:
    """Result of candidate generation or mutation persistence."""

    generated_count: int
    inserted_count: int
    exit_code: int = 0
    regime_key: str | None = None
    pattern_count: int = 0
    export_paths: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationServiceResult:
    """Result bundle for one evaluation round."""

    evaluations: list[EvaluatedAlpha]
    metric_records: list[MetricRecord]
    selection_records: list[SelectionRecord]
    regime_key: str
    evaluation_timestamp: str
    export_paths: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TopAlphaRow:
    rank: int
    alpha_id: str
    validation_fitness: float
    expression: str
    generation_mode: str
    complexity: int
    delay_mode: str
    neutralization: str
    submission_pass_count: int
    cache_hit: bool


@dataclass(slots=True, frozen=True)
class ReportSummary:
    """Formatted report payload for one run."""

    run: RunRecord
    profile_name: str
    dataset_fingerprint: str
    regime_key: str
    selected_timeframe: str
    cache_hits: int
    validation_rows: int
    top_alphas: list[TopAlphaRow] = field(default_factory=list)
    submission_summary: dict[str, dict[str, int]] = field(default_factory=dict)
    top_gene: dict | None = None
    fail_tags: list[dict] = field(default_factory=list)
    rejection_reasons: list[dict] = field(default_factory=list)
    generation_mix: list[dict] = field(default_factory=list)
    hard_filter_summary: dict[str, float | int] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PatternView:
    pattern_kind: str
    pattern_value: str
    pattern_score: float
    support: int
    success_count: int
    failure_count: int


@dataclass(slots=True, frozen=True)
class LineageViewRow:
    depth: int
    run_id: str
    alpha_id: str
    outcome_score: float | None
    fail_tags: tuple[str, ...]
    expression: str


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


@dataclass(slots=True, frozen=True)
class CandidateScore:
    candidate: AlphaCandidate
    local_heuristic_score: float
    novelty_score: float
    family_score: float
    structural_signature: StructuralSignature
    archive_reason: str | None = None

    @property
    def total_score(self) -> float:
        return self.local_heuristic_score + 0.15 * self.novelty_score + 0.25 * self.family_score


@dataclass(slots=True, frozen=True)
class ClosedLoopRoundSummary:
    round_index: int
    status: str
    generated_count: int
    validated_count: int
    submitted_count: int
    completed_count: int
    selected_for_mutation_count: int
    mutated_children_count: int
    batch_id: str | None = None
    export_path: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ClosedLoopRunSummary:
    run_id: str
    backend: str
    status: str
    rounds: tuple[ClosedLoopRoundSummary, ...]
