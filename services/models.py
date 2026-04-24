from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.run_context import RunContext
from data.field_registry import FieldRegistry
from data.schema import MarketDataBundle
from evaluation.filtering import EvaluatedAlpha
from features.transforms import ResearchMatrices
from generator.engine import AlphaCandidate
from memory.case_memory import ObjectiveVector
from memory.pattern_memory import BlendDiagnostics, PatternMemoryService, RegionLearningContext, StructuralSignature
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
    region: str
    regime_key: str
    global_regime_key: str
    region_learning_context: RegionLearningContext
    memory_service: PatternMemoryService
    field_registry: FieldRegistry
    legacy_regime_key: str = ""
    market_regime_key: str = ""
    effective_regime_key: str = ""
    regime_label: str = "unknown"
    regime_confidence: float = 0.0
    regime_features: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RegimeSnapshot:
    region: str
    legacy_regime_key: str
    global_regime_key: str
    market_regime_key: str = ""
    effective_regime_key: str = ""
    regime_label: str = "unknown"
    confidence: float = 0.0
    features: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class DedupDecision:
    alpha_id: str
    normalized_expression: str
    stage: str
    decision: str
    reason_code: str
    matched_run_id: str = ""
    matched_alpha_id: str = ""
    matched_scope: str = ""
    similarity_score: float = 0.0
    normalized_match: bool = False
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class DedupBatchResult:
    kept_candidates: tuple[AlphaCandidate, ...]
    blocked_candidates: tuple[AlphaCandidate, ...]
    decisions: tuple[DedupDecision, ...]
    stage_metrics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CrowdingScore:
    alpha_id: str
    stage: str
    total_penalty: float
    family_penalty: float = 0.0
    motif_penalty: float = 0.0
    operator_path_penalty: float = 0.0
    lineage_penalty: float = 0.0
    batch_penalty: float = 0.0
    historical_penalty: float = 0.0
    hard_blocked: bool = False
    reason_codes: tuple[str, ...] = ()
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SelectionBreakdown:
    score_stage: str
    composite_score: float
    components: dict[str, float]
    reason_codes: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class SelectionDecision:
    alpha_id: str
    score_stage: str
    composite_score: float
    selected: bool
    rank: int | None
    reason_codes: tuple[str, ...] = ()
    breakdown: SelectionBreakdown | None = None
    quality_score: float = 0.0


@dataclass(slots=True, frozen=True)
class PreSimulationSelectionResult:
    selected: tuple["CandidateScore", ...]
    archived: tuple["CandidateScore", ...]
    dedup_result: DedupBatchResult
    crowding_scores: dict[str, CrowdingScore] = field(default_factory=dict)
    selection_decisions: tuple[SelectionDecision, ...] = ()
    stage_metrics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BatchPreparationResult:
    candidates: tuple[AlphaCandidate, ...]
    selected: tuple["CandidateScore", ...]
    regime_key: str
    validated_count: int = 0
    archived_count: int = 0
    mutated_children_count: int = 0
    generation_stage_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationServiceResult:
    """Result of candidate generation or mutation persistence."""

    generated_count: int
    inserted_count: int
    exit_code: int = 0
    region: str = ""
    regime_key: str | None = None
    global_regime_key: str | None = None
    pattern_count: int = 0
    export_paths: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationServiceResult:
    """Result bundle for one evaluation round."""

    evaluations: list[EvaluatedAlpha]
    metric_records: list[MetricRecord]
    selection_records: list[SelectionRecord]
    region: str = ""
    regime_key: str = ""
    global_regime_key: str = ""
    evaluation_timestamp: str = ""
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
    region: str
    dataset_fingerprint: str
    regime_key: str
    global_regime_key: str
    selected_timeframe: str
    cache_hits: int
    validation_rows: int
    pattern_blend: BlendDiagnostics | None = None
    case_blend: BlendDiagnostics | None = None
    top_alphas: list[TopAlphaRow] = field(default_factory=list)
    submission_summary: dict[str, dict[str, int]] = field(default_factory=dict)
    top_gene: dict | None = None
    fail_tags: list[dict] = field(default_factory=list)
    rejection_reasons: list[dict] = field(default_factory=list)
    generation_mix: list[dict] = field(default_factory=list)
    hard_filter_summary: dict[str, float | int] = field(default_factory=dict)
    stage_metrics: list[dict] = field(default_factory=list)
    duplicate_summary: list[dict] = field(default_factory=list)
    avg_crowding_penalty: float = 0.0
    latest_regime_snapshot: dict | None = None


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
    objective_vector: ObjectiveVector
    local_heuristic_score: float
    novelty_score: float
    family_score: float
    structural_signature: StructuralSignature
    diversity_score: float = 0.0
    duplicate_risk: float = 0.0
    crowding_penalty: float = 0.0
    regime_fit: float = 0.0
    composite_score: float | None = None
    archive_reason: str | None = None
    reason_codes: tuple[str, ...] = ()
    ranking_rationale: dict[str, object] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        if self.composite_score is not None:
            return float(self.composite_score)
        return (
            self.local_heuristic_score
            + 0.15 * self.novelty_score
            + 0.20 * self.family_score
            + 0.20 * self.diversity_score
        )


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
    progress_log_path: str | None = None


@dataclass(slots=True, frozen=True)
class ServiceCounters:
    generated: int = 0
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    quarantined: int = 0


@dataclass(slots=True, frozen=True)
class ServiceTickOutcome:
    status: str
    pending_job_count: int = 0
    new_result_count: int = 0
    active_batch_id: str | None = None
    queue_depth: int = 0
    queue_counts: dict[str, int] = field(default_factory=dict)
    next_sleep_seconds: int = 0
    generated_count: int = 0
    submitted_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    quarantined_count: int = 0
    last_error: str | None = None
    persona_url: str | None = None
    cooldown_until: str | None = None
    poll_pending_ms: float = 0.0
    prepare_batch_ms: float = 0.0
    submit_batch_ms: float = 0.0
    pre_prepare_pending_job_count: int | None = None


@dataclass(slots=True, frozen=True)
class ServiceRunSummary:
    run_id: str
    service_name: str
    status: str
    ticks_executed: int
    pending_job_count: int
    progress_log_path: str | None = None
