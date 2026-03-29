from __future__ import annotations

from dataclasses import dataclass, field

from core.run_context import RunContext
from data.schema import MarketDataBundle
from evaluation.filtering import EvaluatedAlpha
from features.transforms import ResearchMatrices
from memory.pattern_memory import PatternMemoryService
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
