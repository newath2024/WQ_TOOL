from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    seed: int
    config_path: str
    config_snapshot: str
    status: str
    started_at: str
    finished_at: str | None
    dataset_summary: str | None
    profile_name: str | None = None
    dataset_fingerprint: str | None = None
    selected_timeframe: str | None = None
    regime_key: str | None = None
    entry_command: str | None = None


@dataclass(frozen=True, slots=True)
class AlphaRecord:
    run_id: str
    alpha_id: str
    expression: str
    normalized_expression: str
    generation_mode: str
    generation_metadata: str
    complexity: int
    created_at: str
    status: str


@dataclass(frozen=True, slots=True)
class MetricRecord:
    run_id: str
    alpha_id: str
    split: str
    sharpe: float
    max_drawdown: float
    win_rate: float
    average_return: float
    turnover: float
    observation_count: int
    cumulative_return: float
    fitness: float
    stability_score: float
    passed_filters: bool
    simulation_signature: str
    simulation_config_snapshot: str
    delay_mode: str
    neutralization: str
    neutralization_profile: str
    submission_pass_count: int
    cache_hit: bool
    created_at: str


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    run_id: str
    alpha_id: str
    rank: int
    selected_at: str
    validation_fitness: float
    reason: str
    ranking_rationale_json: str = ""


@dataclass(frozen=True, slots=True)
class SubmissionTestRecord:
    run_id: str
    alpha_id: str
    test_name: str
    passed: bool
    details_json: str
    created_at: str


@dataclass(frozen=True, slots=True)
class SimulationCacheRecord:
    simulation_signature: str
    normalized_expression: str
    simulation_config_snapshot: str
    delay_mode: str
    neutralization: str
    neutralization_profile: str
    split_metrics_json: str
    submission_tests_json: str
    subuniverse_metrics_json: str
    validation_signal_json: str
    validation_returns_json: str
    created_at: str


@dataclass(frozen=True, slots=True)
class AlphaHistoryRecord:
    run_id: str
    alpha_id: str
    regime_key: str
    expression: str
    normalized_expression: str
    generation_mode: str
    generation_metadata_json: str
    parent_refs_json: str
    structural_signature_json: str
    gene_ids_json: str
    train_metrics_json: str
    validation_metrics_json: str
    test_metrics_json: str
    validation_signal_json: str
    validation_returns_json: str
    outcome_score: float
    behavioral_novelty_score: float
    passed_filters: bool
    selected: bool
    submission_pass_count: int
    diagnosis_summary_json: str
    rejection_reasons_json: str
    created_at: str


@dataclass(frozen=True, slots=True)
class AlphaDiagnosisRecord:
    run_id: str
    alpha_id: str
    tag_type: str
    tag: str
    created_at: str


@dataclass(frozen=True, slots=True)
class AlphaPatternRecord:
    regime_key: str
    pattern_id: str
    pattern_kind: str
    pattern_value: str
    support: int
    success_count: int
    failure_count: int
    avg_outcome: float
    avg_behavioral_novelty: float
    fail_tag_counts_json: str
    pattern_score: float
    last_seen_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class AlphaPatternMembershipRecord:
    run_id: str
    alpha_id: str
    regime_key: str
    pattern_id: str
    pattern_kind: str
    pattern_value: str
    created_at: str
