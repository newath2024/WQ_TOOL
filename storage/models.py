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
    global_regime_key: str | None = None
    market_regime_key: str | None = None
    effective_regime_key: str | None = None
    regime_label: str | None = None
    regime_confidence: float | None = None
    region: str | None = None
    entry_command: str | None = None


@dataclass(frozen=True, slots=True)
class AlphaRecord:
    run_id: str
    alpha_id: str
    expression: str
    normalized_expression: str
    generation_mode: str
    template_name: str
    fields_used_json: str
    operators_used_json: str
    depth: int
    generation_metadata: str
    complexity: int
    created_at: str
    status: str
    structural_signature_json: str = "{}"


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
    region: str
    regime_key: str
    global_regime_key: str
    market_regime_key: str
    effective_regime_key: str
    regime_label: str
    regime_confidence: float
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
    metric_source: str
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
    region: str
    regime_key: str
    global_regime_key: str
    pattern_id: str
    pattern_kind: str
    pattern_value: str
    created_at: str


@dataclass(frozen=True, slots=True)
class AlphaCaseRecord:
    run_id: str
    alpha_id: str
    region: str
    regime_key: str
    global_regime_key: str
    market_regime_key: str
    effective_regime_key: str
    regime_label: str
    regime_confidence: float
    metric_source: str
    family_signature: str
    structural_signature_json: str
    genome_hash: str
    genome_json: str
    motif: str
    field_families_json: str
    operator_path_json: str
    complexity_bucket: str
    turnover_bucket: str
    horizon_bucket: str
    mutation_mode: str
    parent_family_signatures_json: str
    fail_tags_json: str
    success_tags_json: str
    objective_vector_json: str
    outcome_score: float
    created_at: str


@dataclass(frozen=True, slots=True)
class FieldCatalogRecord:
    field_name: str
    dataset: str
    field_type: str
    coverage: float
    alpha_usage_count: int
    category: str
    delay: int
    region: str
    universe: str
    runtime_available: bool
    description: str
    subcategory: str
    user_count: int
    category_weight: float
    field_score: float
    updated_at: str


@dataclass(frozen=True, slots=True)
class RunFieldScoreRecord:
    run_id: str
    field_name: str
    runtime_available: bool
    field_type: str
    category: str
    field_score: float
    coverage: float
    alpha_usage_count: int
    created_at: str


@dataclass(frozen=True, slots=True)
class SubmissionBatchRecord:
    batch_id: str
    run_id: str
    round_index: int
    backend: str
    status: str
    candidate_count: int
    sim_config_snapshot: str
    export_path: str | None
    notes_json: str
    created_at: str
    updated_at: str
    service_status_reason: str | None = None
    last_polled_at: str | None = None
    quarantined_at: str | None = None


@dataclass(frozen=True, slots=True)
class SubmissionRecord:
    job_id: str
    batch_id: str
    run_id: str
    round_index: int
    candidate_id: str
    expression: str
    backend: str
    status: str
    sim_config_snapshot: str
    submitted_at: str
    updated_at: str
    completed_at: str | None
    export_path: str | None
    raw_submission_json: str
    error_message: str | None
    retry_count: int = 0
    last_polled_at: str | None = None
    next_poll_after: str | None = None
    timeout_deadline_at: str | None = None
    stuck_since: str | None = None
    service_failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ServiceDispatchQueueRecord:
    queue_item_id: str
    service_name: str
    run_id: str
    candidate_id: str
    source_round_index: int
    queue_position: int
    status: str
    batch_id: str | None
    job_id: str | None
    failure_reason: str | None
    created_at: str
    updated_at: str


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


@dataclass(frozen=True, slots=True)
class ManualImportRecord:
    import_id: str
    run_id: str
    batch_id: str
    source_path: str
    imported_count: int
    created_at: str


@dataclass(frozen=True, slots=True)
class ClosedLoopRunRecord:
    run_id: str
    backend: str
    status: str
    requested_rounds: int
    completed_rounds: int
    config_snapshot: str
    started_at: str
    finished_at: str | None


@dataclass(frozen=True, slots=True)
class ClosedLoopRoundRecord:
    run_id: str
    round_index: int
    status: str
    generated_count: int
    validated_count: int
    submitted_count: int
    completed_count: int
    selected_for_mutation_count: int
    mutated_children_count: int
    summary_json: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class DuplicateDecisionRecord:
    run_id: str
    round_index: int
    alpha_id: str
    stage: str
    decision: str
    reason_code: str
    matched_run_id: str
    matched_alpha_id: str
    matched_scope: str
    similarity_score: float
    normalized_match: bool
    metrics_json: str
    created_at: str


@dataclass(frozen=True, slots=True)
class CrowdingScoreRecord:
    run_id: str
    round_index: int
    alpha_id: str
    stage: str
    total_penalty: float
    family_penalty: float
    motif_penalty: float
    operator_path_penalty: float
    lineage_penalty: float
    batch_penalty: float
    historical_penalty: float
    hard_blocked: bool
    reason_codes_json: str
    metrics_json: str
    created_at: str


@dataclass(frozen=True, slots=True)
class StageMetricRecord:
    run_id: str
    round_index: int
    stage: str
    metrics_json: str
    created_at: str


@dataclass(frozen=True, slots=True)
class SelectionScoreRecord:
    run_id: str
    round_index: int
    alpha_id: str
    score_stage: str
    composite_score: float
    selected: bool
    rank: int | None
    reason_codes_json: str
    breakdown_json: str
    created_at: str


@dataclass(frozen=True, slots=True)
class RegimeSnapshotRecord:
    run_id: str
    round_index: int
    region: str
    legacy_regime_key: str
    global_regime_key: str
    market_regime_key: str
    effective_regime_key: str
    regime_label: str
    confidence: float
    features_json: str
    created_at: str


@dataclass(frozen=True, slots=True)
class MutationOutcomeRecord:
    run_id: str
    child_alpha_id: str
    parent_alpha_id: str
    parent_run_id: str
    mutation_mode: str
    family_signature: str
    effective_regime_key: str
    outcome_source: str
    parent_post_sim_score: float
    child_post_sim_score: float
    outcome_delta: float
    selected_for_simulation: bool
    selected_for_mutation: bool
    created_at: str


@dataclass(frozen=True, slots=True)
class ServiceRuntimeRecord:
    service_name: str
    service_run_id: str
    owner_token: str
    pid: int
    hostname: str
    status: str
    tick_id: int
    active_batch_id: str | None
    pending_job_count: int
    consecutive_failures: int
    cooldown_until: str | None
    last_heartbeat_at: str | None
    last_success_at: str | None
    last_error: str | None
    persona_url: str | None
    persona_wait_started_at: str | None
    persona_last_notification_at: str | None
    counters_json: str
    lock_expires_at: str | None
    started_at: str
    updated_at: str
    persona_confirmation_nonce: str | None = None
    persona_confirmation_last_prompt_at: str | None = None
    persona_confirmation_granted_at: str | None = None
    persona_confirmation_last_update_id: int | None = None
