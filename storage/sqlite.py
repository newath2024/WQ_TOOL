from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml

from core.signatures import build_simulation_signature


DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    seed INTEGER NOT NULL,
    config_path TEXT NOT NULL,
    config_snapshot TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    dataset_summary TEXT,
    profile_name TEXT NOT NULL DEFAULT '',
    dataset_fingerprint TEXT NOT NULL DEFAULT '',
    selected_timeframe TEXT NOT NULL DEFAULT '',
    regime_key TEXT NOT NULL DEFAULT '',
    global_regime_key TEXT NOT NULL DEFAULT '',
    market_regime_key TEXT NOT NULL DEFAULT '',
    effective_regime_key TEXT NOT NULL DEFAULT '',
    regime_label TEXT NOT NULL DEFAULT 'unknown',
    regime_confidence REAL NOT NULL DEFAULT 0.0,
    region TEXT NOT NULL DEFAULT '',
    entry_command TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS alphas (
    run_id TEXT NOT NULL,
    alpha_id TEXT NOT NULL,
    expression TEXT NOT NULL,
    normalized_expression TEXT NOT NULL,
    generation_mode TEXT NOT NULL,
    template_name TEXT NOT NULL DEFAULT '',
    fields_used_json TEXT NOT NULL DEFAULT '[]',
    operators_used_json TEXT NOT NULL DEFAULT '[]',
    depth INTEGER NOT NULL DEFAULT 0,
    generation_metadata TEXT NOT NULL DEFAULT '{}',
    structural_signature_json TEXT NOT NULL DEFAULT '{}',
    complexity INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'generated',
    PRIMARY KEY (run_id, alpha_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_alphas_run_expression
    ON alphas(run_id, normalized_expression);

CREATE INDEX IF NOT EXISTS idx_alphas_run_created_at
    ON alphas(run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS alpha_parents (
    run_id TEXT NOT NULL,
    alpha_id TEXT NOT NULL,
    parent_run_id TEXT NOT NULL DEFAULT '',
    parent_alpha_id TEXT NOT NULL,
    PRIMARY KEY (run_id, alpha_id, parent_alpha_id)
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    alpha_id TEXT NOT NULL,
    split TEXT NOT NULL,
    sharpe REAL NOT NULL,
    max_drawdown REAL NOT NULL,
    win_rate REAL NOT NULL,
    average_return REAL NOT NULL,
    turnover REAL NOT NULL,
    observation_count INTEGER NOT NULL,
    cumulative_return REAL NOT NULL,
    fitness REAL NOT NULL,
    stability_score REAL NOT NULL,
    passed_filters INTEGER NOT NULL,
    simulation_signature TEXT NOT NULL DEFAULT '',
    simulation_config_snapshot TEXT NOT NULL DEFAULT '',
    delay_mode TEXT NOT NULL DEFAULT 'd1',
    neutralization TEXT NOT NULL DEFAULT 'none',
    neutralization_profile TEXT NOT NULL DEFAULT '',
    submission_pass_count INTEGER NOT NULL DEFAULT 0,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, alpha_id, split)
);

CREATE TABLE IF NOT EXISTS selections (
    run_id TEXT NOT NULL,
    alpha_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    selected_at TEXT NOT NULL,
    validation_fitness REAL NOT NULL,
    reason TEXT NOT NULL,
    ranking_rationale_json TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, alpha_id)
);

CREATE TABLE IF NOT EXISTS submission_tests (
    run_id TEXT NOT NULL,
    alpha_id TEXT NOT NULL,
    test_name TEXT NOT NULL,
    passed INTEGER NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, alpha_id, test_name)
);

CREATE TABLE IF NOT EXISTS simulation_cache (
    simulation_signature TEXT PRIMARY KEY,
    normalized_expression TEXT NOT NULL,
    simulation_config_snapshot TEXT NOT NULL,
    delay_mode TEXT NOT NULL,
    neutralization TEXT NOT NULL,
    neutralization_profile TEXT NOT NULL,
    split_metrics_json TEXT NOT NULL,
    submission_tests_json TEXT NOT NULL,
    subuniverse_metrics_json TEXT NOT NULL,
    validation_signal_json TEXT NOT NULL,
    validation_returns_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alpha_history (
    run_id TEXT NOT NULL,
    alpha_id TEXT NOT NULL,
    region TEXT NOT NULL DEFAULT '',
    regime_key TEXT NOT NULL,
    global_regime_key TEXT NOT NULL DEFAULT '',
    market_regime_key TEXT NOT NULL DEFAULT '',
    effective_regime_key TEXT NOT NULL DEFAULT '',
    regime_label TEXT NOT NULL DEFAULT 'unknown',
    regime_confidence REAL NOT NULL DEFAULT 0.0,
    expression TEXT NOT NULL,
    normalized_expression TEXT NOT NULL,
    generation_mode TEXT NOT NULL,
    generation_metadata_json TEXT NOT NULL,
    parent_refs_json TEXT NOT NULL,
    structural_signature_json TEXT NOT NULL,
    gene_ids_json TEXT NOT NULL,
    train_metrics_json TEXT NOT NULL,
    validation_metrics_json TEXT NOT NULL,
    test_metrics_json TEXT NOT NULL,
    validation_signal_json TEXT NOT NULL,
    validation_returns_json TEXT NOT NULL,
    outcome_score REAL NOT NULL,
    behavioral_novelty_score REAL NOT NULL,
    passed_filters INTEGER NOT NULL,
    selected INTEGER NOT NULL,
    submission_pass_count INTEGER NOT NULL,
    diagnosis_summary_json TEXT NOT NULL,
    rejection_reasons_json TEXT NOT NULL DEFAULT '[]',
    metric_source TEXT NOT NULL DEFAULT 'local_backtest',
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, alpha_id)
);

CREATE INDEX IF NOT EXISTS idx_alpha_history_regime_outcome
    ON alpha_history(regime_key, outcome_score DESC);

CREATE INDEX IF NOT EXISTS idx_alpha_history_global_regime_outcome
    ON alpha_history(global_regime_key, outcome_score DESC);

CREATE INDEX IF NOT EXISTS idx_alpha_history_effective_regime_outcome
    ON alpha_history(effective_regime_key, outcome_score DESC);

CREATE TABLE IF NOT EXISTS alpha_diagnoses (
    run_id TEXT NOT NULL,
    alpha_id TEXT NOT NULL,
    tag_type TEXT NOT NULL,
    tag TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, alpha_id, tag_type, tag)
);

CREATE TABLE IF NOT EXISTS alpha_patterns (
    regime_key TEXT NOT NULL,
    pattern_id TEXT NOT NULL,
    pattern_kind TEXT NOT NULL,
    pattern_value TEXT NOT NULL,
    support INTEGER NOT NULL,
    success_count INTEGER NOT NULL,
    failure_count INTEGER NOT NULL,
    avg_outcome REAL NOT NULL,
    avg_behavioral_novelty REAL NOT NULL,
    fail_tag_counts_json TEXT NOT NULL,
    pattern_score REAL NOT NULL,
    last_seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (regime_key, pattern_id)
);

CREATE INDEX IF NOT EXISTS idx_alpha_patterns_regime_score
    ON alpha_patterns(regime_key, pattern_kind, pattern_score DESC);

CREATE TABLE IF NOT EXISTS alpha_pattern_membership (
    run_id TEXT NOT NULL,
    alpha_id TEXT NOT NULL,
    region TEXT NOT NULL DEFAULT '',
    regime_key TEXT NOT NULL,
    global_regime_key TEXT NOT NULL DEFAULT '',
    pattern_id TEXT NOT NULL,
    pattern_kind TEXT NOT NULL,
    pattern_value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, alpha_id, pattern_id)
);

CREATE INDEX IF NOT EXISTS idx_alpha_pattern_membership_regime
    ON alpha_pattern_membership(regime_key, pattern_kind, pattern_id);

CREATE INDEX IF NOT EXISTS idx_alpha_pattern_membership_global_regime
    ON alpha_pattern_membership(global_regime_key, pattern_kind, pattern_id);

CREATE TABLE IF NOT EXISTS alpha_cases (
    run_id TEXT NOT NULL,
    alpha_id TEXT NOT NULL,
    region TEXT NOT NULL DEFAULT '',
    regime_key TEXT NOT NULL,
    global_regime_key TEXT NOT NULL DEFAULT '',
    market_regime_key TEXT NOT NULL DEFAULT '',
    effective_regime_key TEXT NOT NULL DEFAULT '',
    regime_label TEXT NOT NULL DEFAULT 'unknown',
    regime_confidence REAL NOT NULL DEFAULT 0.0,
    metric_source TEXT NOT NULL DEFAULT 'local_backtest',
    family_signature TEXT NOT NULL DEFAULT '',
    structural_signature_json TEXT NOT NULL DEFAULT '{}',
    genome_hash TEXT NOT NULL DEFAULT '',
    genome_json TEXT NOT NULL DEFAULT '{}',
    motif TEXT NOT NULL DEFAULT '',
    field_families_json TEXT NOT NULL DEFAULT '[]',
    operator_path_json TEXT NOT NULL DEFAULT '[]',
    complexity_bucket TEXT NOT NULL DEFAULT '',
    turnover_bucket TEXT NOT NULL DEFAULT '',
    horizon_bucket TEXT NOT NULL DEFAULT '',
    mutation_mode TEXT NOT NULL DEFAULT '',
    parent_family_signatures_json TEXT NOT NULL DEFAULT '[]',
    fail_tags_json TEXT NOT NULL DEFAULT '[]',
    success_tags_json TEXT NOT NULL DEFAULT '[]',
    objective_vector_json TEXT NOT NULL DEFAULT '{}',
    outcome_score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, alpha_id, metric_source)
);

CREATE INDEX IF NOT EXISTS idx_alpha_cases_regime_outcome
    ON alpha_cases(regime_key, metric_source, outcome_score DESC);

CREATE INDEX IF NOT EXISTS idx_alpha_cases_global_regime_outcome
    ON alpha_cases(global_regime_key, metric_source, outcome_score DESC);

CREATE INDEX IF NOT EXISTS idx_alpha_cases_effective_regime_outcome
    ON alpha_cases(effective_regime_key, metric_source, outcome_score DESC);

CREATE TABLE IF NOT EXISTS field_catalog (
    field_name TEXT PRIMARY KEY,
    dataset TEXT NOT NULL DEFAULT '',
    field_type TEXT NOT NULL DEFAULT 'matrix',
    coverage REAL NOT NULL DEFAULT 0.0,
    alpha_usage_count INTEGER NOT NULL DEFAULT 0,
    category TEXT NOT NULL DEFAULT 'other',
    delay INTEGER NOT NULL DEFAULT 1,
    region TEXT NOT NULL DEFAULT '',
    universe TEXT NOT NULL DEFAULT '',
    runtime_available INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    subcategory TEXT NOT NULL DEFAULT '',
    user_count INTEGER NOT NULL DEFAULT 0,
    category_weight REAL NOT NULL DEFAULT 0.5,
    field_score REAL NOT NULL DEFAULT 0.0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_field_scores (
    run_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    runtime_available INTEGER NOT NULL DEFAULT 0,
    field_type TEXT NOT NULL DEFAULT 'matrix',
    category TEXT NOT NULL DEFAULT 'other',
    field_score REAL NOT NULL DEFAULT 0.0,
    coverage REAL NOT NULL DEFAULT 0.0,
    alpha_usage_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, field_name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS submission_batches (
    batch_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    round_index INTEGER NOT NULL DEFAULT 0,
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    sim_config_snapshot TEXT NOT NULL DEFAULT '{}',
    export_path TEXT,
    notes_json TEXT NOT NULL DEFAULT '{}',
    service_status_reason TEXT,
    last_polled_at TEXT,
    quarantined_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS submissions (
    job_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    round_index INTEGER NOT NULL DEFAULT 0,
    candidate_id TEXT NOT NULL,
    expression TEXT NOT NULL,
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    sim_config_snapshot TEXT NOT NULL DEFAULT '{}',
    submitted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    export_path TEXT,
    raw_submission_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_polled_at TEXT,
    next_poll_after TEXT,
    timeout_deadline_at TEXT,
    stuck_since TEXT,
    service_failure_reason TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (batch_id) REFERENCES submission_batches(batch_id)
);

CREATE INDEX IF NOT EXISTS idx_submissions_run_status
    ON submissions(run_id, status, round_index);

CREATE TABLE IF NOT EXISTS brain_results (
    job_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    round_index INTEGER NOT NULL DEFAULT 0,
    batch_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    expression TEXT NOT NULL,
    status TEXT NOT NULL,
    region TEXT NOT NULL DEFAULT '',
    universe TEXT NOT NULL DEFAULT '',
    delay INTEGER NOT NULL DEFAULT 1,
    neutralization TEXT NOT NULL DEFAULT '',
    decay INTEGER NOT NULL DEFAULT 0,
    sharpe REAL,
    fitness REAL,
    turnover REAL,
    drawdown REAL,
    returns REAL,
    margin REAL,
    submission_eligible INTEGER,
    rejection_reason TEXT,
    raw_result_json TEXT NOT NULL DEFAULT '{}',
    metric_source TEXT NOT NULL DEFAULT 'external_brain',
    quality_score REAL NOT NULL DEFAULT 0.0,
    simulated_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (batch_id) REFERENCES submission_batches(batch_id),
    FOREIGN KEY (job_id) REFERENCES submissions(job_id)
);

CREATE INDEX IF NOT EXISTS idx_brain_results_run_round
    ON brain_results(run_id, round_index, status);

CREATE TABLE IF NOT EXISTS manual_imports (
    import_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    imported_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (batch_id) REFERENCES submission_batches(batch_id)
);

CREATE TABLE IF NOT EXISTS closed_loop_runs (
    run_id TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_rounds INTEGER NOT NULL,
    completed_rounds INTEGER NOT NULL DEFAULT 0,
    config_snapshot TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS closed_loop_rounds (
    run_id TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    generated_count INTEGER NOT NULL DEFAULT 0,
    validated_count INTEGER NOT NULL DEFAULT 0,
    submitted_count INTEGER NOT NULL DEFAULT 0,
    completed_count INTEGER NOT NULL DEFAULT 0,
    selected_for_mutation_count INTEGER NOT NULL DEFAULT 0,
    mutated_children_count INTEGER NOT NULL DEFAULT 0,
    summary_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, round_index),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS service_runtime (
    service_name TEXT PRIMARY KEY,
    service_run_id TEXT NOT NULL DEFAULT '',
    owner_token TEXT NOT NULL DEFAULT '',
    pid INTEGER NOT NULL DEFAULT 0,
    hostname TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'stopped',
    tick_id INTEGER NOT NULL DEFAULT 0,
    active_batch_id TEXT,
    pending_job_count INTEGER NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    cooldown_until TEXT,
    last_heartbeat_at TEXT,
    last_success_at TEXT,
    last_error TEXT,
    persona_url TEXT,
    persona_wait_started_at TEXT,
    persona_last_notification_at TEXT,
    persona_confirmation_nonce TEXT,
    persona_confirmation_last_prompt_at TEXT,
    persona_confirmation_granted_at TEXT,
    persona_confirmation_last_update_id INTEGER,
    counters_json TEXT NOT NULL DEFAULT '{}',
    lock_expires_at TEXT,
    started_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS service_dispatch_queue (
    queue_item_id TEXT PRIMARY KEY,
    service_name TEXT NOT NULL,
    run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    source_round_index INTEGER NOT NULL DEFAULT 0,
    queue_position INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    batch_id TEXT,
    job_id TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (batch_id) REFERENCES submission_batches(batch_id),
    FOREIGN KEY (job_id) REFERENCES submissions(job_id)
);

CREATE INDEX IF NOT EXISTS idx_service_dispatch_queue_active
    ON service_dispatch_queue(service_name, run_id, status, queue_position);

CREATE INDEX IF NOT EXISTS idx_service_dispatch_queue_round
    ON service_dispatch_queue(run_id, source_round_index, queue_position);

CREATE TABLE IF NOT EXISTS alpha_duplicate_decisions (
    run_id TEXT NOT NULL,
    round_index INTEGER NOT NULL DEFAULT 0,
    alpha_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    matched_run_id TEXT NOT NULL DEFAULT '',
    matched_alpha_id TEXT NOT NULL DEFAULT '',
    matched_scope TEXT NOT NULL DEFAULT '',
    similarity_score REAL NOT NULL DEFAULT 0.0,
    normalized_match INTEGER NOT NULL DEFAULT 0,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, round_index, stage, alpha_id, decision, reason_code)
);

CREATE INDEX IF NOT EXISTS idx_alpha_duplicate_decisions_stage
    ON alpha_duplicate_decisions(run_id, round_index, stage, decision, reason_code);

CREATE TABLE IF NOT EXISTS alpha_crowding_scores (
    run_id TEXT NOT NULL,
    round_index INTEGER NOT NULL DEFAULT 0,
    alpha_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    total_penalty REAL NOT NULL DEFAULT 0.0,
    family_penalty REAL NOT NULL DEFAULT 0.0,
    motif_penalty REAL NOT NULL DEFAULT 0.0,
    operator_path_penalty REAL NOT NULL DEFAULT 0.0,
    lineage_penalty REAL NOT NULL DEFAULT 0.0,
    batch_penalty REAL NOT NULL DEFAULT 0.0,
    historical_penalty REAL NOT NULL DEFAULT 0.0,
    hard_blocked INTEGER NOT NULL DEFAULT 0,
    reason_codes_json TEXT NOT NULL DEFAULT '[]',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, round_index, stage, alpha_id)
);

CREATE INDEX IF NOT EXISTS idx_alpha_crowding_scores_stage
    ON alpha_crowding_scores(run_id, round_index, stage, hard_blocked);

CREATE TABLE IF NOT EXISTS round_stage_metrics (
    run_id TEXT NOT NULL,
    round_index INTEGER NOT NULL DEFAULT 0,
    stage TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, round_index, stage)
);

CREATE TABLE IF NOT EXISTS alpha_selection_scores (
    run_id TEXT NOT NULL,
    round_index INTEGER NOT NULL DEFAULT 0,
    alpha_id TEXT NOT NULL,
    score_stage TEXT NOT NULL,
    composite_score REAL NOT NULL DEFAULT 0.0,
    selected INTEGER NOT NULL DEFAULT 0,
    rank INTEGER,
    reason_codes_json TEXT NOT NULL DEFAULT '[]',
    breakdown_json TEXT NOT NULL DEFAULT '{}',
    quality_score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, round_index, score_stage, alpha_id)
);

CREATE INDEX IF NOT EXISTS idx_alpha_selection_scores_stage
    ON alpha_selection_scores(run_id, round_index, score_stage, selected, composite_score DESC);

CREATE TABLE IF NOT EXISTS regime_snapshots (
    run_id TEXT NOT NULL,
    round_index INTEGER NOT NULL DEFAULT 0,
    region TEXT NOT NULL DEFAULT '',
    legacy_regime_key TEXT NOT NULL DEFAULT '',
    global_regime_key TEXT NOT NULL DEFAULT '',
    market_regime_key TEXT NOT NULL DEFAULT '',
    effective_regime_key TEXT NOT NULL DEFAULT '',
    regime_label TEXT NOT NULL DEFAULT 'unknown',
    confidence REAL NOT NULL DEFAULT 0.0,
    features_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, round_index)
);

CREATE INDEX IF NOT EXISTS idx_regime_snapshots_effective
    ON regime_snapshots(run_id, effective_regime_key, round_index);

CREATE TABLE IF NOT EXISTS mutation_outcomes (
    run_id TEXT NOT NULL,
    child_alpha_id TEXT NOT NULL,
    parent_alpha_id TEXT NOT NULL,
    parent_run_id TEXT NOT NULL DEFAULT '',
    mutation_mode TEXT NOT NULL DEFAULT '',
    family_signature TEXT NOT NULL DEFAULT '',
    effective_regime_key TEXT NOT NULL DEFAULT '',
    outcome_source TEXT NOT NULL DEFAULT '',
    parent_post_sim_score REAL NOT NULL DEFAULT 0.0,
    child_post_sim_score REAL NOT NULL DEFAULT 0.0,
    outcome_delta REAL NOT NULL DEFAULT 0.0,
    parent_quality_score REAL NOT NULL DEFAULT 0.0,
    child_quality_score REAL NOT NULL DEFAULT 0.0,
    quality_delta REAL NOT NULL DEFAULT 0.0,
    selected_for_simulation INTEGER NOT NULL DEFAULT 0,
    selected_for_mutation INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, child_alpha_id, parent_alpha_id, outcome_source)
);

CREATE INDEX IF NOT EXISTS idx_mutation_outcomes_effective_family_mode
    ON mutation_outcomes(effective_regime_key, family_signature, mutation_mode, outcome_delta DESC);
"""


def _filter_ddl_statements(*prefixes: str) -> str:
    statements: list[str] = []
    for raw_statement in DDL.split(";"):
        statement = raw_statement.strip()
        if not statement:
            continue
        if statement.startswith(prefixes):
            statements.append(f"{statement};")
    return "\n\n".join(statements)


TABLE_DDL = _filter_ddl_statements("CREATE TABLE")
INDEX_DDL = _filter_ddl_statements("CREATE INDEX", "CREATE UNIQUE INDEX")


REQUIRED_COLUMNS = {
    "runs": {
        "profile_name": "TEXT NOT NULL DEFAULT ''",
        "dataset_fingerprint": "TEXT NOT NULL DEFAULT ''",
        "selected_timeframe": "TEXT NOT NULL DEFAULT ''",
        "regime_key": "TEXT NOT NULL DEFAULT ''",
        "global_regime_key": "TEXT NOT NULL DEFAULT ''",
        "market_regime_key": "TEXT NOT NULL DEFAULT ''",
        "effective_regime_key": "TEXT NOT NULL DEFAULT ''",
        "regime_label": "TEXT NOT NULL DEFAULT 'unknown'",
        "regime_confidence": "REAL NOT NULL DEFAULT 0.0",
        "region": "TEXT NOT NULL DEFAULT ''",
        "entry_command": "TEXT NOT NULL DEFAULT ''",
    },
    "alphas": {
        "generation_metadata": "TEXT NOT NULL DEFAULT '{}'",
        "structural_signature_json": "TEXT NOT NULL DEFAULT '{}'",
        "template_name": "TEXT NOT NULL DEFAULT ''",
        "fields_used_json": "TEXT NOT NULL DEFAULT '[]'",
        "operators_used_json": "TEXT NOT NULL DEFAULT '[]'",
        "depth": "INTEGER NOT NULL DEFAULT 0",
    },
    "alpha_parents": {
        "parent_run_id": "TEXT NOT NULL DEFAULT ''",
    },
    "metrics": {
        "simulation_signature": "TEXT NOT NULL DEFAULT ''",
        "simulation_config_snapshot": "TEXT NOT NULL DEFAULT ''",
        "delay_mode": "TEXT NOT NULL DEFAULT 'd1'",
        "neutralization": "TEXT NOT NULL DEFAULT 'none'",
        "neutralization_profile": "TEXT NOT NULL DEFAULT ''",
        "submission_pass_count": "INTEGER NOT NULL DEFAULT 0",
        "cache_hit": "INTEGER NOT NULL DEFAULT 0",
    },
    "selections": {
        "ranking_rationale_json": "TEXT NOT NULL DEFAULT ''",
    },
    "alpha_history": {
        "region": "TEXT NOT NULL DEFAULT ''",
        "global_regime_key": "TEXT NOT NULL DEFAULT ''",
        "market_regime_key": "TEXT NOT NULL DEFAULT ''",
        "effective_regime_key": "TEXT NOT NULL DEFAULT ''",
        "regime_label": "TEXT NOT NULL DEFAULT 'unknown'",
        "regime_confidence": "REAL NOT NULL DEFAULT 0.0",
        "rejection_reasons_json": "TEXT NOT NULL DEFAULT '[]'",
        "metric_source": "TEXT NOT NULL DEFAULT 'local_backtest'",
    },
    "alpha_pattern_membership": {
        "region": "TEXT NOT NULL DEFAULT ''",
        "global_regime_key": "TEXT NOT NULL DEFAULT ''",
    },
    "alpha_cases": {
        "region": "TEXT NOT NULL DEFAULT ''",
        "global_regime_key": "TEXT NOT NULL DEFAULT ''",
        "market_regime_key": "TEXT NOT NULL DEFAULT ''",
        "effective_regime_key": "TEXT NOT NULL DEFAULT ''",
        "regime_label": "TEXT NOT NULL DEFAULT 'unknown'",
        "regime_confidence": "REAL NOT NULL DEFAULT 0.0",
        "metric_source": "TEXT NOT NULL DEFAULT 'local_backtest'",
        "family_signature": "TEXT NOT NULL DEFAULT ''",
        "structural_signature_json": "TEXT NOT NULL DEFAULT '{}'",
        "genome_hash": "TEXT NOT NULL DEFAULT ''",
        "genome_json": "TEXT NOT NULL DEFAULT '{}'",
        "motif": "TEXT NOT NULL DEFAULT ''",
        "field_families_json": "TEXT NOT NULL DEFAULT '[]'",
        "operator_path_json": "TEXT NOT NULL DEFAULT '[]'",
        "complexity_bucket": "TEXT NOT NULL DEFAULT ''",
        "turnover_bucket": "TEXT NOT NULL DEFAULT ''",
        "horizon_bucket": "TEXT NOT NULL DEFAULT ''",
        "mutation_mode": "TEXT NOT NULL DEFAULT ''",
        "parent_family_signatures_json": "TEXT NOT NULL DEFAULT '[]'",
        "fail_tags_json": "TEXT NOT NULL DEFAULT '[]'",
        "success_tags_json": "TEXT NOT NULL DEFAULT '[]'",
        "objective_vector_json": "TEXT NOT NULL DEFAULT '{}'",
        "outcome_score": "REAL NOT NULL DEFAULT 0.0",
    },
    "submission_batches": {
        "round_index": "INTEGER NOT NULL DEFAULT 0",
        "export_path": "TEXT",
        "notes_json": "TEXT NOT NULL DEFAULT '{}'",
        "service_status_reason": "TEXT",
        "last_polled_at": "TEXT",
        "quarantined_at": "TEXT",
    },
    "submissions": {
        "round_index": "INTEGER NOT NULL DEFAULT 0",
        "completed_at": "TEXT",
        "export_path": "TEXT",
        "raw_submission_json": "TEXT NOT NULL DEFAULT '{}'",
        "error_message": "TEXT",
        "retry_count": "INTEGER NOT NULL DEFAULT 0",
        "last_polled_at": "TEXT",
        "next_poll_after": "TEXT",
        "timeout_deadline_at": "TEXT",
        "stuck_since": "TEXT",
        "service_failure_reason": "TEXT",
    },
    "brain_results": {
        "round_index": "INTEGER NOT NULL DEFAULT 0",
        "submission_eligible": "INTEGER",
        "metric_source": "TEXT NOT NULL DEFAULT 'external_brain'",
        "quality_score": "REAL NOT NULL DEFAULT 0.0",
    },
    "alpha_selection_scores": {
        "quality_score": "REAL NOT NULL DEFAULT 0.0",
    },
    "mutation_outcomes": {
        "parent_quality_score": "REAL NOT NULL DEFAULT 0.0",
        "child_quality_score": "REAL NOT NULL DEFAULT 0.0",
        "quality_delta": "REAL NOT NULL DEFAULT 0.0",
    },
    "closed_loop_runs": {
        "completed_rounds": "INTEGER NOT NULL DEFAULT 0",
        "finished_at": "TEXT",
    },
    "closed_loop_rounds": {
        "summary_json": "TEXT NOT NULL DEFAULT '{}'",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    },
    "service_runtime": {
        "service_run_id": "TEXT NOT NULL DEFAULT ''",
        "owner_token": "TEXT NOT NULL DEFAULT ''",
        "pid": "INTEGER NOT NULL DEFAULT 0",
        "hostname": "TEXT NOT NULL DEFAULT ''",
        "status": "TEXT NOT NULL DEFAULT 'stopped'",
        "tick_id": "INTEGER NOT NULL DEFAULT 0",
        "active_batch_id": "TEXT",
        "pending_job_count": "INTEGER NOT NULL DEFAULT 0",
        "consecutive_failures": "INTEGER NOT NULL DEFAULT 0",
        "cooldown_until": "TEXT",
        "last_heartbeat_at": "TEXT",
        "last_success_at": "TEXT",
        "last_error": "TEXT",
        "persona_url": "TEXT",
        "persona_wait_started_at": "TEXT",
        "persona_last_notification_at": "TEXT",
        "persona_confirmation_nonce": "TEXT",
        "persona_confirmation_last_prompt_at": "TEXT",
        "persona_confirmation_granted_at": "TEXT",
        "persona_confirmation_last_update_id": "INTEGER",
        "counters_json": "TEXT NOT NULL DEFAULT '{}'",
        "lock_expires_at": "TEXT",
        "started_at": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    },
}


def connect_sqlite(path: str) -> sqlite3.Connection:
    if path == ":memory:":
        connection = sqlite3.connect(path)
    else:
        db_path = Path(path).expanduser().resolve()
        connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.executescript(TABLE_DDL)
    _ensure_required_columns(connection)
    _backfill_region_learning_columns(connection)
    connection.executescript(INDEX_DDL)
    connection.commit()
    return connection


def _ensure_required_columns(connection: sqlite3.Connection) -> None:
    for table_name, columns in REQUIRED_COLUMNS.items():
        existing = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, column_definition in columns.items():
            if column_name not in existing:
                connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


def _backfill_region_learning_columns(connection: sqlite3.Connection) -> None:
    run_rows = connection.execute(
        """
        SELECT run_id, config_snapshot, dataset_fingerprint, regime_key, global_regime_key, region
        FROM runs
        """
    ).fetchall()
    for row in run_rows:
        region = str(row["region"] or "").strip().upper()
        regime_key = str(row["regime_key"] or "")
        global_regime_key = str(row["global_regime_key"] or "")
        payload = _parse_config_snapshot(row["config_snapshot"])
        dataset_fingerprint = str(row["dataset_fingerprint"] or "")
        if payload is not None and dataset_fingerprint:
            resolved_region, local_key, global_key = _build_learning_keys_from_payload(
                dataset_fingerprint=dataset_fingerprint,
                payload=payload,
            )
            if resolved_region:
                region = resolved_region
            if resolved_region or not regime_key:
                regime_key = local_key
            if global_key:
                global_regime_key = global_key
        elif regime_key and not global_regime_key:
            global_regime_key = regime_key
        connection.execute(
            """
            UPDATE runs
            SET regime_key = COALESCE(NULLIF(?, ''), regime_key),
                global_regime_key = COALESCE(NULLIF(?, ''), global_regime_key),
                effective_regime_key = COALESCE(NULLIF(effective_regime_key, ''), NULLIF(?, ''), regime_key),
                market_regime_key = COALESCE(NULLIF(market_regime_key, ''), ''),
                regime_label = COALESCE(NULLIF(regime_label, ''), 'unknown'),
                regime_confidence = COALESCE(regime_confidence, 0.0),
                region = COALESCE(NULLIF(?, ''), region)
            WHERE run_id = ?
            """,
            (
                regime_key,
                global_regime_key,
                regime_key,
                region,
                row["run_id"],
            ),
        )

    connection.execute(
        """
        UPDATE runs
        SET region = COALESCE(
                NULLIF(
                    (
                        SELECT brain_results.region
                        FROM brain_results
                        WHERE brain_results.run_id = runs.run_id
                          AND brain_results.region <> ''
                        ORDER BY brain_results.created_at DESC
                        LIMIT 1
                    ),
                    ''
                ),
                region
            ),
            global_regime_key = CASE
                WHEN global_regime_key = '' THEN regime_key
                ELSE global_regime_key
            END,
            effective_regime_key = CASE
                WHEN effective_regime_key = '' THEN regime_key
                ELSE effective_regime_key
            END,
            regime_label = CASE
                WHEN regime_label = '' THEN 'unknown'
                ELSE regime_label
            END
        """
    )

    for table_name, candidate_column in (("alpha_history", "alpha_id"), ("alpha_cases", "alpha_id")):
        connection.execute(
            f"""
            UPDATE {table_name}
            SET region = COALESCE(
                    NULLIF((SELECT runs.region FROM runs WHERE runs.run_id = {table_name}.run_id), ''),
                    region
                ),
                regime_key = COALESCE(
                    NULLIF((SELECT runs.regime_key FROM runs WHERE runs.run_id = {table_name}.run_id), ''),
                    regime_key
                ),
                global_regime_key = COALESCE(
                    NULLIF((SELECT runs.global_regime_key FROM runs WHERE runs.run_id = {table_name}.run_id), ''),
                    global_regime_key
                ),
                effective_regime_key = COALESCE(
                    NULLIF((SELECT runs.effective_regime_key FROM runs WHERE runs.run_id = {table_name}.run_id), ''),
                    effective_regime_key,
                    regime_key
                )
            WHERE EXISTS (SELECT 1 FROM runs WHERE runs.run_id = {table_name}.run_id)
            """
        )
        connection.execute(
            f"""
            UPDATE {table_name}
            SET region = COALESCE(
                    NULLIF(
                        (
                            SELECT brain_results.region
                            FROM brain_results
                            WHERE brain_results.run_id = {table_name}.run_id
                              AND brain_results.candidate_id = {table_name}.{candidate_column}
                              AND brain_results.region <> ''
                            ORDER BY brain_results.created_at DESC
                            LIMIT 1
                        ),
                        ''
                    ),
                    region
                ),
                global_regime_key = CASE
                    WHEN global_regime_key = '' THEN regime_key
                    ELSE global_regime_key
                END,
                effective_regime_key = CASE
                    WHEN effective_regime_key = '' THEN regime_key
                    ELSE effective_regime_key
                END,
                regime_label = CASE
                    WHEN regime_label = '' THEN 'unknown'
                    ELSE regime_label
                END
            WHERE COALESCE(region, '') = '' OR COALESCE(global_regime_key, '') = '' OR COALESCE(effective_regime_key, '') = ''
            """
        )

    connection.execute(
        """
        UPDATE alpha_pattern_membership
        SET region = COALESCE(
                NULLIF((SELECT runs.region FROM runs WHERE runs.run_id = alpha_pattern_membership.run_id), ''),
                region
            ),
            regime_key = COALESCE(
                NULLIF((SELECT runs.regime_key FROM runs WHERE runs.run_id = alpha_pattern_membership.run_id), ''),
                regime_key
            ),
            global_regime_key = COALESCE(
                NULLIF((SELECT runs.global_regime_key FROM runs WHERE runs.run_id = alpha_pattern_membership.run_id), ''),
                global_regime_key
            )
        WHERE EXISTS (SELECT 1 FROM runs WHERE runs.run_id = alpha_pattern_membership.run_id)
        """
    )
    connection.execute(
        """
        UPDATE alpha_pattern_membership
        SET region = COALESCE(
                NULLIF(
                    (
                        SELECT brain_results.region
                        FROM brain_results
                        WHERE brain_results.run_id = alpha_pattern_membership.run_id
                          AND brain_results.candidate_id = alpha_pattern_membership.alpha_id
                          AND brain_results.region <> ''
                        ORDER BY brain_results.created_at DESC
                        LIMIT 1
                    ),
                    ''
                ),
                region
            ),
            global_regime_key = CASE
                WHEN global_regime_key = '' THEN regime_key
                ELSE global_regime_key
            END
        WHERE COALESCE(region, '') = '' OR COALESCE(global_regime_key, '') = ''
        """
    )


def _parse_config_snapshot(payload: str) -> dict | None:
    if not payload:
        return None
    try:
        parsed = yaml.safe_load(payload)
    except yaml.YAMLError:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _build_learning_keys_from_payload(*, dataset_fingerprint: str, payload: dict) -> tuple[str, str, str]:
    generation_payload = dict(payload.get("generation") or {})
    brain_payload = dict(payload.get("brain") or {})
    region = str(brain_payload.get("region") or "").strip().upper()
    base = {
        "dataset_fingerprint": dataset_fingerprint,
        "timeframe": str(((payload.get("backtest") or {}).get("timeframe")) or ""),
        "simulation": dict(payload.get("simulation") or {}),
        "backtest": dict(payload.get("backtest") or {}),
        "allowed_fields": list(generation_payload.get("allowed_fields") or []),
        "allowed_operators": list(generation_payload.get("allowed_operators") or []),
    }
    brain_profile = {
        "universe": str(brain_payload.get("universe") or "").strip().upper(),
        "delay": int(brain_payload.get("delay") or 1),
        "neutralization": str(brain_payload.get("neutralization") or "").strip().upper(),
        "decay": int(brain_payload.get("decay") or 0),
    }
    local_payload = dict(base)
    local_payload["brain_profile"] = dict(brain_profile, region=region)
    global_payload = dict(base)
    global_payload["brain_profile"] = dict(brain_profile)
    return (
        region,
        build_simulation_signature(local_payload),
        build_simulation_signature(global_payload),
    )
