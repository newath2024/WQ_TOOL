from __future__ import annotations

import sqlite3
from pathlib import Path


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
    complexity INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'generated',
    PRIMARY KEY (run_id, alpha_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_alphas_run_expression
    ON alphas(run_id, normalized_expression);

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
    regime_key TEXT NOT NULL,
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
    regime_key TEXT NOT NULL,
    pattern_id TEXT NOT NULL,
    pattern_kind TEXT NOT NULL,
    pattern_value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, alpha_id, pattern_id)
);

CREATE INDEX IF NOT EXISTS idx_alpha_pattern_membership_regime
    ON alpha_pattern_membership(regime_key, pattern_kind, pattern_id);

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
"""


REQUIRED_COLUMNS = {
    "runs": {
        "profile_name": "TEXT NOT NULL DEFAULT ''",
        "dataset_fingerprint": "TEXT NOT NULL DEFAULT ''",
        "selected_timeframe": "TEXT NOT NULL DEFAULT ''",
        "regime_key": "TEXT NOT NULL DEFAULT ''",
        "entry_command": "TEXT NOT NULL DEFAULT ''",
    },
    "alphas": {
        "generation_metadata": "TEXT NOT NULL DEFAULT '{}'",
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
        "rejection_reasons_json": "TEXT NOT NULL DEFAULT '[]'",
        "metric_source": "TEXT NOT NULL DEFAULT 'local_backtest'",
    },
    "submission_batches": {
        "round_index": "INTEGER NOT NULL DEFAULT 0",
        "export_path": "TEXT",
        "notes_json": "TEXT NOT NULL DEFAULT '{}'",
    },
    "submissions": {
        "round_index": "INTEGER NOT NULL DEFAULT 0",
        "completed_at": "TEXT",
        "export_path": "TEXT",
        "raw_submission_json": "TEXT NOT NULL DEFAULT '{}'",
        "error_message": "TEXT",
    },
    "brain_results": {
        "round_index": "INTEGER NOT NULL DEFAULT 0",
        "submission_eligible": "INTEGER",
        "metric_source": "TEXT NOT NULL DEFAULT 'external_brain'",
    },
    "closed_loop_runs": {
        "completed_rounds": "INTEGER NOT NULL DEFAULT 0",
        "finished_at": "TEXT",
    },
    "closed_loop_rounds": {
        "summary_json": "TEXT NOT NULL DEFAULT '{}'",
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
    connection.executescript(DDL)
    _ensure_required_columns(connection)
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
