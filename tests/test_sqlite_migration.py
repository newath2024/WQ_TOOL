from __future__ import annotations

import sqlite3
from pathlib import Path

from storage.sqlite import connect_sqlite


def test_connect_sqlite_migrates_legacy_tables_before_creating_new_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    legacy = sqlite3.connect(str(db_path))
    try:
        legacy.executescript(
            """
            CREATE TABLE runs (
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
                regime_key TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE alpha_history (
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
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, alpha_id)
            );
            """
        )
        legacy.commit()
    finally:
        legacy.close()

    connection = connect_sqlite(str(db_path))
    try:
        runs_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }
        alpha_history_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(alpha_history)").fetchall()
        }
        indexes = {
            row["name"]
            for row in connection.execute("PRAGMA index_list(alpha_history)").fetchall()
        }
    finally:
        connection.close()

    assert "global_regime_key" in runs_columns
    assert "global_regime_key" in alpha_history_columns
    assert "idx_alpha_history_global_regime_outcome" in indexes


def test_connect_sqlite_migrates_legacy_alphas_structural_signature(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-alphas.sqlite3"
    legacy = sqlite3.connect(str(db_path))
    try:
        legacy.executescript(
            """
            CREATE TABLE alphas (
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
                PRIMARY KEY (run_id, alpha_id)
            );
            """
        )
        legacy.commit()
    finally:
        legacy.close()

    connection = connect_sqlite(str(db_path))
    try:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(alphas)").fetchall()
        }
        indexes = {
            row["name"]
            for row in connection.execute("PRAGMA index_list(alphas)").fetchall()
        }
    finally:
        connection.close()

    assert "structural_signature_json" in columns
    assert "idx_alphas_run_expression" in indexes
    assert "idx_alphas_run_created_at" in indexes


def test_connect_sqlite_migrates_quality_score_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-quality.sqlite3"
    legacy = sqlite3.connect(str(db_path))
    try:
        legacy.executescript(
            """
            CREATE TABLE brain_results (
                job_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                round_index INTEGER NOT NULL DEFAULT 0,
                batch_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                expression TEXT NOT NULL,
                status TEXT NOT NULL,
                region TEXT NOT NULL,
                universe TEXT NOT NULL,
                delay INTEGER NOT NULL,
                neutralization TEXT NOT NULL,
                decay INTEGER NOT NULL,
                sharpe REAL,
                fitness REAL,
                turnover REAL,
                drawdown REAL,
                returns REAL,
                margin REAL,
                submission_eligible INTEGER,
                rejection_reason TEXT,
                raw_result_json TEXT NOT NULL,
                metric_source TEXT NOT NULL DEFAULT 'external_brain',
                simulated_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE alpha_selection_scores (
                run_id TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                alpha_id TEXT NOT NULL,
                score_stage TEXT NOT NULL,
                composite_score REAL NOT NULL,
                selected INTEGER NOT NULL,
                rank INTEGER,
                reason_codes_json TEXT NOT NULL,
                breakdown_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, round_index, score_stage, alpha_id)
            );

            CREATE TABLE mutation_outcomes (
                run_id TEXT NOT NULL,
                child_alpha_id TEXT NOT NULL,
                parent_alpha_id TEXT NOT NULL,
                parent_run_id TEXT NOT NULL,
                mutation_mode TEXT NOT NULL,
                family_signature TEXT NOT NULL,
                effective_regime_key TEXT NOT NULL,
                outcome_source TEXT NOT NULL,
                parent_post_sim_score REAL NOT NULL,
                child_post_sim_score REAL NOT NULL,
                outcome_delta REAL NOT NULL,
                selected_for_simulation INTEGER NOT NULL,
                selected_for_mutation INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, child_alpha_id, parent_alpha_id, outcome_source)
            );
            """
        )
        legacy.commit()
    finally:
        legacy.close()

    connection = connect_sqlite(str(db_path))
    try:
        brain_columns = {row["name"] for row in connection.execute("PRAGMA table_info(brain_results)").fetchall()}
        selection_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(alpha_selection_scores)").fetchall()
        }
        mutation_columns = {row["name"] for row in connection.execute("PRAGMA table_info(mutation_outcomes)").fetchall()}
    finally:
        connection.close()

    assert "quality_score" in brain_columns
    assert "quality_score" in selection_columns
    assert {"parent_quality_score", "child_quality_score", "quality_delta"} <= mutation_columns
