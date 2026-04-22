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
