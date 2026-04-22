from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from storage.models import (
    BrainResultRecord,
    ClosedLoopRoundRecord,
    RegimeSnapshotRecord,
    SelectionScoreRecord,
    ServiceDispatchQueueRecord,
    ServiceRuntimeRecord,
    SubmissionBatchRecord,
    SubmissionRecord,
)
from storage.repository import SQLiteRepository


def test_show_brain_status_script_outputs_json_for_run(tmp_path: Path) -> None:
    database_path = tmp_path / "status.sqlite3"
    repository = SQLiteRepository(str(database_path))
    try:
        timestamp = datetime.now(UTC).isoformat()
        repository.upsert_run(
            run_id="run-status",
            seed=7,
            config_path="config/dev.yaml",
            config_snapshot="{}",
            status="service_running",
            started_at=timestamp,
            entry_command="run-service",
        )
        repository.save_dataset_summary("run-status", summary={}, regime_key="status-regime")
        repository.service_dispatch_queue.upsert_items(
            [
                ServiceDispatchQueueRecord(
                    queue_item_id="queue-1",
                    service_name="brain-service",
                    run_id="run-status",
                    candidate_id="alpha-2",
                    source_round_index=1,
                    queue_position=1,
                    status="queued",
                    batch_id=None,
                    job_id=None,
                    failure_reason=None,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            ]
        )
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-1",
                run_id="run-status",
                round_index=1,
                backend="api",
                status="running",
                candidate_count=1,
                sim_config_snapshot="{}",
                export_path=None,
                notes_json="{}",
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        repository.submissions.upsert_submissions(
            [
                SubmissionRecord(
                    job_id="job-1",
                    batch_id="batch-1",
                    run_id="run-status",
                    round_index=1,
                    candidate_id="alpha-1",
                    expression="rank(close)",
                    backend="api",
                    status="running",
                    sim_config_snapshot="{}",
                    submitted_at=timestamp,
                    updated_at=timestamp,
                    completed_at=None,
                    export_path=None,
                    raw_submission_json="{}",
                    error_message=None,
                )
            ]
        )
        repository.brain_results.save_results(
            [
                BrainResultRecord(
                    job_id="job-1",
                    run_id="run-status",
                    round_index=1,
                    batch_id="batch-1",
                    candidate_id="alpha-1",
                    expression="rank(close)",
                    status="completed",
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                    neutralization="SECTOR",
                    decay=0,
                    sharpe=1.1,
                    fitness=0.9,
                    turnover=0.4,
                    drawdown=0.2,
                    returns=0.06,
                    margin=0.03,
                    submission_eligible=True,
                    rejection_reason=None,
                    raw_result_json="{}",
                    metric_source="external_brain",
                    simulated_at=timestamp,
                    created_at=timestamp,
                )
            ]
        )
    finally:
        repository.close()

    result = subprocess.run(
        [
            sys.executable,
            "tools/show_brain_status.py",
            "--db",
            str(database_path),
            "--run-id",
            "run-status",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["run_id"] == "run-status"
    assert payload["submission_batches"][0]["batch_id"] == "batch-1"
    assert payload["submissions"][0]["job_id"] == "job-1"
    assert payload["brain_results"][0]["fitness"] == 0.9
    assert payload["service_dispatch_queue"][0]["queue_item_id"] == "queue-1"


def test_status_report_script_outputs_kpi_json_for_run(tmp_path: Path) -> None:
    database_path = tmp_path / "status_report.sqlite3"
    repository = SQLiteRepository(str(database_path))
    try:
        timestamp = datetime.now(UTC).isoformat()
        repository.upsert_run(
            run_id="run-kpi-tool",
            seed=7,
            config_path="config/dev.yaml",
            config_snapshot="{}",
            status="service_running",
            started_at=timestamp,
            entry_command="run-service",
        )
        repository.service_runtime.upsert_state(
            ServiceRuntimeRecord(
                service_name="brain-service",
                service_run_id="run-kpi-tool",
                owner_token="owner-token",
                pid=1234,
                hostname="test-host",
                status="idle",
                tick_id=3,
                active_batch_id=None,
                pending_job_count=0,
                consecutive_failures=0,
                cooldown_until=None,
                last_heartbeat_at=timestamp,
                last_success_at=timestamp,
                last_error=None,
                persona_url=None,
                persona_wait_started_at=None,
                persona_last_notification_at=None,
                counters_json="{}",
                lock_expires_at=None,
                started_at=timestamp,
                updated_at=timestamp,
            )
        )
        repository.brain_results.upsert_closed_loop_round(
            ClosedLoopRoundRecord(
                run_id="run-kpi-tool",
                round_index=1,
                status="completed",
                generated_count=12,
                validated_count=6,
                submitted_count=1,
                completed_count=1,
                selected_for_mutation_count=1,
                mutated_children_count=0,
                summary_json="{}",
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        repository.save_selection_scores(
            [
                SelectionScoreRecord(
                    run_id="run-kpi-tool",
                    round_index=1,
                    alpha_id="alpha-1",
                    score_stage="pre_sim",
                    composite_score=0.42,
                    selected=True,
                    rank=1,
                    reason_codes_json="[]",
                    breakdown_json=json.dumps(
                        {
                            "score_stage": "pre_sim",
                            "composite_score": 0.42,
                            "components": {
                                "predicted_quality": 0.42,
                                "heuristic_predicted_quality": 0.31,
                                "ml_positive_outcome_prob": 0.86,
                                "blended_predicted_quality": 0.42,
                                "meta_model_train_rows": 550,
                                "meta_model_positive_rows": 70,
                                "meta_model_used": 1.0,
                            },
                            "reason_codes": [],
                        },
                        sort_keys=True,
                    ),
                    created_at=timestamp,
                )
            ]
        )
        repository.connection.execute(
            """
            INSERT INTO alpha_cases
            (run_id, alpha_id, region, regime_key, global_regime_key, market_regime_key, effective_regime_key,
             regime_label, regime_confidence, metric_source, family_signature, structural_signature_json,
             genome_hash, genome_json, motif, field_families_json, operator_path_json, complexity_bucket,
             turnover_bucket, horizon_bucket, mutation_mode, parent_family_signatures_json, fail_tags_json,
             success_tags_json, objective_vector_json, outcome_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-kpi-tool",
                "alpha-1",
                "USA",
                "regime-local",
                "regime-global",
                "market-key",
                "effective-key",
                "normal",
                0.6,
                "external_brain",
                "family-alpha-1",
                "{}",
                "",
                "{}",
                "motif-a",
                json.dumps(["price"]),
                json.dumps(["rank", "ts_mean"]),
                "moderate",
                "balanced",
                "medium",
                "novelty",
                "[]",
                "[]",
                "[]",
                "{}",
                0.9,
                timestamp,
            ),
        )
        repository.connection.commit()
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-1",
                run_id="run-kpi-tool",
                round_index=1,
                backend="api",
                status="completed",
                candidate_count=2,
                sim_config_snapshot="{}",
                export_path=None,
                notes_json="{}",
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        repository.submissions.upsert_submissions(
            [
                SubmissionRecord(
                    job_id="job-1",
                    batch_id="batch-1",
                    run_id="run-kpi-tool",
                    round_index=1,
                    candidate_id="alpha-1",
                    expression="rank(close)",
                    backend="api",
                    status="completed",
                    sim_config_snapshot="{}",
                    submitted_at=timestamp,
                    updated_at=timestamp,
                    completed_at=timestamp,
                    export_path=None,
                    raw_submission_json="{}",
                    error_message=None,
                ),
                SubmissionRecord(
                    job_id="job-2",
                    batch_id="batch-1",
                    run_id="run-kpi-tool",
                    round_index=1,
                    candidate_id="alpha-2",
                    expression="rank(open)",
                    backend="api",
                    status="timeout",
                    sim_config_snapshot="{}",
                    submitted_at=timestamp,
                    updated_at=timestamp,
                    completed_at=timestamp,
                    export_path=None,
                    raw_submission_json="{}",
                    error_message="poll_timeout_after_downtime",
                    service_failure_reason="poll_timeout_after_downtime",
                ),
            ]
        )
        repository.brain_results.save_results(
            [
                BrainResultRecord(
                    job_id="job-1",
                    run_id="run-kpi-tool",
                    round_index=1,
                    batch_id="batch-1",
                    candidate_id="alpha-1",
                    expression="rank(close)",
                    status="completed",
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                    neutralization="SECTOR",
                    decay=0,
                    sharpe=1.1,
                    fitness=0.9,
                    turnover=0.4,
                    drawdown=0.2,
                    returns=0.06,
                    margin=0.03,
                    submission_eligible=True,
                    rejection_reason=None,
                    raw_result_json="{}",
                    metric_source="external_brain",
                    simulated_at=timestamp,
                    created_at=timestamp,
                ),
                BrainResultRecord(
                    job_id="job-2",
                    run_id="run-kpi-tool",
                    round_index=1,
                    batch_id="batch-1",
                    candidate_id="alpha-2",
                    expression="rank(open)",
                    status="timeout",
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                    neutralization="SECTOR",
                    decay=0,
                    sharpe=None,
                    fitness=None,
                    turnover=None,
                    drawdown=None,
                    returns=None,
                    margin=None,
                    submission_eligible=None,
                    rejection_reason="poll_timeout_after_downtime",
                    raw_result_json="{}",
                    metric_source="external_brain",
                    simulated_at=timestamp,
                    created_at=timestamp,
                ),
            ]
        )
        repository.save_regime_snapshots(
            [
                RegimeSnapshotRecord(
                    run_id="run-kpi-tool",
                    round_index=1,
                    region="USA",
                    legacy_regime_key="legacy-key",
                    global_regime_key="global-key",
                    market_regime_key="learned_cluster:4",
                    effective_regime_key="effective-key",
                    regime_label="learned",
                    confidence=0.7,
                    features_json=json.dumps({"learned_cluster_id": 4, "learned_confidence": 0.7}, sort_keys=True),
                    created_at=timestamp,
                )
            ]
        )
    finally:
        repository.close()

    result = subprocess.run(
        [
            sys.executable,
            "tools/status_report.py",
            "--db",
            str(database_path),
            "--run-id",
            "run-kpi-tool",
            "--recent-rounds",
            "1",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["run_id"] == "run-kpi-tool"
    assert payload["scope"]["scope_round_count"] == 1
    assert payload["health"]["completed_jobs"] == 1
    assert payload["health"]["timeout_jobs"] == 1
    assert payload["health"]["poll_timeout_after_downtime_jobs"] == 1
    assert payload["funnel"]["generated_count"] == 12
    assert payload["meta_model"]["meta_model_used_rate"] == 1.0
    assert payload["regime"]["latest_market_regime_key"] == "learned_cluster:4"
    assert payload["recent"]["raw_results"]["label"] == "recent_raw_results"
    assert payload["recent"]["raw_results"]["top_timeout_reasons"] == {"poll_timeout_after_downtime": 1}
    assert payload["timeout_reasons"]["recent"] == {"poll_timeout_after_downtime": 1}
    assert payload["delta_flags"]["raw_results"]["operations"] == "flat"

    human = subprocess.run(
        [
            sys.executable,
            "tools/status_report.py",
            "--db",
            str(database_path),
            "--run-id",
            "run-kpi-tool",
            "--recent-rounds",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "timeout_breakdown: live=0 after_downtime=1 legacy=0 other=0" in human.stdout
    assert "recent_vs_baseline:" in human.stdout
