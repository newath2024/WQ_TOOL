from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from storage.models import BrainResultRecord, SubmissionBatchRecord, SubmissionRecord
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
