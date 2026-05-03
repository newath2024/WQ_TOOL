from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from cli.commands import service_status
from core.config import load_config
from core.run_context import RunContext
from services.runtime_service import build_command_environment
from storage.models import (
    BrainResultRecord,
    ServiceDispatchQueueRecord,
    ServiceRuntimeRecord,
    SubmissionBatchRecord,
    SubmissionRecord,
)
from storage.repository import SQLiteRepository


def test_service_status_command_prints_human_summary(tmp_path: Path, capsys) -> None:
    repository = SQLiteRepository(str(tmp_path / "status.sqlite3"))
    try:
        timestamp = datetime.now(UTC).isoformat()
        _seed_run(repository, run_id="run-status", timestamp=timestamp)
        repository.service_runtime.upsert_state(_runtime_record(run_id="run-status", timestamp=timestamp))
        repository.service_dispatch_queue.upsert_items(
            [
                ServiceDispatchQueueRecord(
                    queue_item_id="queue-1",
                    service_name="brain-service",
                    run_id="run-status",
                    candidate_id="alpha-3",
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
                candidate_count=3,
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
                    run_id="run-status",
                    round_index=1,
                    candidate_id="alpha-2",
                    expression="rank(open)",
                    backend="api",
                    status="running",
                    sim_config_snapshot="{}",
                    submitted_at=timestamp,
                    updated_at=timestamp,
                    completed_at=None,
                    export_path=None,
                    raw_submission_json="{}",
                    error_message=None,
                ),
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
                    sharpe=1.2,
                    fitness=0.8,
                    turnover=0.3,
                    drawdown=0.1,
                    returns=0.05,
                    margin=0.02,
                    submission_eligible=True,
                    rejection_reason=None,
                    raw_result_json="{}",
                    metric_source="external_brain",
                    simulated_at=timestamp,
                    created_at=timestamp,
                    hard_fail_checks_json=json.dumps(["LOW_SHARPE"]),
                    derived_submit_ready=False,
                )
            ]
        )

        args = argparse.Namespace(run_id=None, json=False, limit=5, service_name=None)
        config = load_config("config/dev.yaml")
        environment = build_command_environment(
            config_path=str(Path("config/dev.yaml").resolve()),
            command_name="service-status",
            context=RunContext.create(seed=7, config_path="config/dev.yaml", run_id="run-status"),
        )

        code = service_status.handle(args, config, repository, environment)
        output = capsys.readouterr().out

        assert code == 0
        assert "service_name: brain-service" in output
        assert "run_id: run-status" in output
        assert "active_batch_id: batch-1" in output
        assert "active_batch_submission_counts: completed=1 running=1" in output
        assert "result_counts: completed=1" in output
        assert "derived_submit_ready_counts: no=1" in output
        assert "top_hard_fail_checks: LOW_SHARPE=1" in output
        assert "dispatch_queue_depth: 1" in output
        assert "dispatch_queue_counts: queued=1" in output
        assert "recent_dispatch_queue:" in output
    finally:
        repository.close()


def test_service_status_command_prints_json(tmp_path: Path, capsys) -> None:
    repository = SQLiteRepository(str(tmp_path / "status.sqlite3"))
    try:
        timestamp = datetime.now(UTC).isoformat()
        _seed_run(repository, run_id="run-json", timestamp=timestamp)
        repository.service_runtime.upsert_state(_runtime_record(run_id="run-json", timestamp=timestamp))
        repository.service_dispatch_queue.upsert_items(
            [
                ServiceDispatchQueueRecord(
                    queue_item_id="queue-json-1",
                    service_name="brain-service",
                    run_id="run-json",
                    candidate_id="alpha-json-1",
                    source_round_index=2,
                    queue_position=3,
                    status="dispatching",
                    batch_id=None,
                    job_id=None,
                    failure_reason=None,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            ]
        )

        args = argparse.Namespace(run_id="run-json", json=True, limit=3, service_name="brain-service")
        config = load_config("config/dev.yaml")
        environment = build_command_environment(
            config_path=str(Path("config/dev.yaml").resolve()),
            command_name="service-status",
            context=RunContext.create(seed=7, config_path="config/dev.yaml", run_id="run-json"),
        )

        code = service_status.handle(args, config, repository, environment)
        payload = json.loads(capsys.readouterr().out)

        assert code == 0
        assert payload["run_id"] == "run-json"
        assert payload["service_runtime"]["status"] == "running"
        assert payload["summary"]["submission_counts"] == {}
        assert payload["summary"]["queue_depth"] == 1
        assert payload["summary"]["queue_counts"] == {"dispatching": 1}
        assert payload["summary"]["top_hard_fail_checks"] == {}
        assert payload["recent_queue_items"][0]["queue_item_id"] == "queue-json-1"
    finally:
        repository.close()


def _runtime_record(*, run_id: str, timestamp: str) -> ServiceRuntimeRecord:
    return ServiceRuntimeRecord(
        service_name="brain-service",
        service_run_id=run_id,
        owner_token="owner-token",
        pid=1234,
        hostname="test-host",
        status="running",
        tick_id=4,
        active_batch_id="batch-1",
        pending_job_count=1,
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


def _seed_run(repository: SQLiteRepository, *, run_id: str, timestamp: str) -> None:
    repository.upsert_run(
        run_id=run_id,
        seed=7,
        config_path="config/dev.yaml",
        config_snapshot="{}",
        status="service_running",
        started_at=timestamp,
        profile_name="dev",
        selected_timeframe="1d",
        entry_command="run-service",
    )
