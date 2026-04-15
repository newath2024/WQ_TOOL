from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from cli.commands import recover_brain_jobs
from core.config import load_config
from core.run_context import RunContext
from services.models import CommandEnvironment
from storage.models import SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository


class FakeRecoverAdapter:
    def __init__(self, *, authenticated: bool, status_payloads: dict[str, dict] | None = None) -> None:
        self.authenticated = authenticated
        self.status_payloads = dict(status_payloads or {})
        self.persona_timeout_seconds = 1800

    def probe_authenticated_session(self) -> dict:
        return {
            "authenticated": self.authenticated,
            "mode": "session_cookie" if self.authenticated else "not_authenticated",
            "session_path": "session.json",
        }

    def get_simulation_status(self, job_id: str) -> dict:
        return self.status_payloads.get(job_id, {"job_id": job_id, "status": "completed"})

    def get_simulation_result(self, job_id: str) -> dict:
        return {
            "job_id": job_id,
            "status": "completed",
            "raw_result": {
                "metrics": {
                    "sharpe": 1.3,
                    "fitness": 1.1,
                    "turnover": 0.4,
                    "drawdown": 0.1,
                    "returns": 0.08,
                    "margin": 0.05,
                },
                "submission_eligible": True,
            },
        }


def test_recover_brain_jobs_command_exits_cleanly_when_auth_is_not_ready(tmp_path, monkeypatch, capsys) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        run_id = "run-recover-auth"
        _seed_run(repository, run_id=run_id)
        _seed_pending_submission(repository, run_id=run_id, batch_id="batch-1", job_id="job-1")
        environment = _environment(run_id)
        args = argparse.Namespace(batch_id=None, job_id=["job-1"], json=True)
        monkeypatch.setattr(
            recover_brain_jobs,
            "_build_adapter",
            lambda _config: FakeRecoverAdapter(authenticated=False),
        )

        code = recover_brain_jobs.handle(args, config, repository, environment)
        payload = json.loads(capsys.readouterr().out)
        submission = repository.submissions.get_submission("job-1")
        results = repository.brain_results.list_results(run_id=run_id)
    finally:
        repository.close()

    assert code == 1
    assert payload["status"] == "auth_not_ready"
    assert submission is not None
    assert submission.status == "submitted"
    assert submission.timeout_deadline_at is None
    assert results == []


def test_recover_brain_jobs_command_recovers_pending_batch_by_id(monkeypatch, capsys) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        run_id = "run-recover-batch"
        _seed_run(repository, run_id=run_id)
        _seed_pending_submission(repository, run_id=run_id, batch_id="batch-1", job_id="job-1")
        environment = _environment(run_id)
        args = argparse.Namespace(batch_id="batch-1", job_id=None, json=True)
        monkeypatch.setattr(
            recover_brain_jobs,
            "_build_adapter",
            lambda _config: FakeRecoverAdapter(authenticated=True),
        )

        code = recover_brain_jobs.handle(args, config, repository, environment)
        payload = json.loads(capsys.readouterr().out)
        submission = repository.submissions.get_submission("job-1")
        batch = repository.submissions.get_batch("batch-1")
        results = repository.brain_results.list_results(run_id=run_id)
    finally:
        repository.close()

    assert code == 0
    assert payload["status"] == "recovered"
    assert payload["submission_counts"] == {"completed": 1}
    assert submission is not None
    assert submission.status == "completed"
    assert submission.timeout_deadline_at is None
    assert batch is not None
    assert batch.status == "completed"
    assert len(results) == 1


def _environment(run_id: str) -> CommandEnvironment:
    return CommandEnvironment(
        config_path="config/dev.yaml",
        command_name="recover-brain-jobs",
        context=RunContext.create(seed=7, config_path="config/dev.yaml", run_id=run_id),
    )


def _seed_run(repository: SQLiteRepository, *, run_id: str) -> None:
    timestamp = datetime.now(UTC).isoformat()
    repository.upsert_run(
        run_id=run_id,
        seed=7,
        config_path="config/dev.yaml",
        config_snapshot="{}",
        status="service_running",
        started_at=timestamp,
        entry_command="run-service",
    )


def _seed_pending_submission(repository: SQLiteRepository, *, run_id: str, batch_id: str, job_id: str) -> None:
    timestamp = datetime.now(UTC).isoformat()
    repository.submissions.upsert_batch(
        SubmissionBatchRecord(
            batch_id=batch_id,
            run_id=run_id,
            round_index=1,
            backend="api",
            status="submitted",
            candidate_count=1,
            sim_config_snapshot=json.dumps(
                {
                    "region": "USA",
                    "universe": "TOP3000",
                    "delay": 1,
                    "neutralization": "SECTOR",
                    "decay": 0,
                }
            ),
            export_path=None,
            notes_json="{}",
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    repository.submissions.upsert_submissions(
        [
            SubmissionRecord(
                job_id=job_id,
                batch_id=batch_id,
                run_id=run_id,
                round_index=1,
                candidate_id="alpha-1",
                expression="rank(close)",
                backend="api",
                status="submitted",
                sim_config_snapshot=json.dumps(
                    {
                        "region": "USA",
                        "universe": "TOP3000",
                        "delay": 1,
                        "neutralization": "SECTOR",
                        "decay": 0,
                    }
                ),
                submitted_at=timestamp,
                updated_at=timestamp,
                completed_at=None,
                export_path=None,
                raw_submission_json="{}",
                error_message=None,
            )
        ]
    )
