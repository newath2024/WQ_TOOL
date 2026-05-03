from __future__ import annotations

import json
from types import SimpleNamespace

from cli.commands.backfill_brain_checks import handle
from core.config import load_config
from domain.brain import BrainResultRecord
from storage.models import SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository


def test_backfill_brain_checks_populates_check_columns_idempotently() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_result(repository)
        args = SimpleNamespace(run_id="run-1", limit=None)
        config = load_config("config/dev.yaml")

        assert handle(args, config, repository, SimpleNamespace()) == 0
        first = repository.brain_results.get_result("job-1")
        assert first is not None
        assert first.hard_fail_checks_json == '["LOW_SHARPE"]'
        assert first.blocking_warning_checks_json == '["REVERSION_COMPONENT"]'
        assert first.derived_submit_ready is False
        assert first.rejection_reason == "reversion"
        generic = repository.brain_results.get_result("job-2")
        assert generic is not None
        assert generic.hard_fail_checks_json == '["LOW_2Y_SHARPE"]'
        assert generic.derived_submit_ready is False
        assert generic.rejection_reason is None

        assert handle(args, config, repository, SimpleNamespace()) == 0
        second = repository.brain_results.get_result("job-1")
    finally:
        repository.close()

    assert second is not None
    assert second.check_summary_json == first.check_summary_json
    assert second.quality_score == first.quality_score


def _seed_result(repository: SQLiteRepository) -> None:
    created_at = "2026-01-01T00:00:00+00:00"
    repository.upsert_run(
        run_id="run-1",
        seed=7,
        config_path="config/dev.yaml",
        config_snapshot="{}",
        status="running",
        started_at=created_at,
    )
    repository.submissions.upsert_batch(
        SubmissionBatchRecord(
            batch_id="batch-1",
            run_id="run-1",
            round_index=1,
            backend="api",
            status="completed",
            candidate_count=2,
            sim_config_snapshot="{}",
            export_path=None,
            notes_json="{}",
            created_at=created_at,
            updated_at=created_at,
        )
    )
    repository.submissions.upsert_submissions(
        [
            SubmissionRecord(
                job_id="job-1",
                batch_id="batch-1",
                run_id="run-1",
                round_index=1,
                candidate_id="alpha-1",
                expression="rank(close)",
                backend="api",
                status="completed",
                sim_config_snapshot="{}",
                submitted_at=created_at,
                updated_at=created_at,
                completed_at=created_at,
                export_path=None,
                raw_submission_json="{}",
                error_message=None,
            ),
            SubmissionRecord(
                job_id="job-2",
                batch_id="batch-1",
                run_id="run-1",
                round_index=1,
                candidate_id="alpha-2",
                expression="rank(open)",
                backend="api",
                status="completed",
                sim_config_snapshot="{}",
                submitted_at=created_at,
                updated_at=created_at,
                completed_at=created_at,
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
                run_id="run-1",
                round_index=1,
                batch_id="batch-1",
                candidate_id="alpha-1",
                expression="rank(close)",
                status="completed",
                region="USA",
                universe="TOP3000",
                delay=1,
                neutralization="SUBINDUSTRY",
                decay=5,
                sharpe=0.7,
                fitness=0.4,
                turnover=0.1,
                drawdown=0.1,
                returns=0.01,
                margin=0.01,
                submission_eligible=None,
                rejection_reason=None,
                raw_result_json=json.dumps(
                    {
                        "alpha": {
                            "is": {
                                "checks": [
                                    {"name": "LOW_SHARPE", "result": "FAIL"},
                                    {
                                        "name": "REVERSION_COMPONENT",
                                        "result": "WARNING",
                                        "message": "reversion",
                                    },
                                ]
                            }
                        }
                    },
                    sort_keys=True,
                ),
                metric_source="external_brain",
                simulated_at=created_at,
                created_at=created_at,
            ),
            BrainResultRecord(
                job_id="job-2",
                run_id="run-1",
                round_index=1,
                batch_id="batch-1",
                candidate_id="alpha-2",
                expression="rank(open)",
                status="completed",
                region="USA",
                universe="TOP3000",
                delay=1,
                neutralization="SUBINDUSTRY",
                decay=5,
                sharpe=0.9,
                fitness=0.7,
                turnover=0.1,
                drawdown=0.1,
                returns=0.01,
                margin=0.01,
                submission_eligible=None,
                rejection_reason="2Y Sharpe too low",
                raw_result_json=json.dumps(
                    {
                        "alpha": {
                            "is": {
                                "checks": [
                                    {
                                        "name": "LOW_2Y_SHARPE",
                                        "result": "FAIL",
                                        "message": "2Y Sharpe too low",
                                    },
                                ]
                            }
                        }
                    },
                    sort_keys=True,
                ),
                metric_source="external_brain",
                simulated_at=created_at,
                created_at=created_at,
            ),
        ]
    )
