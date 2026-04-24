from __future__ import annotations

from core.config import load_config
from core.quality_score import MultiObjectiveQualityScorer
from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemorySnapshot
from services.brain_learning_service import BrainLearningService
from services.models import SimulationResult
from storage.models import BrainResultRecord, SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository


def test_persist_results_skips_poll_timeout_after_downtime_learning_rows() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        service = BrainLearningService(repository)
        candidate = AlphaCandidate(
            alpha_id="alpha-timeout",
            expression="rank(close)",
            normalized_expression="rank(close)",
            generation_mode="template",
            parent_ids=(),
            complexity=2,
            created_at="2026-04-21T00:00:00+00:00",
            template_name="momentum",
            fields_used=("close",),
            operators_used=("rank",),
            depth=2,
            generation_metadata={"motif": "momentum"},
        )
        result = SimulationResult(
            expression="rank(close)",
            job_id="job-timeout",
            status="timeout",
            region="USA",
            universe="TOP3000",
            delay=1,
            neutralization="SECTOR",
            decay=0,
            metrics={},
            submission_eligible=None,
            rejection_reason="poll_timeout_after_downtime",
            raw_result={},
            simulated_at="2026-04-21T00:00:00+00:00",
            candidate_id="alpha-timeout",
            batch_id="batch-timeout",
            run_id="run-timeout",
            round_index=1,
        )

        service.persist_results(
            config=config,
            regime_key="regime",
            region="USA",
            global_regime_key="global-regime",
            snapshot=PatternMemorySnapshot(regime_key="regime"),
            candidates_by_id={"alpha-timeout": candidate},
            results=[result],
        )

        history_count = repository.connection.execute("SELECT COUNT(*) AS total FROM alpha_history").fetchone()["total"]
        mutation_count = repository.connection.execute("SELECT COUNT(*) AS total FROM mutation_outcomes").fetchone()["total"]
    finally:
        repository.close()

    assert history_count == 0
    assert mutation_count == 0


def test_persist_results_writes_mutation_quality_scores() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        service = BrainLearningService(repository)
        repository.upsert_run(
            run_id="run-quality",
            seed=42,
            config_path="config/dev.yaml",
            config_snapshot="{}",
            status="running",
            started_at="2026-04-21T00:00:00+00:00",
        )
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-parent",
                run_id="run-quality",
                round_index=1,
                backend="api",
                status="completed",
                candidate_count=1,
                sim_config_snapshot="{}",
                export_path=None,
                notes_json="{}",
                created_at="2026-04-21T00:00:00+00:00",
                updated_at="2026-04-21T00:00:00+00:00",
            )
        )
        repository.submissions.upsert_submissions(
            [
                SubmissionRecord(
                    job_id="job-parent",
                    batch_id="batch-parent",
                    run_id="run-quality",
                    round_index=1,
                    candidate_id="parent-1",
                    expression="rank(open)",
                    backend="api",
                    status="completed",
                    sim_config_snapshot="{}",
                    submitted_at="2026-04-21T00:00:00+00:00",
                    updated_at="2026-04-21T00:00:00+00:00",
                    completed_at="2026-04-21T00:00:00+00:00",
                    export_path=None,
                    raw_submission_json="{}",
                    error_message=None,
                )
            ]
        )
        parent_result = BrainResultRecord(
            job_id="job-parent",
            run_id="run-quality",
            round_index=1,
            batch_id="batch-parent",
            candidate_id="parent-1",
            expression="rank(open)",
            status="completed",
            region="USA",
            universe="TOP3000",
            delay=1,
            neutralization="SECTOR",
            decay=0,
            sharpe=0.1,
            fitness=0.1,
            turnover=0.5,
            drawdown=0.3,
            returns=0.01,
            margin=0.01,
            submission_eligible=False,
            rejection_reason=None,
            raw_result_json="{}",
            metric_source="external_brain",
            simulated_at="2026-04-21T00:00:00+00:00",
            created_at="2026-04-21T00:00:00+00:00",
        )
        repository.brain_results.save_results([parent_result])
        candidate = AlphaCandidate(
            alpha_id="child-1",
            expression="rank(close)",
            normalized_expression="rank(close)",
            generation_mode="structural",
            parent_ids=("parent-1",),
            complexity=2,
            created_at="2026-04-21T00:01:00+00:00",
            template_name="momentum",
            fields_used=("close",),
            operators_used=("rank",),
            depth=2,
            generation_metadata={"motif": "momentum"},
        )
        result = SimulationResult(
            expression="rank(close)",
            job_id="job-child",
            status="completed",
            region="USA",
            universe="TOP3000",
            delay=1,
            neutralization="SECTOR",
            decay=0,
            metrics={"fitness": 0.5, "sharpe": 0.6, "turnover": 0.2, "drawdown": 0.05, "returns": 0.08, "margin": 0.06},
            submission_eligible=True,
            rejection_reason=None,
            raw_result={},
            simulated_at="2026-04-21T00:02:00+00:00",
            candidate_id="child-1",
            batch_id="batch-child",
            run_id="run-quality",
            round_index=2,
        )

        service.persist_results(
            config=config,
            regime_key="regime",
            region="USA",
            global_regime_key="global-regime",
            snapshot=PatternMemorySnapshot(regime_key="regime"),
            candidates_by_id={"child-1": candidate},
            results=[result],
        )

        row = repository.connection.execute("SELECT * FROM mutation_outcomes WHERE child_alpha_id = 'child-1'").fetchone()
    finally:
        repository.close()

    assert row is not None
    expected_parent_quality = MultiObjectiveQualityScorer.score_record(parent_result)
    expected_child_quality = MultiObjectiveQualityScorer.score_result(result)
    assert row["parent_quality_score"] == expected_parent_quality
    assert row["child_quality_score"] == expected_child_quality
    assert row["quality_delta"] == expected_child_quality - expected_parent_quality
