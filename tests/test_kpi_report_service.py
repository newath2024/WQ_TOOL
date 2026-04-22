from __future__ import annotations

import json

import pytest

from services.kpi_report_service import _fetch_alpha_outcomes, build_run_kpi_report, run_kpi_report_to_dict
from storage.models import (
    BrainResultRecord,
    ClosedLoopRoundRecord,
    MutationOutcomeRecord,
    RegimeSnapshotRecord,
    SelectionScoreRecord,
    ServiceRuntimeRecord,
    StageMetricRecord,
    SubmissionBatchRecord,
    SubmissionRecord,
)
from storage.repository import SQLiteRepository


def test_build_run_kpi_report_scopes_recent_rounds_and_extracts_kpis() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_kpi_report_run(repository)

        report = build_run_kpi_report(
            repository,
            service_name="brain-service",
            run_id="run-kpi",
            recent_rounds=2,
        )

        assert report.scope_label == "last_2_rounds"
        assert report.scope_round_start == 2
        assert report.scope_round_end == 3
        assert report.scope_round_count == 2

        assert report.health["submitted_jobs"] == 4
        assert report.health["terminal_jobs"] == 4
        assert report.health["completed_jobs"] == 2
        assert report.health["timeout_jobs"] == 1
        assert report.health["failed_jobs"] == 1
        assert report.health["completed_rate"] == pytest.approx(0.5)
        assert report.health["timeout_rate"] == pytest.approx(0.25)
        assert report.health["failed_rate"] == pytest.approx(0.25)
        assert report.health["pending_jobs_runtime"] == 1
        assert report.health["poll_timeout_live_jobs"] == 1
        assert report.health["poll_timeout_after_downtime_jobs"] == 0
        assert report.health["legacy_poll_timeout_jobs"] == 0
        assert report.health["other_timeout_jobs"] == 0

        assert report.funnel["generated_count"] == 50
        assert report.funnel["validated_count"] == 25
        assert report.funnel["selected_for_simulation"] == 2
        assert report.funnel["validation_rate"] == pytest.approx(0.5)
        assert report.funnel["selection_rate"] == pytest.approx(0.04)
        assert report.funnel["validate_fail_count"] == 22
        assert report.funnel["validation_disallowed_field_rate"] == pytest.approx(0.09)
        assert report.funnel["blocked_by_near_duplicate_rate"] == pytest.approx(0.1)

        assert report.quality["completed_results"] == 2
        assert report.quality["distinct_candidates"] == 4
        assert report.quality["positive_fitness_rate"] == pytest.approx(0.5)
        assert report.quality["positive_sharpe_rate"] == pytest.approx(1.0)
        assert report.quality["avg_fitness"] == pytest.approx(0.1)
        assert report.quality["max_sharpe"] == pytest.approx(0.6)
        assert report.quality["avg_drawdown"] == pytest.approx(0.15)
        assert report.quality["avg_returns"] == pytest.approx(0.025)

        assert report.meta_model["rows"] == 4
        assert report.meta_model["meta_model_used_rate"] == pytest.approx(1.0)
        assert report.meta_model["avg_train_rows"] == pytest.approx(600.0)
        assert report.meta_model["avg_positive_rows"] == pytest.approx(80.0)
        assert report.meta_model["avg_selected_prob"] == pytest.approx(0.85)
        assert report.meta_model["avg_archived_prob"] == pytest.approx(0.15)
        assert report.meta_model["selected_positive_outcome_rate"] == pytest.approx(1.0)
        assert report.meta_model["archived_positive_outcome_rate"] == pytest.approx(0.0)

        assert report.regime["latest_market_regime_key"] == "normal|flat|low"
        assert report.regime["latest_learned_cluster_id"] == 3
        assert report.regime["latest_learned_confidence"] == pytest.approx(0.2)
        assert report.regime["learned_active_rate"] == pytest.approx(0.5)
        assert report.regime["fallback_rate"] == pytest.approx(0.5)

        assert report.mutation["selected_for_mutation_count"] == 3
        assert report.mutation["mutated_children_count"] == 3
        assert report.mutation["child_better_than_parent_rate"] == pytest.approx(0.5)
        assert report.mutation["mutation_outcome_rows"] == 2
    finally:
        repository.close()


def test_build_run_kpi_report_marks_full_run_scope_when_recent_rounds_is_zero() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_kpi_report_run(repository)

        report = build_run_kpi_report(
            repository,
            service_name="brain-service",
            run_id="run-kpi",
            recent_rounds=0,
        )

        assert report.scope_label == "full_run"
        assert report.scope_round_start == 1
        assert report.scope_round_end == 3
        assert report.scope_round_count == 3
        assert report.funnel["generated_count"] == 60
    finally:
        repository.close()


def test_run_kpi_report_to_dict_includes_recent_vs_baseline_blocks() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        _seed_kpi_report_run(repository)

        payload = run_kpi_report_to_dict(
            build_run_kpi_report(
                repository,
                service_name="brain-service",
                run_id="run-kpi",
                recent_rounds=2,
            )
        )

        assert payload["recent"]["raw_results"]["label"] == "recent_raw_results"
        assert payload["recent"]["raw_results"]["top_timeout_reasons"] == {"poll_timeout_live": 1}
        assert payload["recent"]["rounds"]["top_generation_fail_reasons"]["validation_disallowed_field"] == 10
        assert payload["baseline"]["raw_results"]["top_timeout_reasons"] == {}
        assert payload["timeout_reasons"]["recent"] == {"poll_timeout_live": 1}
        assert payload["generation_fail_reasons"]["recent"]["validation_disallowed_field"] == 10
        assert payload["delta_flags"]["raw_results"]["quality"] == "flat"
        assert payload["delta_flags"]["raw_results"]["operations"] == "flat"
    finally:
        repository.close()


def test_fetch_alpha_outcomes_chunks_large_id_sets_without_hitting_sqlite_variable_limit() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        repository.upsert_run(
            run_id="run-kpi",
            seed=7,
            config_path="config/dev.yaml",
            config_snapshot="{}",
            status="service_running",
            started_at="2026-01-01T00:00:00+00:00",
        )
        alpha_ids: set[str] = set()
        for index in range(1205):
            alpha_id = f"alpha-{index:04d}"
            alpha_ids.add(alpha_id)
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
                    "run-kpi",
                    alpha_id,
                    "USA",
                    "regime-local",
                    "regime-global",
                    "market-key",
                    "effective-key",
                    "normal",
                    0.6,
                    "external_brain",
                    f"family-{alpha_id}",
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
                    float(index % 7) - 3.0,
                    "2026-01-01T00:00:00+00:00",
                ),
            )
        repository.connection.commit()

        outcomes = _fetch_alpha_outcomes(
            repository.connection,
            run_id="run-kpi",
            alpha_ids=alpha_ids,
        )

        assert len(outcomes) == 1205
        assert outcomes["alpha-0000"] == pytest.approx(-3.0)
        assert outcomes["alpha-1204"] == pytest.approx((1204 % 7) - 3.0)
    finally:
        repository.close()


def _seed_kpi_report_run(repository: SQLiteRepository) -> None:
    timestamp = "2026-01-01T00:00:00+00:00"
    repository.upsert_run(
        run_id="run-kpi",
        seed=7,
        config_path="config/dev.yaml",
        config_snapshot="{}",
        status="service_running",
        started_at=timestamp,
        profile_name="dev",
        selected_timeframe="1d",
        entry_command="run-service",
    )
    repository.service_runtime.upsert_state(
        ServiceRuntimeRecord(
            service_name="brain-service",
            service_run_id="run-kpi",
            owner_token="owner-token",
            pid=123,
            hostname="test-host",
            status="running",
            tick_id=9,
            active_batch_id="batch-3",
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
    )

    repository.brain_results.upsert_closed_loop_round(
        ClosedLoopRoundRecord(
            run_id="run-kpi",
            round_index=1,
            status="completed",
            generated_count=10,
            validated_count=5,
            submitted_count=2,
            completed_count=2,
            selected_for_mutation_count=1,
            mutated_children_count=0,
            summary_json="{}",
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    repository.brain_results.upsert_closed_loop_round(
        ClosedLoopRoundRecord(
            run_id="run-kpi",
            round_index=2,
            status="completed",
            generated_count=20,
            validated_count=10,
            submitted_count=2,
            completed_count=2,
            selected_for_mutation_count=2,
            mutated_children_count=1,
            summary_json="{}",
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    repository.brain_results.upsert_closed_loop_round(
        ClosedLoopRoundRecord(
            run_id="run-kpi",
            round_index=3,
            status="completed",
            generated_count=30,
            validated_count=15,
            submitted_count=2,
            completed_count=2,
            selected_for_mutation_count=1,
            mutated_children_count=2,
            summary_json="{}",
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    repository.save_stage_metrics(
        [
            StageMetricRecord(
                run_id="run-kpi",
                round_index=1,
                stage="generation",
                metrics_json=json.dumps(
                    {
                        "attempt_count": 20,
                        "generated": 10,
                        "selected_for_simulation": 1,
                        "validate_fail_count": 5,
                        "failure_reason_counts": {
                            "validation_disallowed_field": 1,
                            "validation_unknown_error": 4,
                        },
                    }
                ),
                created_at=timestamp,
            ),
            StageMetricRecord(
                run_id="run-kpi",
                round_index=2,
                stage="generation",
                metrics_json=json.dumps(
                    {
                        "attempt_count": 40,
                        "generated": 20,
                        "selected_for_simulation": 2,
                        "validate_fail_count": 10,
                        "failure_reason_counts": {
                            "validation_disallowed_field": 3,
                            "duplicate_normalized_expression": 7,
                        },
                    }
                ),
                created_at=timestamp,
            ),
            StageMetricRecord(
                run_id="run-kpi",
                round_index=3,
                stage="generation",
                metrics_json=json.dumps(
                    {
                        "attempt_count": 60,
                        "generated": 30,
                        "selected_for_simulation": 2,
                        "validate_fail_count": 12,
                        "failure_reason_counts": {
                            "validation_disallowed_field": 6,
                            "guard_unit_mismatch_history": 2,
                            "validation_unknown_error": 4,
                        },
                    }
                ),
                created_at=timestamp,
            ),
            StageMetricRecord(
                run_id="run-kpi",
                round_index=1,
                stage="pre_sim",
                metrics_json=json.dumps({"blocked_by_near_duplicate": 1}),
                created_at=timestamp,
            ),
            StageMetricRecord(
                run_id="run-kpi",
                round_index=2,
                stage="pre_sim",
                metrics_json=json.dumps({"blocked_by_near_duplicate": 2}),
                created_at=timestamp,
            ),
            StageMetricRecord(
                run_id="run-kpi",
                round_index=3,
                stage="pre_sim",
                metrics_json=json.dumps({"blocked_by_near_duplicate": 3}),
                created_at=timestamp,
            ),
        ]
    )

    repository.save_selection_scores(
        [
            _selection_score(round_index=1, alpha_id="alpha-old-selected", selected=True, probability=0.7, blended=0.5),
            _selection_score(round_index=2, alpha_id="alpha-selected-2", selected=True, probability=0.8, blended=0.55),
            _selection_score(round_index=2, alpha_id="alpha-archived-2", selected=False, probability=0.2, blended=0.35),
            _selection_score(round_index=3, alpha_id="alpha-selected-3", selected=True, probability=0.9, blended=0.60),
            _selection_score(round_index=3, alpha_id="alpha-archived-3", selected=False, probability=0.1, blended=0.30),
        ]
    )

    for alpha_id, round_index, outcome_score in (
        ("alpha-old-selected", 1, 1.0),
        ("alpha-selected-2", 2, 0.8),
        ("alpha-archived-2", 2, -0.3),
        ("alpha-selected-3", 3, 0.6),
        ("alpha-archived-3", 3, 0.0),
    ):
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
                "run-kpi",
                alpha_id,
                "USA",
                "regime-local",
                "regime-global",
                "market-key",
                "effective-key",
                "normal",
                0.6,
                "external_brain",
                f"family-{alpha_id}",
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
                outcome_score,
                f"2026-01-01T00:0{round_index}:00+00:00",
            ),
        )
    repository.connection.commit()

    repository.submissions.upsert_batch(
        SubmissionBatchRecord(
            batch_id="batch-1",
            run_id="run-kpi",
            round_index=1,
            backend="api",
            status="completed",
            candidate_count=1,
            sim_config_snapshot="{}",
            export_path=None,
            notes_json="{}",
            created_at="2026-01-01T00:01:00+00:00",
            updated_at="2026-01-01T00:01:00+00:00",
        )
    )
    repository.submissions.upsert_batch(
        SubmissionBatchRecord(
            batch_id="batch-2",
            run_id="run-kpi",
            round_index=2,
            backend="api",
            status="completed",
            candidate_count=2,
            sim_config_snapshot="{}",
            export_path=None,
            notes_json="{}",
            created_at="2026-01-01T00:02:00+00:00",
            updated_at="2026-01-01T00:02:00+00:00",
        )
    )
    repository.submissions.upsert_batch(
        SubmissionBatchRecord(
            batch_id="batch-3",
            run_id="run-kpi",
            round_index=3,
            backend="api",
            status="completed",
            candidate_count=2,
            sim_config_snapshot="{}",
            export_path=None,
            notes_json="{}",
            created_at="2026-01-01T00:03:00+00:00",
            updated_at="2026-01-01T00:03:00+00:00",
        )
    )
    repository.submissions.upsert_submissions(
        [
            SubmissionRecord(
                job_id="job-old",
                batch_id="batch-1",
                run_id="run-kpi",
                round_index=1,
                candidate_id="alpha-old-selected",
                expression="rank(close)",
                backend="api",
                status="completed",
                sim_config_snapshot="{}",
                submitted_at="2026-01-01T00:01:00+00:00",
                updated_at="2026-01-01T00:01:20+00:00",
                completed_at="2026-01-01T00:01:20+00:00",
                export_path=None,
                raw_submission_json="{}",
                error_message=None,
            ),
            SubmissionRecord(
                job_id="job-21",
                batch_id="batch-2",
                run_id="run-kpi",
                round_index=2,
                candidate_id="alpha-selected-2",
                expression="rank(close)",
                backend="api",
                status="completed",
                sim_config_snapshot="{}",
                submitted_at="2026-01-01T00:02:00+00:00",
                updated_at="2026-01-01T00:02:45+00:00",
                completed_at="2026-01-01T00:02:45+00:00",
                export_path=None,
                raw_submission_json="{}",
                error_message=None,
            ),
            SubmissionRecord(
                job_id="job-22",
                batch_id="batch-2",
                run_id="run-kpi",
                round_index=2,
                candidate_id="alpha-archived-2",
                expression="rank(open)",
                backend="api",
                status="timeout",
                sim_config_snapshot="{}",
                submitted_at="2026-01-01T00:02:00+00:00",
                updated_at="2026-01-01T00:05:00+00:00",
                completed_at="2026-01-01T00:05:00+00:00",
                export_path=None,
                raw_submission_json="{}",
                error_message="poll_timeout_live",
                service_failure_reason="poll_timeout_live",
            ),
            SubmissionRecord(
                job_id="job-31",
                batch_id="batch-3",
                run_id="run-kpi",
                round_index=3,
                candidate_id="alpha-selected-3",
                expression="rank(volume)",
                backend="api",
                status="completed",
                sim_config_snapshot="{}",
                submitted_at="2026-01-01T00:03:00+00:00",
                updated_at="2026-01-01T00:04:00+00:00",
                completed_at="2026-01-01T00:04:00+00:00",
                export_path=None,
                raw_submission_json="{}",
                error_message=None,
            ),
            SubmissionRecord(
                job_id="job-32",
                batch_id="batch-3",
                run_id="run-kpi",
                round_index=3,
                candidate_id="alpha-archived-3",
                expression="rank(low)",
                backend="api",
                status="failed",
                sim_config_snapshot="{}",
                submitted_at="2026-01-01T00:03:00+00:00",
                updated_at="2026-01-01T00:06:00+00:00",
                completed_at="2026-01-01T00:06:00+00:00",
                export_path=None,
                raw_submission_json="{}",
                error_message="api error",
            ),
        ]
    )
    repository.brain_results.save_results(
        [
            BrainResultRecord(
                job_id="job-old",
                run_id="run-kpi",
                round_index=1,
                batch_id="batch-1",
                candidate_id="alpha-old-selected",
                expression="rank(close)",
                status="completed",
                region="USA",
                universe="TOP3000",
                delay=1,
                neutralization="SECTOR",
                decay=0,
                sharpe=0.7,
                fitness=0.5,
                turnover=0.3,
                drawdown=0.1,
                returns=0.04,
                margin=0.02,
                submission_eligible=True,
                rejection_reason=None,
                raw_result_json="{}",
                metric_source="external_brain",
                simulated_at="2026-01-01T00:01:30+00:00",
                created_at="2026-01-01T00:01:30+00:00",
            ),
            BrainResultRecord(
                job_id="job-21",
                run_id="run-kpi",
                round_index=2,
                batch_id="batch-2",
                candidate_id="alpha-selected-2",
                expression="rank(close)",
                status="completed",
                region="USA",
                universe="TOP3000",
                delay=1,
                neutralization="SECTOR",
                decay=0,
                sharpe=0.6,
                fitness=0.4,
                turnover=0.3,
                drawdown=0.1,
                returns=0.04,
                margin=0.02,
                submission_eligible=True,
                rejection_reason=None,
                raw_result_json="{}",
                metric_source="external_brain",
                simulated_at="2026-01-01T00:02:50+00:00",
                created_at="2026-01-01T00:02:50+00:00",
            ),
            BrainResultRecord(
                job_id="job-31",
                run_id="run-kpi",
                round_index=3,
                batch_id="batch-3",
                candidate_id="alpha-selected-3",
                expression="rank(volume)",
                status="completed",
                region="USA",
                universe="TOP3000",
                delay=1,
                neutralization="SECTOR",
                decay=0,
                sharpe=0.1,
                fitness=-0.2,
                turnover=0.4,
                drawdown=0.2,
                returns=0.01,
                margin=0.01,
                submission_eligible=False,
                rejection_reason=None,
                raw_result_json="{}",
                metric_source="external_brain",
                simulated_at="2026-01-01T00:04:10+00:00",
                created_at="2026-01-01T00:04:10+00:00",
            ),
        ]
    )
    repository.save_regime_snapshots(
        [
            RegimeSnapshotRecord(
                run_id="run-kpi",
                round_index=2,
                region="USA",
                legacy_regime_key="legacy-key",
                global_regime_key="global-key",
                market_regime_key="learned_cluster:2",
                effective_regime_key="effective-key",
                regime_label="learned",
                confidence=0.6,
                features_json=json.dumps({"learned_cluster_id": 2, "learned_confidence": 0.55}, sort_keys=True),
                created_at="2026-01-01T00:02:00+00:00",
            ),
            RegimeSnapshotRecord(
                run_id="run-kpi",
                round_index=3,
                region="USA",
                legacy_regime_key="legacy-key",
                global_regime_key="global-key",
                market_regime_key="normal|flat|low",
                effective_regime_key="effective-key",
                regime_label="heuristic",
                confidence=0.2,
                features_json=json.dumps({"learned_cluster_id": 3, "learned_confidence": 0.2}, sort_keys=True),
                created_at="2026-01-01T00:03:00+00:00",
            ),
        ]
    )
    repository.save_mutation_outcomes(
        [
            MutationOutcomeRecord(
                run_id="run-kpi",
                child_alpha_id="child-good",
                parent_alpha_id="parent-a",
                parent_run_id="run-kpi",
                mutation_mode="structural",
                family_signature="family-a",
                effective_regime_key="effective-key",
                outcome_source="external_brain",
                parent_post_sim_score=0.3,
                child_post_sim_score=0.5,
                outcome_delta=0.2,
                selected_for_simulation=True,
                selected_for_mutation=True,
                created_at="2026-01-01T00:03:30+00:00",
            ),
            MutationOutcomeRecord(
                run_id="run-kpi",
                child_alpha_id="child-bad",
                parent_alpha_id="parent-b",
                parent_run_id="run-kpi",
                mutation_mode="structural",
                family_signature="family-b",
                effective_regime_key="effective-key",
                outcome_source="external_brain",
                parent_post_sim_score=0.4,
                child_post_sim_score=0.1,
                outcome_delta=-0.3,
                selected_for_simulation=True,
                selected_for_mutation=True,
                created_at="2026-01-01T00:03:40+00:00",
            ),
        ]
    )


def _selection_score(
    *,
    round_index: int,
    alpha_id: str,
    selected: bool,
    probability: float,
    blended: float,
) -> SelectionScoreRecord:
    return SelectionScoreRecord(
        run_id="run-kpi",
        round_index=round_index,
        alpha_id=alpha_id,
        score_stage="pre_sim",
        composite_score=blended,
        selected=selected,
        rank=1 if selected else None,
        reason_codes_json="[]",
        breakdown_json=json.dumps(
            {
                "score_stage": "pre_sim",
                "composite_score": blended,
                "components": {
                    "predicted_quality": blended,
                    "heuristic_predicted_quality": 0.4,
                    "ml_positive_outcome_prob": probability,
                    "blended_predicted_quality": blended,
                    "meta_model_train_rows": 600,
                    "meta_model_positive_rows": 80,
                    "meta_model_used": 1.0,
                },
                "reason_codes": [],
            },
            sort_keys=True,
        ),
        created_at=f"2026-01-01T00:0{round_index}:00+00:00",
    )
