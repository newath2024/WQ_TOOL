from __future__ import annotations

import json

from core.config import load_config
from data.field_registry import FieldRegistry, FieldSpec
from domain.brain import BrainResultRecord
from domain.candidate import AlphaCandidate
from services.search_space_filter import build_search_space_filter_context
from storage.models import StageMetricRecord, SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository


def _field(
    name: str,
    *,
    score: float,
    runtime: bool = False,
    region: str = "USA",
    universe: str = "TOP3000",
    field_type: str = "matrix",
) -> FieldSpec:
    return FieldSpec(
        name=name,
        dataset="test",
        field_type=field_type,
        coverage=1.0,
        alpha_usage_count=1,
        category="group" if field_type == "vector" else "analyst",
        region=region,
        universe=universe,
        runtime_available=runtime,
        field_score=score,
        category_weight=1.0,
    )


def _registry() -> FieldRegistry:
    return FieldRegistry(
        fields={
            "close": _field("close", score=1.0, runtime=True, region="", universe=""),
            "anl4_eps": _field("anl4_eps", score=0.9, region="GLB", universe="TOP3000"),
            "eur_metric": _field("eur_metric", score=0.8, region="EUR", universe="TOP3000"),
            "unknown_profile": _field("unknown_profile", score=0.7, region="", universe=""),
            "bad_field": _field("bad_field", score=0.6, region="GLB", universe="TOP3000"),
            "sector": _field("sector", score=1.0, runtime=True, region="", universe="", field_type="vector"),
        }
    )


def test_search_space_filter_scores_fields_and_builds_lane_profiles() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        config.brain.region = "USA"
        config.brain.universe = "TOP3000"
        config.generation.allowed_fields = []
        config.generation.allowed_operators = ["rank", "zscore", "ts_delta"]
        config.generation.allow_catalog_fields_without_runtime = True
        search_filter = config.adaptive_generation.search_space_filter
        search_filter.enabled = True
        search_filter.lane_field_caps = {"fresh": 2}
        search_filter.lane_operator_allowlists = {"quality_polish": ["rank"]}

        repository.save_stage_metrics(
            [
                StageMetricRecord(
                    run_id="run",
                    round_index=3,
                    stage="generation",
                    metrics_json=json.dumps({"validation_disallowed_field_counts": {"unknown_profile": 2}}),
                    created_at="2026-04-27T00:00:00+00:00",
                )
            ]
        )

        context = build_search_space_filter_context(
            repository=repository,
            config=config,
            field_registry=_registry(),
            run_id="run",
            round_index=4,
            blocked_fields={"bad_field"},
        )
    finally:
        repository.close()

    assert context.enabled is True
    assert not context.field_registry.contains("bad_field")
    assert context.field_registry.contains("anl4_eps")
    assert context.field_registry.contains("eur_metric")
    assert context.field_multipliers["eur_metric"] == 0.25
    assert context.field_multipliers["unknown_profile"] == 0.2625
    assert context.lane_field_pools["fresh"] == {"close", "anl4_eps"}
    assert context.field_registry_for_lane("fresh").contains("sector")

    scoped = context.generation_config_for_lane(config.generation, "quality_polish")
    assert scoped.allowed_operators == ["rank"]
    assert context.expression_allowed_for_lane("rank(close)", "quality_polish") is True
    assert context.expression_allowed_for_lane("zscore(close)", "quality_polish") is False
    assert context.expression_allowed_for_lane("rank(bad_field)", "quality_polish") is False

    metrics = context.to_metrics()
    assert metrics["search_space_filter_active_field_count"] == 5
    assert metrics["search_space_filter_hard_blocked_field_count"] == 1
    assert metrics["search_space_filter_soft_penalized_field_count"] == 2
    assert metrics["search_space_filter_lane_field_pool_counts"] == {"fresh": 2}


def test_search_space_filter_winner_prior_promotes_and_demotes_completed_history() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        config.brain.region = "USA"
        config.brain.universe = "TOP3000"
        config.generation.allowed_fields = []
        config.generation.allow_catalog_fields_without_runtime = True
        search_filter = config.adaptive_generation.search_space_filter
        search_filter.enabled = True
        search_filter.field_result_multiplier = 1.0
        search_filter.operator_result_multiplier = 1.0
        search_filter.winner_prior_enabled = True
        search_filter.winner_prior_min_support = 2
        registry = _prior_registry()
        _seed_run(repository, run_id="run")
        _seed_prior_candidate_results(
            repository,
            run_id="run",
            alpha_id="strong-1",
            expression="ts_mean(strong_field,10)",
            fields=("strong_field",),
            operators=("ts_mean",),
            sharpe=0.60,
            fitness=0.35,
        )
        _seed_prior_candidate_results(
            repository,
            run_id="run",
            alpha_id="strong-2",
            expression="ts_mean(strong_field,20)",
            fields=("strong_field",),
            operators=("ts_mean",),
            sharpe=0.55,
            fitness=0.31,
        )
        _seed_prior_candidate_results(
            repository,
            run_id="run",
            alpha_id="pass-1",
            expression="ts_scale(pass_field,10)",
            fields=("pass_field",),
            operators=("ts_scale",),
            sharpe=0.35,
            fitness=0.12,
        )
        _seed_prior_candidate_results(
            repository,
            run_id="run",
            alpha_id="pass-2",
            expression="ts_scale(pass_field,20)",
            fields=("pass_field",),
            operators=("ts_scale",),
            sharpe=0.40,
            fitness=0.15,
        )
        _seed_prior_candidate_results(
            repository,
            run_id="run",
            alpha_id="weak-1",
            expression="ts_delta(weak_field,10)",
            fields=("weak_field",),
            operators=("ts_delta",),
            sharpe=0.10,
            fitness=0.02,
        )
        _seed_prior_candidate_results(
            repository,
            run_id="run",
            alpha_id="weak-2",
            expression="ts_delta(weak_field,20)",
            fields=("weak_field",),
            operators=("ts_delta",),
            sharpe=0.05,
            fitness=0.01,
        )
        _seed_prior_candidate_results(
            repository,
            run_id="run",
            alpha_id="reject-1",
            expression="ts_rank(reject_field,10)",
            fields=("reject_field",),
            operators=("ts_rank",),
            sharpe=0.80,
            fitness=0.50,
            rejection_reason="Alpha expression includes a reversion component.",
        )
        _seed_prior_candidate_results(
            repository,
            run_id="run",
            alpha_id="reject-2",
            expression="ts_rank(reject_field,20)",
            fields=("reject_field",),
            operators=("ts_rank",),
            sharpe=0.70,
            fitness=0.40,
            rejection_reason="Alpha expression includes a reversion component.",
        )
        _seed_prior_candidate_results(
            repository,
            run_id="run",
            alpha_id="timeout-1",
            expression="ts_corr(timeout_field,close,10)",
            fields=("timeout_field",),
            operators=("ts_corr",),
            sharpe=None,
            fitness=None,
            status="timeout",
            rejection_reason="poll_timeout_after_downtime",
        )
        _seed_prior_candidate_results(
            repository,
            run_id="run",
            alpha_id="timeout-2",
            expression="ts_corr(timeout_field,close,20)",
            fields=("timeout_field",),
            operators=("ts_corr",),
            sharpe=None,
            fitness=None,
            status="timeout",
            rejection_reason="Persona verification timeout",
        )

        context = build_search_space_filter_context(
            repository=repository,
            config=config,
            field_registry=registry,
            run_id="run",
            round_index=4,
            blocked_fields=set(),
        )
    finally:
        repository.close()

    assert context.field_multipliers["strong_field"] == 1.8
    assert context.field_multipliers["pass_field"] == 1.35
    assert context.field_multipliers["weak_field"] == 0.65
    assert context.field_multipliers["reject_field"] == 0.65
    assert "timeout_field" not in context.field_multipliers
    assert context.operator_multipliers["ts_mean"] == 1.8
    assert context.operator_multipliers["ts_scale"] == 1.35
    assert context.operator_multipliers["ts_delta"] == 0.65
    assert "ts_corr" not in context.operator_multipliers

    metrics = context.to_metrics()
    assert metrics["search_space_filter_promoted_field_count"] == 2
    assert metrics["search_space_filter_demoted_field_count"] == 2
    assert metrics["search_space_filter_promoted_operator_count"] == 2
    assert metrics["search_space_filter_demoted_operator_count"] == 2
    assert metrics["search_space_filter_top_promoted_fields"][0]["name"] == "strong_field"


def test_search_space_filter_lane_pool_uses_configured_cap_and_minimum() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        config.generation.allowed_fields = []
        config.generation.allow_catalog_fields_without_runtime = True
        search_filter = config.adaptive_generation.search_space_filter
        search_filter.enabled = True
        search_filter.lane_field_caps = {"fresh": 5}
        search_filter.lane_field_min_count = 30

        context = build_search_space_filter_context(
            repository=repository,
            config=config,
            field_registry=_many_field_registry(40),
            run_id="run",
            round_index=1,
            blocked_fields=set(),
        )
    finally:
        repository.close()

    assert len(context.lane_field_pools["fresh"]) == 30
    assert context.to_metrics()["search_space_filter_lane_field_pool_before_counts"] == {"fresh": 40}
    assert context.to_metrics()["search_space_filter_lane_field_pool_after_counts"] == {"fresh": 30}


def _prior_registry() -> FieldRegistry:
    return FieldRegistry(
        fields={
            name: _field(name, score=1.0, runtime=True, region="", universe="")
            for name in (
                "strong_field",
                "pass_field",
                "weak_field",
                "reject_field",
                "timeout_field",
                "close",
            )
        }
    )


def _many_field_registry(count: int) -> FieldRegistry:
    return FieldRegistry(
        fields={
            f"field_{index:02d}": _field(
                f"field_{index:02d}",
                score=float(count - index),
                runtime=True,
                region="",
                universe="",
            )
            for index in range(count)
        }
    )


def _seed_run(repository: SQLiteRepository, *, run_id: str) -> None:
    repository.upsert_run(
        run_id=run_id,
        seed=7,
        config_path="config/dev.yaml",
        config_snapshot="{}",
        status="running",
        started_at="2026-04-27T00:00:00+00:00",
    )


def _seed_prior_candidate_results(
    repository: SQLiteRepository,
    *,
    run_id: str,
    alpha_id: str,
    expression: str,
    fields: tuple[str, ...],
    operators: tuple[str, ...],
    sharpe: float | None,
    fitness: float | None,
    status: str = "completed",
    rejection_reason: str | None = None,
) -> None:
    batch_id = f"batch-{alpha_id}"
    created_at = "2026-04-27T00:01:00+00:00"
    repository.save_alpha_candidates(
        run_id,
        [
            AlphaCandidate(
                alpha_id=alpha_id,
                expression=expression,
                normalized_expression=expression,
                generation_mode="guided_explore",
                parent_ids=(),
                complexity=4,
                created_at="2026-04-27T00:00:00+00:00",
                fields_used=fields,
                operators_used=operators,
                depth=3,
                generation_metadata={},
            )
        ],
    )
    repository.submissions.upsert_batch(
        SubmissionBatchRecord(
            batch_id=batch_id,
            run_id=run_id,
            round_index=2,
            backend="api",
            status="completed",
            candidate_count=1,
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
                job_id=f"job-{alpha_id}",
                batch_id=batch_id,
                run_id=run_id,
                round_index=2,
                candidate_id=alpha_id,
                expression=expression,
                backend="api",
                status=status,
                sim_config_snapshot="{}",
                submitted_at=created_at,
                updated_at=created_at,
                completed_at=created_at,
                export_path=None,
                raw_submission_json="{}",
                error_message=rejection_reason,
            )
        ]
    )
    repository.brain_results.save_results(
        [
            BrainResultRecord(
                job_id=f"job-{alpha_id}",
                run_id=run_id,
                round_index=2,
                batch_id=batch_id,
                candidate_id=alpha_id,
                expression=expression,
                status=status,
                region="USA",
                universe="TOP3000",
                delay=1,
                neutralization="SUBINDUSTRY",
                decay=5,
                sharpe=sharpe,
                fitness=fitness,
                turnover=0.10 if status == "completed" else None,
                drawdown=0.10 if status == "completed" else None,
                returns=0.01 if status == "completed" else None,
                margin=0.01 if status == "completed" else None,
                submission_eligible=None,
                rejection_reason=rejection_reason,
                raw_result_json="{}",
                metric_source="external_brain",
                simulated_at=created_at,
                created_at=created_at,
            )
        ]
    )
