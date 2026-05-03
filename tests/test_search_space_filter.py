from __future__ import annotations

import json

import pytest

from core.config import load_config
from data.field_registry import FieldRegistry, FieldSpec
from domain.brain import BrainResultRecord
from domain.candidate import AlphaCandidate
import services.search_space_filter as search_space_filter
from services.search_space_filter import (
    _prior_multiplier_from_counts,
    _select_lane_field_pool,
    apply_operator_floor,
    build_search_space_filter_context,
    compute_field_floor,
    compute_field_weight,
    invalidate_winner_prior_cache,
    is_winner_result,
)
from storage.models import StageMetricRecord, SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository


@pytest.fixture(autouse=True)
def _clear_winner_prior_cache() -> None:
    invalidate_winner_prior_cache()
    yield
    invalidate_winner_prior_cache()


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
    assert context.field_multipliers["eur_metric"] == 0.3
    assert context.field_multipliers["unknown_profile"] == 0.3
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
    assert metrics["search_space_filter_field_floor_activation_count"] == 2


def test_floor_prevents_zero_trap() -> None:
    filter_config = load_config("config/dev.yaml").adaptive_generation.search_space_filter
    field = _field("high_catalog", score=0.80, runtime=True, region="", universe="")

    assert compute_field_weight(field, 0.15, filter_config) == 0.24


def test_hard_block_bypasses_floor() -> None:
    filter_config = load_config("config/dev.yaml").adaptive_generation.search_space_filter
    field = _field("schema_bad", score=0.80, runtime=True, region="", universe="")

    assert compute_field_weight(field, 0.15, filter_config, hard_blocked=True) == 0.0


def test_operator_floor_applies() -> None:
    filter_config = load_config("config/dev.yaml").adaptive_generation.search_space_filter

    assert apply_operator_floor(0.02, filter_config) == 0.05


def test_insufficient_data_returns_neutral() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _prior_guard_config()
        registry = _named_registry("thin_field", "close")
        _seed_run(repository, run_id="run")
        for index in range(10):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"thin-{index}",
                expression=f"ts_mean(thin_field,{index + 5})",
                fields=("thin_field",),
                operators=("ts_mean",),
                sharpe=1.0,
                fitness=0.5,
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

    assert "thin_field" not in context.field_multipliers


def test_laplace_smoothing_prevents_extreme_prior() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _prior_guard_config()
        registry = _named_registry("smooth_field", "support_field", "close")
        _seed_run(repository, run_id="run")
        for index in range(5):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"smooth-win-{index}",
                expression=f"ts_mean(smooth_field,{index + 5})",
                fields=("smooth_field",),
                operators=("ts_mean",),
                sharpe=1.0,
                fitness=0.5,
            )
        for index in range(10):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"support-loss-{index}",
                expression=f"ts_mean(support_field,{index + 5})",
                fields=("support_field",),
                operators=("ts_mean",),
                sharpe=0.0,
                fitness=0.0,
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

    assert 1.0 < context.field_multipliers["smooth_field"] < 1.5


def test_time_window_excludes_old_rounds() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _prior_guard_config()
        registry = _named_registry("window_field", "close")
        _seed_run(repository, run_id="run")
        for index in range(10):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"old-win-{index}",
                expression=f"ts_mean(window_field,{index + 5})",
                fields=("window_field",),
                operators=("ts_mean",),
                sharpe=1.0,
                fitness=0.5,
                round_index=12000 + index,
            )
        for index in range(2):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"recent-loss-{index}",
                expression=f"ts_mean(window_field,{index + 30})",
                fields=("window_field",),
                operators=("ts_mean",),
                sharpe=0.0,
                fitness=0.0,
                round_index=12760 + index,
            )

        context = build_search_space_filter_context(
            repository=repository,
            config=config,
            field_registry=registry,
            run_id="run",
            round_index=12770,
            blocked_fields=set(),
        )
    finally:
        repository.close()

    assert "window_field" not in context.field_multipliers


def test_alltime_fallback_with_dampen() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _prior_guard_config()
        registry = _named_registry("fallback_field", "close")
        _seed_run(repository, run_id="run")
        for index in range(30):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"fallback-win-{index}",
                expression=f"ts_mean(fallback_field,{index + 5})",
                fields=("fallback_field",),
                operators=("ts_mean",),
                sharpe=1.0,
                fitness=0.5,
                round_index=12000 + index,
            )

        context = build_search_space_filter_context(
            repository=repository,
            config=config,
            field_registry=registry,
            run_id="run",
            round_index=12770,
            blocked_fields=set(),
        )
    finally:
        repository.close()

    dampened = context.field_multipliers["fallback_field"]
    pure = _prior_multiplier_from_counts(
        win_count=30,
        loss_count=0,
        laplace_k=1.0,
        multiplier_min=0.5,
        multiplier_max=1.5,
    )
    assert 1.0 < dampened < pure


def test_winner_definition_explicit() -> None:
    filter_config = _prior_guard_config().adaptive_generation.search_space_filter

    assert is_winner_result(
        {"status": "completed", "sharpe": 0.6, "fitness": 0.2, "rejection_reason": ""},
        filter_config,
    ) is True
    assert is_winner_result(
        {"status": "completed", "sharpe": 0.4, "fitness": 0.2, "rejection_reason": ""},
        filter_config,
    ) is False
    assert is_winner_result(
        {"status": "rejected", "sharpe": 1.0, "fitness": 1.0, "rejection_reason": "check failed"},
        filter_config,
    ) is False


def test_prior_cache_hit_and_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _prior_guard_config()
        registry = _named_registry("cache_field", "close")
        _seed_run(repository, run_id="run")
        for index in range(15):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"cache-win-{index}",
                expression=f"ts_mean(cache_field,{index + 5})",
                fields=("cache_field",),
                operators=("ts_mean",),
                sharpe=1.0,
                fitness=0.5,
            )
        counting_connection = _CountingConnection(repository.connection)
        repository.connection = counting_connection
        now = [1000.0]
        monkeypatch.setattr(search_space_filter.time, "monotonic", lambda: now[0])

        for _ in range(2):
            build_search_space_filter_context(
                repository=repository,
                config=config,
                field_registry=registry,
                run_id="run",
                round_index=4,
                blocked_fields=set(),
            )
        assert counting_connection.prior_query_count == 1

        now[0] += config.adaptive_generation.search_space_filter.winner_prior_cache_ttl_seconds + 1
        build_search_space_filter_context(
            repository=repository,
            config=config,
            field_registry=registry,
            run_id="run",
            round_index=4,
            blocked_fields=set(),
        )
    finally:
        repository.close()

    assert counting_connection.prior_query_count == 2


def test_operational_timeouts_excluded_from_prior() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _prior_guard_config()
        registry = _named_registry("timeout_only_field", "support_field", "close")
        _seed_run(repository, run_id="run")
        for index in range(15):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"support-win-{index}",
                expression=f"ts_mean(support_field,{index + 5})",
                fields=("support_field",),
                operators=("ts_mean",),
                sharpe=1.0,
                fitness=0.5,
            )
        for index in range(5):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"operational-timeout-only-{index}",
                expression=f"ts_mean(timeout_only_field,{index + 5})",
                fields=("timeout_only_field",),
                operators=("ts_mean",),
                sharpe=None,
                fitness=None,
                status="timeout",
                rejection_reason="BRAIN 429 rate_limit",
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

    assert "timeout_only_field" not in context.field_multipliers
    assert context.to_metrics()["search_space_filter_timeout_cause_counts"] == {"operational": 5}


def test_prior_multiplier_bounded() -> None:
    assert _prior_multiplier_from_counts(
        win_count=100,
        loss_count=0,
        laplace_k=1.0,
        multiplier_min=0.5,
        multiplier_max=1.5,
    ) <= 1.5
    assert _prior_multiplier_from_counts(
        win_count=0,
        loss_count=100,
        laplace_k=1.0,
        multiplier_min=0.5,
        multiplier_max=1.5,
    ) >= 0.5


def test_exploration_budget_includes_high_catalog_score_fields() -> None:
    filter_config = load_config("config/dev.yaml").adaptive_generation.search_space_filter
    filter_config.exploration_budget_pct = 0.50
    adjusted_fields = [
        _field("high_catalog_low_adjusted", score=0.0, runtime=True, region="", universe=""),
        _field("low_catalog_high_adjusted", score=1.0, runtime=True, region="", universe=""),
        _field("middle_catalog", score=0.9, runtime=True, region="", universe=""),
    ]
    catalog_registry = FieldRegistry(
        fields={
            "high_catalog_low_adjusted": _field(
                "high_catalog_low_adjusted",
                score=0.85,
                runtime=True,
                region="",
                universe="",
            ),
            "low_catalog_high_adjusted": _field(
                "low_catalog_high_adjusted",
                score=0.40,
                runtime=True,
                region="",
                universe="",
            ),
            "middle_catalog": _field("middle_catalog", score=0.60, runtime=True, region="", universe=""),
        }
    )

    pool = _select_lane_field_pool(
        adjusted_fields,
        target_count=2,
        filter_config=filter_config,
        catalog_field_registry=catalog_registry,
    )

    assert "high_catalog_low_adjusted" in pool
    assert "low_catalog_high_adjusted" in pool


def test_floor_ratio_configurable() -> None:
    filter_config = load_config("config/dev.yaml").adaptive_generation.search_space_filter
    filter_config.field_floor_ratio = 0.50
    field = _field("high_catalog", score=0.80, runtime=True, region="", universe="")

    assert compute_field_floor(field, filter_config) == 0.40


def test_no_floor_below_absolute_min() -> None:
    filter_config = load_config("config/dev.yaml").adaptive_generation.search_space_filter
    low_score_field = _field("low_catalog", score=0.20, runtime=True, region="", universe="")
    medium_score_field = _field("medium_catalog", score=0.50, runtime=True, region="", universe="")

    assert compute_field_floor(low_score_field, filter_config) == 0.10
    assert compute_field_floor(medium_score_field, filter_config) == 0.15


def test_existing_behavior_unchanged_when_above_floor() -> None:
    filter_config = load_config("config/dev.yaml").adaptive_generation.search_space_filter
    field = _field("healthy_raw", score=0.80, runtime=True, region="", universe="")

    assert round(compute_field_weight(field, 0.75, filter_config), 6) == 0.60


def test_search_space_filter_winner_prior_promotes_clean_completed_history() -> None:
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
        search_filter.winner_prior_min_completed = 8
        search_filter.winner_prior_min_winners_for_boost = 2
        search_filter.winner_prior_min_losers_for_penalty = 3
        search_filter.winner_prior_min_sharpe = 0.30
        search_filter.winner_prior_min_fitness = 0.10
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

    assert context.field_multipliers["strong_field"] == 1.25
    assert context.field_multipliers["pass_field"] == 1.25
    assert "weak_field" not in context.field_multipliers
    assert "reject_field" not in context.field_multipliers
    assert "timeout_field" not in context.field_multipliers
    assert context.operator_multipliers["ts_mean"] == 1.25
    assert context.operator_multipliers["ts_scale"] == 1.25
    assert "ts_delta" not in context.operator_multipliers
    assert "ts_rank" not in context.operator_multipliers
    assert "ts_corr" not in context.operator_multipliers

    metrics = context.to_metrics()
    assert metrics["search_space_filter_promoted_field_count"] == 2
    assert metrics["search_space_filter_demoted_field_count"] == 0
    assert metrics["search_space_filter_promoted_operator_count"] == 2
    assert metrics["search_space_filter_demoted_operator_count"] == 0
    assert {item["name"] for item in metrics["search_space_filter_top_promoted_fields"]} >= {
        "strong_field",
        "pass_field",
    }


def test_operational_timeout_not_penalize_field() -> None:
    repository = SQLiteRepository(":memory:")
    field_name = "operational_timeout_field"
    try:
        config = _timeout_prior_config()
        registry = _timeout_prior_registry(field_name)
        _seed_run(repository, run_id="run")
        for index in range(5):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"operational-timeout-{index}",
                expression=f"ts_mean({field_name},{index + 20})",
                fields=(field_name,),
                operators=("ts_mean",),
                sharpe=None,
                fitness=None,
                status="timeout",
                rejection_reason="BRAIN 429 rate_limit",
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

    assert field_name not in context.field_multipliers
    metrics = context.to_metrics()
    assert metrics["search_space_filter_timeout_cause_counts"] == {"operational": 5}
    assert metrics["search_space_filter_timeout_prior_demoted_field_counts"] == {}


def test_quality_timeout_penalizes_field() -> None:
    repository = SQLiteRepository(":memory:")
    field_name = "quality_timeout_field"
    try:
        config = _timeout_prior_config()
        registry = _timeout_prior_registry(field_name)
        _seed_run(repository, run_id="run")
        for index in range(5):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id="quality-timeout-alpha",
                batch_id=f"batch-quality-timeout-{index}",
                job_id=f"job-quality-timeout-{index}",
                expression=f"ts_mean({field_name},{index + 5})",
                fields=(field_name,),
                operators=("ts_mean",),
                sharpe=None,
                fitness=None,
                status="timeout",
                rejection_reason="poll_timeout_live",
            )
        for index in range(5):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"quality-neutral-{index}",
                expression=f"ts_mean({field_name},{index + 20})",
                fields=(field_name,),
                operators=("ts_mean",),
                sharpe=0.0,
                fitness=0.0,
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

    assert context.field_multipliers[field_name] == pytest.approx(0.583333)
    metrics = context.to_metrics()
    assert metrics["search_space_filter_timeout_cause_counts"] == {"quality": 5}
    assert metrics["search_space_filter_timeout_prior_demoted_field_counts"] == {field_name: 5}


def test_mixed_timeout_only_quality_counts() -> None:
    repository = SQLiteRepository(":memory:")
    field_name = "mixed_timeout_field"
    try:
        config = _timeout_prior_config()
        registry = _timeout_prior_registry(field_name)
        _seed_run(repository, run_id="run")
        for index in range(3):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"mixed-operational-{index}",
                expression=f"ts_mean({field_name},{index + 5})",
                fields=(field_name,),
                operators=("ts_mean",),
                sharpe=None,
                fitness=None,
                status="timeout",
                rejection_reason="persona auth throttle",
            )
        for index in range(2):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id="mixed-quality-alpha",
                batch_id=f"batch-mixed-quality-{index}",
                job_id=f"job-mixed-quality-{index}",
                expression=f"ts_mean({field_name},{index + 10})",
                fields=(field_name,),
                operators=("ts_mean",),
                sharpe=None,
                fitness=None,
                status="timeout",
                rejection_reason="poll_timeout_live",
            )
        for index in range(8):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"mixed-neutral-{index}",
                expression=f"ts_mean({field_name},{index + 20})",
                fields=(field_name,),
                operators=("ts_mean",),
                sharpe=0.0,
                fitness=0.0,
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

    assert context.field_multipliers[field_name] == pytest.approx(0.583333)
    metrics = context.to_metrics()
    assert metrics["search_space_filter_timeout_cause_counts"] == {"operational": 3, "quality": 2}
    assert metrics["search_space_filter_timeout_prior_demoted_field_counts"] == {field_name: 2}


def test_min_support_guard_returns_neutral() -> None:
    repository = SQLiteRepository(":memory:")
    field_name = "small_timeout_field"
    try:
        config = _timeout_prior_config()
        registry = _timeout_prior_registry(field_name)
        _seed_run(repository, run_id="run")
        for index in range(5):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id="small-quality-alpha",
                batch_id=f"batch-small-quality-{index}",
                job_id=f"job-small-quality-{index}",
                expression=f"ts_mean({field_name},{index + 5})",
                fields=(field_name,),
                operators=("ts_mean",),
                sharpe=None,
                fitness=None,
                status="timeout",
                rejection_reason="poll_timeout_live",
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

    assert field_name not in context.field_multipliers
    metrics = context.to_metrics()
    assert metrics["search_space_filter_timeout_cause_counts"] == {"quality": 5}
    assert metrics["search_space_filter_timeout_prior_demoted_field_counts"] == {}


def test_search_space_filter_promotes_near_miss_with_robustness_checks() -> None:
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
        search_filter.winner_prior_min_completed = 8
        search_filter.check_penalty_min_support = 2
        registry = FieldRegistry(
            fields={
                "check_bad": _field("check_bad", score=1.0, runtime=True, region="", universe=""),
                "close": _field("close", score=1.0, runtime=True, region="", universe=""),
            }
        )
        _seed_run(repository, run_id="run")
        for index in range(8):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"check-bad-{index}",
                expression=f"ts_mean(check_bad,{index + 5})",
                fields=("check_bad",),
                operators=("ts_mean",),
                sharpe=1.20,
                fitness=0.80,
                hard_fail_checks=("LOW_2Y_SHARPE",),
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

    assert context.field_multipliers["check_bad"] == pytest.approx(1.19)
    assert context.operator_multipliers["ts_mean"] == 1.4

    metrics = context.to_metrics()
    assert metrics["search_space_filter_promoted_field_count"] == 1
    assert metrics["search_space_filter_demoted_field_count"] == 0
    assert metrics["search_space_filter_field_hard_fail_penalty_counts"] == {}
    assert metrics["search_space_filter_operator_hard_fail_penalty_counts"] == {}
    assert metrics["search_space_filter_field_robustness_penalty_counts"] == {"check_bad": 8}


def test_search_space_filter_demotes_repeated_structural_risk_checks_by_field_only() -> None:
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
        search_filter.check_penalty_min_support = 2
        registry = FieldRegistry(
            fields={
                "check_bad": _field("check_bad", score=1.0, runtime=True, region="", universe=""),
                "close": _field("close", score=1.0, runtime=True, region="", universe=""),
            }
        )
        _seed_run(repository, run_id="run")
        for index in range(2):
            _seed_prior_candidate_results(
                repository,
                run_id="run",
                alpha_id=f"concentrated-{index}",
                expression=f"ts_mean(check_bad,{index + 5})",
                fields=("check_bad",),
                operators=("ts_mean",),
                sharpe=1.20,
                fitness=0.80,
                hard_fail_checks=("CONCENTRATED_WEIGHT",),
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

    assert context.field_multipliers["check_bad"] == 0.35
    assert "ts_mean" not in context.operator_multipliers

    metrics = context.to_metrics()
    assert metrics["search_space_filter_promoted_field_count"] == 0
    assert metrics["search_space_filter_demoted_field_count"] == 0
    assert metrics["search_space_filter_field_hard_fail_penalty_counts"] == {"check_bad": 2}
    assert metrics["search_space_filter_operator_hard_fail_penalty_counts"] == {}


def test_search_space_filter_applies_field_diagnostic_coverage_penalty() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        config.brain.region = "USA"
        config.brain.universe = "TOP3000"
        config.brain.delay = 1
        config.generation.allowed_fields = []
        config.generation.allow_catalog_fields_without_runtime = True
        search_filter = config.adaptive_generation.search_space_filter
        search_filter.enabled = True
        _seed_run(repository, run_id="run")
        repository.connection.execute(
            """
            INSERT INTO field_diagnostics
            (diagnostic_id, run_id, field_name, region, universe, delay, neutralization, decay,
             diagnostic_name, params_json, expression, status, coverage_ratio, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "diag-low-coverage",
                "run",
                "diag_field",
                "USA",
                "TOP3000",
                1,
                "NONE",
                0,
                "raw",
                "{}",
                "diag_field",
                "completed",
                0.01,
                "2026-04-27T00:00:00+00:00",
                "2026-04-27T00:00:00+00:00",
            ),
        )
        repository.connection.commit()
        registry = FieldRegistry(
            fields={
                "diag_field": _field("diag_field", score=1.0, runtime=True, region="", universe=""),
                "close": _field("close", score=1.0, runtime=True, region="", universe=""),
            }
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

    assert context.field_multipliers["diag_field"] == 0.3
    assert context.to_metrics()["search_space_filter_diagnostic_field_multiplier_count"] == 1


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


class _CountingConnection:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.prior_query_count = 0

    def execute(self, sql, parameters=()):
        if "WITH latest_runtime" in str(sql) and "FROM brain_results r" in str(sql):
            self.prior_query_count += 1
        return self._delegate.execute(sql, parameters)

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


def _prior_guard_config():
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
    search_filter.winner_prior_lookback_rounds = 50
    search_filter.winner_prior_min_completed = 15
    search_filter.winner_prior_min_winners_for_boost = 3
    search_filter.winner_prior_min_losers_for_penalty = 3
    search_filter.winner_prior_laplace_k = 1.0
    search_filter.winner_prior_multiplier_max = 1.5
    search_filter.winner_prior_multiplier_min = 0.5
    search_filter.winner_prior_alltime_dampen = 0.5
    search_filter.winner_prior_cache_ttl_seconds = 300
    search_filter.winner_prior_min_sharpe = 0.5
    search_filter.winner_prior_min_fitness = 0.0
    return config


def _named_registry(*field_names: str) -> FieldRegistry:
    return FieldRegistry(
        fields={
            name: _field(name, score=1.0, runtime=True, region="", universe="")
            for name in field_names
        }
    )


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


def _timeout_prior_registry(field_name: str) -> FieldRegistry:
    return FieldRegistry(
        fields={
            field_name: _field(field_name, score=1.0, runtime=True, region="", universe=""),
            "close": _field("close", score=1.0, runtime=True, region="", universe=""),
        }
    )


def _timeout_prior_config():
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
    search_filter.winner_prior_min_completed = 10
    search_filter.winner_prior_min_losers_for_penalty = 2
    return config


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
    hard_fail_checks: tuple[str, ...] = (),
    blocking_warning_checks: tuple[str, ...] = (),
    batch_id: str | None = None,
    job_id: str | None = None,
    round_index: int = 2,
    created_at: str | None = None,
) -> None:
    resolved_batch_id = batch_id or f"batch-{alpha_id}"
    resolved_job_id = job_id or f"job-{alpha_id}"
    resolved_created_at = created_at or "2026-04-27T00:01:00+00:00"
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
            batch_id=resolved_batch_id,
            run_id=run_id,
            round_index=round_index,
            backend="api",
            status="completed",
            candidate_count=1,
            sim_config_snapshot="{}",
            export_path=None,
            notes_json="{}",
            created_at=resolved_created_at,
            updated_at=resolved_created_at,
        )
    )
    repository.submissions.upsert_submissions(
        [
            SubmissionRecord(
                job_id=resolved_job_id,
                batch_id=resolved_batch_id,
                run_id=run_id,
                round_index=round_index,
                candidate_id=alpha_id,
                expression=expression,
                backend="api",
                status=status,
                sim_config_snapshot="{}",
                submitted_at=resolved_created_at,
                updated_at=resolved_created_at,
                completed_at=resolved_created_at,
                export_path=None,
                raw_submission_json="{}",
                error_message=rejection_reason,
            )
        ]
    )
    repository.brain_results.save_results(
        [
            BrainResultRecord(
                job_id=resolved_job_id,
                run_id=run_id,
                round_index=round_index,
                batch_id=resolved_batch_id,
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
                simulated_at=resolved_created_at,
                created_at=resolved_created_at,
                check_summary_json=json.dumps(
                    {
                        "hard_fail_checks": list(hard_fail_checks),
                        "warning_checks": list(blocking_warning_checks),
                        "blocking_warning_checks": list(blocking_warning_checks),
                    },
                    sort_keys=True,
                ),
                hard_fail_checks_json=json.dumps(list(hard_fail_checks), sort_keys=True),
                blocking_warning_checks_json=json.dumps(list(blocking_warning_checks), sort_keys=True),
            )
        ]
    )
