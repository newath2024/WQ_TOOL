from __future__ import annotations

from dataclasses import replace

from core.config import BrainRobustnessProxyConfig, PreSimSelectionWeightsConfig, SelectionConfig
from data.field_registry import FieldRegistry, FieldSpec
from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemorySnapshot
from services.candidate_selection_service import CandidateSelectionService
from services.models import CrowdingScore, DedupBatchResult, DedupDecision, SimulationResult
from services.selection_service import SelectionService
from storage.models import BrainResultRecord, SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository


def test_pre_sim_score_prefers_lower_duplicate_and_crowding_risk() -> None:
    service = SelectionService(config=SelectionConfig())
    field_registry = _field_registry()
    snapshot = PatternMemorySnapshot(regime_key="regime")
    candidates = [
        _candidate("alpha-risky", "rank(ts_mean(close, 5))", field_name="close"),
        _candidate("alpha-clean", "rank(ts_mean(volume, 5))", field_name="volume"),
    ]
    dedup_result = DedupBatchResult(
        kept_candidates=tuple(candidates),
        blocked_candidates=(),
        decisions=(
            DedupDecision(
                alpha_id="alpha-risky",
                normalized_expression=candidates[0].normalized_expression,
                stage="pre_sim",
                decision="kept",
                reason_code="unique",
                metrics={"duplicate_risk": 0.9},
            ),
            DedupDecision(
                alpha_id="alpha-clean",
                normalized_expression=candidates[1].normalized_expression,
                stage="pre_sim",
                decision="kept",
                reason_code="unique",
                metrics={"duplicate_risk": 0.0},
            ),
        ),
    )
    crowding_scores = {
        "alpha-risky": CrowdingScore(alpha_id="alpha-risky", stage="pre_sim", total_penalty=0.8),
        "alpha-clean": CrowdingScore(alpha_id="alpha-clean", stage="pre_sim", total_penalty=0.1),
    }

    scored = service.score_pre_sim_candidates(
        candidates,
        snapshot=snapshot,
        field_registry=field_registry,
        min_pattern_support=1,
        dedup_result=dedup_result,
        crowding_scores=crowding_scores,
    )

    assert scored[0].candidate.alpha_id == "alpha-clean"
    assert scored[0].composite_score > scored[1].composite_score


def test_pre_sim_score_applies_quality_polish_prior_without_bypassing_penalties() -> None:
    service = SelectionService(config=SelectionConfig())
    field_registry = _field_registry()
    snapshot = PatternMemorySnapshot(regime_key="regime")
    plain = _candidate("alpha-plain", "rank(ts_mean(close, 5))", field_name="close")
    polish = replace(
        _candidate("alpha-polish", "zscore(ts_mean(close, 5))", field_name="close"),
        generation_mode="quality_polish",
        generation_metadata={"quality_polish_prior": 0.10},
    )

    scored = service.score_pre_sim_candidates(
        [plain, polish],
        snapshot=snapshot,
        field_registry=field_registry,
        min_pattern_support=1,
    )

    assert scored[0].candidate.alpha_id == "alpha-polish"
    assert scored[0].ranking_rationale["selection_breakdown"]["components"]["quality_polish_prior"] == 0.10
    assert "quality_polish_candidate" in scored[0].reason_codes


def test_pre_sim_score_applies_recipe_bucket_prior() -> None:
    service = SelectionService(config=SelectionConfig())
    field_registry = _field_registry()
    snapshot = PatternMemorySnapshot(regime_key="regime")
    plain = _candidate("alpha-plain", "rank(ts_mean(close, 5))", field_name="close")
    recipe = replace(
        _candidate("alpha-recipe", "rank(ts_mean(volume, 5))", field_name="volume"),
        generation_mode="recipe_guided",
        generation_metadata={
            "recipe_bucket_prior": 0.096,
            "search_bucket_id": "fundamental_quality|fundamental|balanced",
        },
    )

    scored = service.score_pre_sim_candidates(
        [plain, recipe],
        snapshot=snapshot,
        field_registry=field_registry,
        min_pattern_support=1,
    )

    assert scored[0].candidate.alpha_id == "alpha-recipe"
    assert scored[0].ranking_rationale["selection_breakdown"]["components"]["recipe_bucket_prior"] == 0.096
    assert "recipe_guided_candidate" in scored[0].reason_codes


def test_pre_sim_score_applies_family_proxy_penalty_from_recent_history() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        repository.upsert_run(
            run_id="run-proxy",
            seed=7,
            config_path="config/dev.yaml",
            config_snapshot="{}",
            status="running",
            started_at="2026-04-23T00:00:00+00:00",
        )
        repository.save_alpha_candidates(
            "run-proxy",
            [
                _candidate(
                    "hist-hot",
                    "rank(ts_mean(close, 5))",
                    field_name="close",
                    metadata={"family_signature": "family-hot"},
                )
            ],
        )
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-hist",
                run_id="run-proxy",
                round_index=1,
                backend="api",
                status="completed",
                candidate_count=1,
                sim_config_snapshot="{}",
                export_path=None,
                notes_json="{}",
                created_at="2026-04-23T00:00:00+00:00",
                updated_at="2026-04-23T00:00:10+00:00",
            )
        )
        repository.submissions.upsert_submissions(
            [
                SubmissionRecord(
                    job_id="job-hist",
                    batch_id="batch-hist",
                    run_id="run-proxy",
                    round_index=1,
                    candidate_id="hist-hot",
                    expression="rank(ts_mean(close, 5))",
                    backend="api",
                    status="completed",
                    sim_config_snapshot="{}",
                    submitted_at="2026-04-23T00:00:00+00:00",
                    updated_at="2026-04-23T00:00:10+00:00",
                    completed_at="2026-04-23T00:00:10+00:00",
                    export_path=None,
                    raw_submission_json="{}",
                    error_message=None,
                )
            ]
        )
        repository.brain_results.save_results(
            [
                BrainResultRecord(
                    job_id="job-hist",
                    run_id="run-proxy",
                    round_index=1,
                    batch_id="batch-hist",
                    candidate_id="hist-hot",
                    expression="rank(ts_mean(close, 5))",
                    status="completed",
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                    neutralization="SECTOR",
                    decay=0,
                    sharpe=-0.20,
                    fitness=-0.10,
                    turnover=0.80,
                    drawdown=0.40,
                    returns=-0.02,
                    margin=0.01,
                    submission_eligible=False,
                    rejection_reason=None,
                    raw_result_json="{}",
                    metric_source="external_brain",
                    simulated_at="2026-04-23T00:10:00+00:00",
                    created_at="2026-04-23T00:10:00+00:00",
                )
            ]
        )

        service = SelectionService(
            config=SelectionConfig(),
            repository=repository,
            family_proxy_lookback_rounds=12,
            family_proxy_min_support=1,
        )
        field_registry = _field_registry()
        snapshot = PatternMemorySnapshot(regime_key="regime")
        hot = _candidate(
            "alpha-hot",
            "rank(ts_mean(close, 6))",
            field_name="close",
            metadata={
                "family_signature": "family-hot",
                "parent_refs": [{"family_signature": "family-hot"}],
            },
        )
        clean = _candidate(
            "alpha-clean",
            "rank(ts_mean(volume, 6))",
            field_name="volume",
            metadata={"family_signature": "family-clean"},
        )

        scored = service.score_pre_sim_candidates(
            [hot, clean],
            snapshot=snapshot,
            field_registry=field_registry,
            min_pattern_support=1,
            run_id="run-proxy",
            round_index=2,
        )
    finally:
        repository.close()

    breakdowns = {
        item.candidate.alpha_id: item.ranking_rationale["selection_breakdown"]["components"]
        for item in scored
    }
    assert breakdowns["alpha-hot"]["family_correlation_proxy_penalty"] > breakdowns["alpha-clean"]["family_correlation_proxy_penalty"]
    assert "family_proxy_penalty_applied" in next(item for item in scored if item.candidate.alpha_id == "alpha-hot").reason_codes


def test_pre_sim_score_applies_brain_robustness_proxy_penalty_from_recent_history() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        repository.upsert_run(
            run_id="run-brain-proxy",
            seed=7,
            config_path="config/dev.yaml",
            config_snapshot="{}",
            status="running",
            started_at="2026-04-23T00:00:00+00:00",
        )
        history_candidate = _candidate(
            "hist-weak",
            "rank(ts_mean(close, 5))",
            field_name="close",
            metadata={
                "generation_source": "fresh",
                "family_signature": "family-weak",
                "search_bucket_id": "bucket-weak",
            },
        )
        repository.save_alpha_candidates("run-brain-proxy", [history_candidate])
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-hist",
                run_id="run-brain-proxy",
                round_index=1,
                backend="api",
                status="completed",
                candidate_count=5,
                sim_config_snapshot="{}",
                export_path=None,
                notes_json="{}",
                created_at="2026-04-23T00:00:00+00:00",
                updated_at="2026-04-23T00:05:00+00:00",
            )
        )
        repository.submissions.upsert_submissions(
            [
                SubmissionRecord(
                    job_id=f"job-hist-{index}",
                    batch_id="batch-hist",
                    run_id="run-brain-proxy",
                    round_index=index + 1,
                    candidate_id="hist-weak",
                    expression=history_candidate.expression,
                    backend="api",
                    status="completed",
                    sim_config_snapshot="{}",
                    submitted_at=f"2026-04-23T00:0{index}:00+00:00",
                    updated_at=f"2026-04-23T00:0{index}:30+00:00",
                    completed_at=f"2026-04-23T00:0{index}:30+00:00",
                    export_path=None,
                    raw_submission_json="{}",
                    error_message=None,
                )
                for index in range(5)
            ]
        )
        repository.brain_results.save_results(
            [
                BrainResultRecord(
                    job_id=f"job-hist-{index}",
                    run_id="run-brain-proxy",
                    round_index=index + 1,
                    batch_id="batch-hist",
                    candidate_id="hist-weak",
                    expression=history_candidate.expression,
                    status="completed",
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                    neutralization="SECTOR",
                    decay=0,
                    sharpe=0.05,
                    fitness=0.00,
                    turnover=0.40,
                    drawdown=0.20,
                    returns=0.00,
                    margin=0.01,
                    submission_eligible=False,
                    rejection_reason=None,
                    raw_result_json="{}",
                    metric_source="external_brain",
                    simulated_at=f"2026-04-23T00:0{index}:00+00:00",
                    created_at=f"2026-04-23T00:0{index}:00+00:00",
                )
                for index in range(5)
            ]
        )

        service = SelectionService(
            config=SelectionConfig(
                pre_sim=PreSimSelectionWeightsConfig(brain_robustness_proxy_penalty=0.50),
                brain_robustness_proxy=BrainRobustnessProxyConfig(
                    enabled=True,
                    lookback_rounds=12,
                    min_support=5,
                    sharpe_floor=0.30,
                    fitness_floor=0.10,
                ),
            ),
            repository=repository,
            family_proxy_min_support=999,
        )
        field_registry = _field_registry()
        snapshot = PatternMemorySnapshot(regime_key="regime")
        weak = _candidate(
            "alpha-weak-family",
            "rank(ts_mean(close, 6))",
            field_name="close",
            metadata={
                "generation_source": "fresh",
                "family_signature": "family-weak",
                "search_bucket_id": "bucket-weak",
            },
        )
        clean = _candidate(
            "alpha-clean-family",
            "rank(ts_mean(volume, 6))",
            field_name="volume",
            metadata={
                "generation_source": "recipe_guided",
                "family_signature": "family-clean",
                "search_bucket_id": "bucket-clean",
            },
        )

        scored = service.score_pre_sim_candidates(
            [weak, clean],
            snapshot=snapshot,
            field_registry=field_registry,
            min_pattern_support=1,
            run_id="run-brain-proxy",
            round_index=10,
        )
    finally:
        repository.close()

    breakdowns = {
        item.candidate.alpha_id: item.ranking_rationale["selection_breakdown"]["components"]
        for item in scored
    }
    weak_score = next(item for item in scored if item.candidate.alpha_id == "alpha-weak-family")
    clean_score = next(item for item in scored if item.candidate.alpha_id == "alpha-clean-family")
    assert breakdowns["alpha-weak-family"]["brain_robustness_proxy_penalty"] > 0.0
    assert breakdowns["alpha-clean-family"]["brain_robustness_proxy_penalty"] == 0.0
    assert weak_score.composite_score < clean_score.composite_score
    assert "brain_robustness_proxy_penalty_applied" in weak_score.reason_codes


def test_pre_sim_score_applies_elite_motif_bonus() -> None:
    service = SelectionService(
        config=SelectionConfig(pre_sim=PreSimSelectionWeightsConfig(elite_motif_bonus=0.20))
    )
    field_registry = _field_registry()
    snapshot = PatternMemorySnapshot(regime_key="regime")
    plain = _candidate("alpha-plain", "rank(ts_mean(close, 5))", field_name="close")
    elite = _candidate(
        "alpha-elite",
        "zscore(ts_mean(close, 5))",
        field_name="close",
        metadata={"elite_motif_match_score": 1.0, "elite_motif_ids": ["elite_seed_1"]},
    )

    scored = service.score_pre_sim_candidates(
        [plain, elite],
        snapshot=snapshot,
        field_registry=field_registry,
        min_pattern_support=1,
    )

    elite_score = next(item for item in scored if item.candidate.alpha_id == "alpha-elite")
    assert scored[0].candidate.alpha_id == "alpha-elite"
    assert elite_score.ranking_rationale["selection_breakdown"]["components"]["elite_motif_bonus"] == 0.20
    assert "elite_motif_bonus_applied" in elite_score.reason_codes


def test_pre_sim_score_applies_elite_seed_similarity_penalty() -> None:
    service = SelectionService(
        config=SelectionConfig(pre_sim=PreSimSelectionWeightsConfig(elite_seed_similarity_penalty=0.50))
    )
    field_registry = _field_registry()
    snapshot = PatternMemorySnapshot(regime_key="regime")
    similar = _candidate(
        "alpha-similar",
        "rank(ts_mean(close, 5))",
        field_name="close",
        metadata={
            "elite_seed_similarity": 0.95,
            "elite_seed_similarity_penalty": 1.0,
        },
    )
    clean = _candidate("alpha-clean", "zscore(ts_mean(close, 5))", field_name="close")

    scored = service.score_pre_sim_candidates(
        [similar, clean],
        snapshot=snapshot,
        field_registry=field_registry,
        min_pattern_support=1,
    )

    similar_score = next(item for item in scored if item.candidate.alpha_id == "alpha-similar")
    assert scored[0].candidate.alpha_id == "alpha-clean"
    assert similar_score.ranking_rationale["selection_breakdown"]["components"]["elite_seed_similarity_penalty"] == 0.50
    assert "elite_seed_similarity_penalty_applied" in similar_score.reason_codes


def test_post_sim_score_orders_by_quality() -> None:
    service = SelectionService(config=SelectionConfig())
    candidates = {
        "alpha-good": _candidate("alpha-good", "rank(ts_mean(close, 5))", field_name="close"),
        "alpha-weak": _candidate("alpha-weak", "rank(ts_mean(volume, 5))", field_name="volume"),
    }
    results = [
        _result("alpha-good", fitness=1.2, sharpe=1.4, turnover=0.3, margin=0.12),
        _result("alpha-weak", fitness=0.2, sharpe=0.1, turnover=0.8, margin=0.02),
    ]

    breakdowns, ordered = service.score_post_sim(results, candidates_by_id=candidates)

    assert list(ordered.keys())[0] == "alpha-good"
    assert "multi_objective_quality_score" in breakdowns["alpha-good"].components
    assert "performance_quality" in breakdowns["alpha-good"].components
    assert breakdowns["alpha-good"].components["multi_objective_quality_score"] > breakdowns["alpha-weak"].components["multi_objective_quality_score"]


def test_score_post_sim_uses_field_registry_for_category() -> None:
    service = SelectionService(config=SelectionConfig())
    registry = _field_registry()
    candidate = AlphaCandidate(
        alpha_id="alpha-registry",
        expression="rank(ts_mean(close, 5))",
        normalized_expression="rank(ts_mean(close, 5))",
        generation_mode="template",
        parent_ids=(),
        complexity=3,
        created_at="2026-01-01T00:00:00+00:00",
        template_name="template",
        fields_used=("close",),
        operators_used=("rank", "ts_mean"),
        depth=2,
        generation_metadata={},
    )
    result = _result(candidate.alpha_id, fitness=1.0, sharpe=1.0, turnover=0.3, margin=0.1)

    _, ranked = service.score_post_sim(
        [result],
        candidates_by_id={candidate.alpha_id: candidate},
        field_registry=registry,
    )

    assert ranked[candidate.alpha_id].primary_field_category == "price"


def test_mutation_parent_score_can_boost_learnable_branch() -> None:
    service = SelectionService(config=SelectionConfig())
    candidates = {
        "alpha-a": _candidate("alpha-a", "rank(ts_mean(close, 5))", field_name="close", lineage="branch-a"),
        "alpha-b": _candidate("alpha-b", "rank(ts_mean(volume, 5))", field_name="volume", lineage="branch-b"),
    }
    results = [
        _result("alpha-a", fitness=0.9, sharpe=0.9, turnover=0.3, margin=0.12),
        _result("alpha-b", fitness=0.9, sharpe=0.9, turnover=0.3, margin=0.12),
    ]

    selected, _, mutation_decisions = service.select_mutation_parents(
        results,
        candidates_by_id=candidates,
        top_k=1,
        mutation_learnability_by_id={"alpha-b": 1.0},
    )

    assert selected[0].candidate_id == "alpha-b"
    assert mutation_decisions[0].alpha_id == "alpha-b"
    assert "multi_objective_quality_score" in mutation_decisions[0].breakdown.components
    assert mutation_decisions[0].quality_score > 0.0


def test_pre_screen_candidates_wrap_rejections_with_priority_reason() -> None:
    service = CandidateSelectionService()
    candidate = replace(
        _candidate("alpha-reject", "close", field_name="close"),
        complexity=1,
        depth=1,
        operators_used=(),
    )

    passed, rejected = service.pre_screen_candidates([candidate], field_registry=_field_registry())

    assert passed == []
    assert len(rejected) == 1
    rejected_score = rejected[0]
    assert rejected_score.archive_reason == "pre_screen_low_complexity"
    assert rejected_score.composite_score == 0.0
    assert rejected_score.reason_codes == (
        "pre_screen_low_complexity",
        "pre_screen_trivial_depth",
        "pre_screen_low_operator_diversity",
        "pre_screen_low_field_diversity",
    )
    assert rejected_score.ranking_rationale["pre_screen_reasons"] == list(rejected_score.reason_codes)


def test_pre_screen_candidates_keep_soft_low_field_diversity_candidates() -> None:
    service = CandidateSelectionService()
    candidate = _candidate("alpha-soft", "rank(ts_mean(close, 5))", field_name="close")

    passed, rejected = service.pre_screen_candidates([candidate], field_registry=_field_registry())

    assert passed == [candidate]
    assert rejected == []
    assert candidate.generation_metadata["pre_screen_flags"] == ["pre_screen_low_field_diversity"]


def _field_registry() -> FieldRegistry:
    return FieldRegistry(
        fields={
            "close": FieldSpec(
                name="close",
                dataset="runtime",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="price",
                runtime_available=True,
                category_weight=1.0,
                field_score=1.0,
            ),
            "volume": FieldSpec(
                name="volume",
                dataset="runtime",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="volume",
                runtime_available=True,
                category_weight=0.9,
                field_score=0.9,
            ),
        }
    )


def _candidate(
    alpha_id: str,
    expression: str,
    *,
    field_name: str,
    lineage: str = "",
    metadata: dict | None = None,
) -> AlphaCandidate:
    return AlphaCandidate(
        alpha_id=alpha_id,
        expression=expression,
        normalized_expression=expression,
        generation_mode="template",
        parent_ids=(),
        complexity=3,
        created_at="2026-01-01T00:00:00+00:00",
        template_name="template",
        fields_used=(field_name,),
        operators_used=("rank", "ts_mean"),
        depth=2,
        generation_metadata={
            "field_families": ["price" if field_name == "close" else "volume"],
            "lineage_branch_key": lineage,
            **dict(metadata or {}),
        },
    )


def _result(candidate_id: str, *, fitness: float, sharpe: float, turnover: float, margin: float) -> SimulationResult:
    return SimulationResult(
        expression="expr",
        job_id=f"job-{candidate_id}",
        status="completed",
        region="USA",
        universe="TOP3000",
        delay=1,
        neutralization="SECTOR",
        decay=0,
        metrics={
            "fitness": fitness,
            "sharpe": sharpe,
            "turnover": turnover,
            "margin": margin,
        },
        submission_eligible=True,
        rejection_reason=None,
        raw_result={},
        simulated_at="2026-01-01T00:05:00+00:00",
        candidate_id=candidate_id,
        batch_id="batch-1",
        run_id="run-1",
        round_index=1,
        backend="api",
    )
