from __future__ import annotations

from data.field_registry import FieldRegistry, FieldSpec
from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemorySnapshot
from services.models import CrowdingScore, DedupBatchResult, DedupDecision, SimulationResult
from services.selection_service import SelectionService
from core.config import SelectionConfig


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

    _, ordered = service.score_post_sim(results, candidates_by_id=candidates)

    assert list(ordered.keys())[0] == "alpha-good"


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
