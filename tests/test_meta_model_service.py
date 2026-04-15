from __future__ import annotations

import json

from core.config import MetaModelConfig, SelectionConfig
from data.field_registry import FieldRegistry, FieldSpec
from generator.engine import AlphaCandidate
from memory.case_memory import ObjectiveVector
from memory.pattern_memory import PatternMemorySnapshot, StructuralSignature
from services.meta_model_service import MetaModelFeatureInput, MetaModelPrediction, MetaModelService
from services.selection_service import SelectionService
from storage.models import RunFieldScoreRecord, SelectionScoreRecord
from storage.repository import SQLiteRepository


def test_meta_model_ignores_current_round_rows_when_training() -> None:
    repository = SQLiteRepository(":memory:")
    _seed_training_example(
        repository,
        run_id="run-main",
        round_index=1,
        alpha_id="alpha-good-hist",
        motif="good",
        outcome_score=0.8,
    )
    _seed_training_example(
        repository,
        run_id="run-main",
        round_index=2,
        alpha_id="alpha-bad-hist",
        motif="bad",
        outcome_score=-0.4,
    )
    _seed_training_example(
        repository,
        run_id="run-main",
        round_index=5,
        alpha_id="alpha-same-round",
        motif="good",
        outcome_score=1.2,
    )
    service = MetaModelService(
        repository,
        config=MetaModelConfig(
            min_train_rows=2,
            min_positive_rows=1,
            lookback_rounds=20,
        ),
    )

    prediction = service.score_candidates(
        run_id="run-main",
        round_index=5,
        effective_regime_key="eff-1",
        feature_inputs=[_meta_input(alpha_id="alpha-current", motif="good")],
    )["alpha-current"]

    assert prediction.used is True
    assert prediction.train_rows == 2
    assert prediction.positive_rows == 1


def test_meta_model_falls_back_cleanly_when_history_is_insufficient() -> None:
    repository = SQLiteRepository(":memory:")
    _seed_training_example(
        repository,
        run_id="run-main",
        round_index=1,
        alpha_id="alpha-good-hist",
        motif="good",
        outcome_score=0.8,
    )
    service = MetaModelService(
        repository,
        config=MetaModelConfig(
            min_train_rows=5,
            min_positive_rows=2,
            lookback_rounds=20,
        ),
    )

    prediction = service.score_candidates(
        run_id="run-main",
        round_index=3,
        effective_regime_key="eff-1",
        feature_inputs=[_meta_input(alpha_id="alpha-current", motif="good", heuristic=0.44)],
    )["alpha-current"]

    assert prediction.used is False
    assert prediction.ml_positive_outcome_prob == 0.0
    assert prediction.blended_predicted_quality == 0.44


def test_selection_service_blends_meta_model_probability_into_breakdown(monkeypatch) -> None:
    service = SelectionService(
        config=SelectionConfig(),
        meta_model_service=_StubMetaModelService(),
    )
    signature = _signature()
    monkeypatch.setattr(
        service.memory_service,
        "score_expression",
        lambda *args, **kwargs: (0.5, 0.4, signature, None),
    )
    monkeypatch.setattr(
        service.case_memory_service,
        "predict_objectives",
        lambda *args, **kwargs: ObjectiveVector(
            fitness=0.4,
            sharpe=0.4,
            eligibility=0.0,
            robustness=0.4,
            novelty=0.4,
            diversity=0.4,
            turnover_cost=0.2,
            complexity_cost=0.1,
        ),
    )
    field_registry = _field_registry()
    candidates = [
        _candidate("alpha-ml-good", motif="good"),
        _candidate("alpha-ml-bad", motif="bad"),
    ]

    scored = service.score_pre_sim_candidates(
        candidates,
        snapshot=PatternMemorySnapshot(regime_key="regime-1"),
        field_registry=field_registry,
        min_pattern_support=1,
        run_id="run-current",
        round_index=11,
        effective_regime_key="eff-1",
    )

    assert scored[0].candidate.alpha_id == "alpha-ml-good"
    components = scored[0].ranking_rationale["selection_breakdown"]["components"]
    assert components["ml_positive_outcome_prob"] == 0.95
    assert components["meta_model_used"] == 1.0
    assert components["blended_predicted_quality"] > components["heuristic_predicted_quality"]


class _StubMetaModelService:
    def score_candidates(self, **kwargs) -> dict[str, MetaModelPrediction]:
        return {
            "alpha-ml-good": MetaModelPrediction(
                alpha_id="alpha-ml-good",
                heuristic_predicted_quality=0.34,
                ml_positive_outcome_prob=0.95,
                blended_predicted_quality=0.462,
                train_rows=12,
                positive_rows=4,
                used=True,
            ),
            "alpha-ml-bad": MetaModelPrediction(
                alpha_id="alpha-ml-bad",
                heuristic_predicted_quality=0.34,
                ml_positive_outcome_prob=0.05,
                blended_predicted_quality=0.282,
                train_rows=12,
                positive_rows=4,
                used=True,
            ),
        }


def _seed_training_example(
    repository: SQLiteRepository,
    *,
    run_id: str,
    round_index: int,
    alpha_id: str,
    motif: str,
    outcome_score: float,
) -> None:
    repository.upsert_run(
        run_id=run_id,
        seed=7,
        config_path="config/test.yaml",
        config_snapshot="{}",
        status="running",
        started_at="2026-01-01T00:00:00+00:00",
    )
    repository.save_alpha_candidates(
        run_id,
        [
            AlphaCandidate(
                alpha_id=alpha_id,
                expression=f"rank(ts_mean(close, {5 + round_index}))",
                normalized_expression=f"rank(ts_mean(close, {5 + round_index}))",
                generation_mode="guided_explore",
                parent_ids=(),
                complexity=4,
                created_at="2026-01-01T00:00:00+00:00",
                template_name=motif,
                fields_used=("close",),
                operators_used=("rank", "ts_mean"),
                depth=3,
                generation_metadata={
                    "motif": motif,
                    "field_families": ["price"],
                    "operator_path": ["rank", "ts_mean"],
                    "mutation_mode": "novelty",
                    "lineage_branch_key": f"branch-{motif}",
                },
            )
        ],
    )
    repository.replace_run_field_scores(
        run_id,
        [
            RunFieldScoreRecord(
                run_id=run_id,
                field_name="close",
                runtime_available=True,
                field_type="matrix",
                category="price",
                field_score=0.9,
                coverage=1.0,
                alpha_usage_count=1,
                created_at="2026-01-01T00:00:00+00:00",
            )
        ],
    )
    repository.save_selection_scores(
        [
            SelectionScoreRecord(
                run_id=run_id,
                round_index=round_index,
                alpha_id=alpha_id,
                score_stage="pre_sim",
                composite_score=0.31,
                selected=True,
                rank=1,
                reason_codes_json="[]",
                breakdown_json=json.dumps(
                    {
                        "score_stage": "pre_sim",
                        "composite_score": 0.31,
                        "components": {
                            "predicted_quality": 0.33,
                            "novelty": 0.4,
                            "family_diversity": 0.3,
                            "regime_fit": 0.2,
                            "exploration_bonus": 1.0,
                            "duplicate_risk": 0.05,
                            "crowding_penalty": 0.01,
                            "complexity_cost": 0.1,
                        },
                        "reason_codes": [],
                    },
                    sort_keys=True,
                ),
                created_at="2026-01-01T00:00:00+00:00",
            )
        ]
    )
    repository.connection.execute(
        """
        INSERT INTO alpha_cases
        (run_id, alpha_id, region, regime_key, global_regime_key, market_regime_key, effective_regime_key,
         regime_label, regime_confidence, metric_source, family_signature, structural_signature_json, genome_hash,
         genome_json, motif, field_families_json, operator_path_json, complexity_bucket, turnover_bucket,
         horizon_bucket, mutation_mode, parent_family_signatures_json, fail_tags_json, success_tags_json,
         objective_vector_json, outcome_score, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            alpha_id,
            "USA",
            "regime-1",
            "global-1",
            "market-1",
            "eff-1",
            "normal",
            0.5,
            "external_brain",
            f"family-{motif}",
            "{}",
            "",
            "{}",
            motif,
            json.dumps(["price"]),
            json.dumps(["rank", "ts_mean"]),
            "moderate",
            "balanced",
            "medium",
            "novelty",
            "[]",
            "[]",
            "[]",
            json.dumps(
                {
                    "fitness": max(0.0, outcome_score),
                    "sharpe": max(0.0, outcome_score),
                    "eligibility": 0.0,
                    "robustness": 1.0,
                    "novelty": 0.5,
                    "diversity": 0.5,
                    "turnover_cost": 0.1,
                    "complexity_cost": 0.2,
                },
                sort_keys=True,
            ),
            outcome_score,
            "2026-01-01T00:01:00+00:00",
        ),
    )
    repository.connection.commit()


def _meta_input(alpha_id: str, motif: str, heuristic: float = 0.33) -> MetaModelFeatureInput:
    return MetaModelFeatureInput(
        alpha_id=alpha_id,
        generation_mode="guided_explore",
        template_name=motif,
        motif=motif,
        mutation_mode="novelty",
        lineage_branch_key=f"branch-{motif}",
        effective_regime_key="eff-1",
        primary_field_family="price",
        field_families=("price",),
        operator_path_key="rank>ts_mean",
        operator_path_head="rank",
        operator_path_depth=2,
        complexity=4,
        depth=3,
        complexity_bucket="moderate",
        horizon_bucket="medium",
        turnover_bucket="balanced",
        field_score=0.9,
        novelty_score=0.4,
        family_diversity=0.3,
        duplicate_risk=0.05,
        crowding_penalty=0.01,
        regime_fit=0.2,
        heuristic_predicted_quality=heuristic,
    )


def _signature() -> StructuralSignature:
    return StructuralSignature(
        operators=("rank", "ts_mean"),
        operator_families=("ranking", "smoothing"),
        operator_path=("rank", "ts_mean"),
        fields=("close",),
        field_families=("price",),
        lookbacks=(5,),
        wrappers=(),
        depth=3,
        complexity=4,
        complexity_bucket="moderate",
        horizon_bucket="medium",
        turnover_bucket="balanced",
        motif="good",
        family_signature="family-good",
        subexpressions=("ts_mean(close, 5)",),
    )


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
                field_score=0.9,
            )
        }
    )


def _candidate(alpha_id: str, *, motif: str) -> AlphaCandidate:
    return AlphaCandidate(
        alpha_id=alpha_id,
        expression=f"rank(ts_mean(close, {5 if motif == 'good' else 6}))",
        normalized_expression=f"rank(ts_mean(close, {5 if motif == 'good' else 6}))",
        generation_mode="guided_explore",
        parent_ids=(),
        complexity=4,
        created_at="2026-01-01T00:00:00+00:00",
        template_name=motif,
        fields_used=("close",),
        operators_used=("rank", "ts_mean"),
        depth=3,
        generation_metadata={
            "motif": motif,
            "field_families": ["price"],
            "operator_path": ["rank", "ts_mean"],
            "mutation_mode": "novelty",
            "lineage_branch_key": f"branch-{motif}",
        },
    )
