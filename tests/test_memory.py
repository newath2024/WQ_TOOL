from __future__ import annotations

import json

from core.config import (
    AdaptiveGenerationConfig,
    AppConfig,
    AuxDataConfig,
    BacktestConfig,
    BrainConfig,
    DataConfig,
    EvaluationConfig,
    GenerationConfig,
    PeriodConfig,
    RegionLearningConfig,
    RuntimeConfig,
    SimulationConfig,
    SplitConfig,
    StorageConfig,
    SubmissionTestConfig,
)
from generator.engine import AlphaCandidate
from generator.genome import ComplexityGene, FeatureGene, Genome, HorizonGene, RegimeGene, TransformGene, TurnoverGene, WrapperGene
from memory.case_memory import CaseMemoryRecord, CaseMemoryService, ObjectiveVector
from memory.pattern_memory import FAIL_TAG_PENALTIES, PatternMemoryService, PatternMemorySnapshot, PatternScore
from storage.alpha_history import AlphaHistoryStore
from storage.sqlite import connect_sqlite


def build_app_config(*, delay_mode: str = "d1", holding_period: int = 2, region: str = "USA") -> AppConfig:
    return AppConfig(
        data=DataConfig(path="examples/sample_data/daily_ohlcv.csv"),
        aux_data=AuxDataConfig(),
        splits=SplitConfig(
            train=PeriodConfig(start="2021-01-01", end="2021-02-01"),
            validation=PeriodConfig(start="2021-02-02", end="2021-03-01"),
            test=PeriodConfig(start="2021-03-02", end="2021-03-31"),
        ),
        generation=GenerationConfig(
            allowed_fields=["open", "high", "low", "close", "volume", "returns"],
            allowed_operators=["rank", "delta", "ts_mean", "zscore", "decay_linear"],
            lookbacks=[2, 5, 10],
            max_depth=5,
            complexity_limit=20,
            template_count=8,
            grammar_count=8,
            mutation_count=4,
            normalization_wrappers=["rank", "zscore", "sign"],
            random_seed=7,
        ),
        adaptive_generation=AdaptiveGenerationConfig(),
        simulation=SimulationConfig(delay_mode=delay_mode, neutralization="sector"),
        backtest=BacktestConfig(
            timeframe="1d",
            mode="cross_sectional",
            portfolio_construction="long_short",
            selection_fraction=0.25,
            signal_delay=1,
            holding_period=holding_period,
            volatility_scaling=False,
            volatility_lookback=10,
            transaction_cost_bps=5.0,
            annualization_factor=252,
            symbol_rank_window=10,
            upper_quantile=0.8,
            lower_quantile=0.2,
            turnover_penalty=0.1,
            drawdown_penalty=0.5,
        ),
        evaluation=EvaluationConfig(
            min_sharpe=0.0,
            max_turnover=1.0,
            min_observations=5,
            max_drawdown=0.5,
            min_stability=0.2,
            signal_correlation_threshold=0.95,
            returns_correlation_threshold=0.95,
            top_k=5,
        ),
        submission_tests=SubmissionTestConfig(),
        storage=StorageConfig(path=":memory:"),
        brain=BrainConfig(region=region),
        runtime=RuntimeConfig(log_level="WARNING"),
    )


def test_pattern_memory_extracts_structural_signature_and_genes() -> None:
    service = PatternMemoryService()
    signature = service.extract_signature("rank(decay_linear(delta(close, 5), 10))")

    assert set(signature.operators) == {"delta", "decay_linear", "rank"}
    assert signature.fields == ("close",)
    assert signature.lookbacks == (5, 10)
    assert signature.wrappers == ("rank",)
    assert signature.depth >= 3
    assert "delta(close,5)" in signature.subexpressions
    assert "decay_linear(delta(close,5),10)" in signature.subexpressions

    observations = service.build_observations(signature)
    kinds = {item.pattern_kind for item in observations}
    assert {"family", "operator", "field", "lookback", "wrapper", "subexpression"} <= kinds


def test_pattern_memory_regime_key_changes_with_simulation_profile() -> None:
    service = PatternMemoryService()
    base = build_app_config(delay_mode="d1", holding_period=2)
    changed = build_app_config(delay_mode="fast_d1", holding_period=1)

    regime_a = service.build_regime_key("dataset-fingerprint", base)
    regime_b = service.build_regime_key("dataset-fingerprint", changed)

    assert regime_a != regime_b


def test_region_learning_context_is_region_local_but_global_key_ignores_region() -> None:
    service = PatternMemoryService()
    usa = build_app_config(region="USA")
    eur = build_app_config(region="EUR")

    usa_context = service.build_learning_context("dataset-fingerprint", usa)
    eur_context = service.build_learning_context("dataset-fingerprint", eur)

    assert usa_context.regime_key != eur_context.regime_key
    assert usa_context.global_regime_key == eur_context.global_regime_key


def test_region_learning_blend_weights_ramp_from_global_to_local() -> None:
    service = PatternMemoryService()
    config = RegionLearningConfig(
        min_local_pattern_samples=10,
        full_local_pattern_samples=30,
        min_local_case_samples=5,
        full_local_case_samples=15,
    )

    cold = service.compute_blend_diagnostics(scope="pattern", local_samples=0, global_samples=12, config=config)
    mid = service.compute_blend_diagnostics(scope="pattern", local_samples=20, global_samples=12, config=config)
    hot = service.compute_blend_diagnostics(scope="case", local_samples=20, global_samples=8, config=config)

    assert cold.local_weight == 0.0
    assert cold.global_weight == 1.0
    assert 0.0 < mid.local_weight < 1.0
    assert round(mid.local_weight, 2) == 0.50
    assert hot.local_weight == 1.0
    assert hot.global_weight == 0.0


def test_pattern_memory_outcome_score_and_thresholded_pattern_scoring() -> None:
    service = PatternMemoryService()
    expression = "rank(delta(close, 5))"
    signature = service.extract_signature(expression)
    observations = service.build_observations(signature)

    strong_patterns = {}
    for item in observations:
        support = 5 if item.pattern_kind != "wrapper" else 1
        strong_patterns[item.pattern_id] = PatternScore(
            pattern_id=item.pattern_id,
            pattern_kind=item.pattern_kind,
            pattern_value=item.pattern_value,
            support=support,
            success_count=4,
            failure_count=1,
            avg_outcome=0.45,
            avg_behavioral_novelty=0.70,
            fail_tag_counts={"high_turnover": 1} if item.pattern_kind == "wrapper" else {},
            pattern_score=0.55 if support >= 3 else -1.50,
        )

    score, novelty, _, used_observations = service.score_expression(
        expression,
        PatternMemorySnapshot(regime_key="regime-1", patterns=strong_patterns),
        min_pattern_support=3,
    )
    outcome_score = service.compute_outcome_score(
        validation_fitness=1.5,
        passed_filters=True,
        selected_top_alpha=True,
        behavioral_novelty_score=0.80,
        fail_tags=["high_turnover", "weak_validation"],
    )

    assert score > 0.0
    assert novelty == 0.70
    assert len(used_observations) >= 4
    assert outcome_score > 0.0
    assert outcome_score < 1.0 + 0.25 + 0.25 + 0.10 * 0.80
    assert FAIL_TAG_PENALTIES["high_turnover"] > 0


def test_blended_patterns_does_not_double_compute(monkeypatch) -> None:
    from memory import pattern_memory as pm

    call_count = 0
    original = pm._merge_pattern_scores

    def counted(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pm, "_merge_pattern_scores", counted)

    snapshot = PatternMemorySnapshot(
        regime_key="r",
        patterns={"a": _make_pattern("a"), "b": _make_pattern("b")},
        global_patterns={"b": _make_pattern("b"), "c": _make_pattern("c")},
    )

    result = snapshot._blended_patterns()
    unique_ids = set(snapshot.patterns) | set(snapshot.global_patterns)

    assert set(result) == unique_ids
    assert call_count == len(unique_ids)


def test_alpha_history_get_outcome_score_returns_none_for_missing() -> None:
    connection = connect_sqlite(":memory:")
    try:
        store = AlphaHistoryStore(connection, PatternMemoryService())
        result = store.get_outcome_score("nonexistent-run", "nonexistent-alpha")
    finally:
        connection.close()

    assert result is None


def test_alpha_history_get_outcome_score_returns_latest() -> None:
    connection = connect_sqlite(":memory:")
    try:
        store = AlphaHistoryStore(connection, PatternMemoryService())
        run_id = "run-1"
        alpha_id = "alpha-1"
        _insert_alpha_history_row(
            store,
            run_id=run_id,
            alpha_id=alpha_id,
            outcome_score=0.5,
            created_at="2026-01-01T00:00:00+00:00",
        )
        store.connection.execute(
            """
            UPDATE alpha_history
            SET outcome_score = ?, created_at = ?
            WHERE run_id = ? AND alpha_id = ?
            """,
            (0.8, "2026-01-01T00:05:00+00:00", run_id, alpha_id),
        )
        store.connection.commit()

        result = store.get_outcome_score(run_id, alpha_id)
    finally:
        connection.close()

    assert result == 0.8


def test_case_memory_snapshot_tracks_combo_stats_and_blends() -> None:
    service = CaseMemoryService()
    local = _make_case_record(
        alpha_id="local-alpha",
        outcome_score=2.0,
        neutralization="sector",
        decay=3,
    )
    global_case = _make_case_record(
        alpha_id="global-alpha",
        outcome_score=0.0,
        neutralization="sector",
        decay=3,
        created_at="2026-01-02T00:00:00+00:00",
    )

    snapshot = service.build_snapshot(
        "local-regime",
        [local],
        global_regime_key="shared-global",
        global_records=[global_case],
        blend=PatternMemoryService().compute_blend_diagnostics(
            scope="case",
            local_samples=1,
            global_samples=1,
            config=RegionLearningConfig(
                min_local_case_samples=2,
                full_local_case_samples=4,
                min_local_pattern_samples=2,
                full_local_pattern_samples=4,
            ),
        ),
    )

    motif_neutralization = snapshot.aggregate_for("motif_neutralization", "momentum|sector")
    motif_decay = snapshot.aggregate_for("motif_decay", "momentum|3")
    field_operator = snapshot.aggregate_for("field_operator", "close|ts_delta")
    local_motif_neutralization = snapshot.aggregate_for("motif_neutralization", "momentum|sector", scope="local")
    global_motif_neutralization = snapshot.aggregate_for("motif_neutralization", "momentum|sector", scope="global")

    assert motif_neutralization is not None
    assert motif_decay is not None
    assert field_operator is not None
    assert local_motif_neutralization is not None
    assert global_motif_neutralization is not None
    assert local_motif_neutralization.avg_outcome == 2.0
    assert global_motif_neutralization.avg_outcome == 0.0
    assert motif_decay.support == 1
    assert field_operator.support == 1
    assert "momentum|sector" in snapshot.stats_for_scope("motif_neutralization", scope="blended")


def test_alpha_history_case_record_persists_combo_context() -> None:
    connection = connect_sqlite(":memory:")
    try:
        store = AlphaHistoryStore(connection, PatternMemoryService())
        genome = _make_genome()
        candidate = AlphaCandidate(
            alpha_id="alpha-1",
            expression="rank(ts_delta(close,5))",
            normalized_expression="rank(ts_delta(close,5))",
            generation_mode="guided_exploit",
            parent_ids=(),
            complexity=4,
            created_at="2026-01-01T00:00:00+00:00",
            template_name="momentum",
            fields_used=("close",),
            operators_used=("rank", "ts_delta"),
            depth=3,
            generation_metadata={
                "genome": genome.to_dict(),
                "genome_hash": genome.stable_hash,
                "motif": "momentum",
                "mutation_mode": "guided_exploit",
            },
        )
        signature = PatternMemoryService().extract_signature(candidate.expression, generation_metadata={"motif": "momentum"})

        record = store._build_case_record(  # noqa: SLF001
            run_id="run-1",
            regime_key="regime-1",
            region="USA",
            global_regime_key="global-1",
            market_regime_key="market-1",
            effective_regime_key="effective-1",
            regime_label="unknown",
            regime_confidence=0.0,
            candidate=candidate,
            structural_signature=signature,
            metric_source="local_backtest",
            simulation_context={"neutralization": "SECTOR", "decay": 5},
            fail_tags=(),
            success_tags=("passed_validation_filters",),
            objective_vector=ObjectiveVector(),
            outcome_score=1.0,
            created_at="2026-01-01T00:00:00+00:00",
        )

        payload = json.loads(record.genome_json)
        restored = store.case_memory_service.record_from_persisted_payload(
            row={
                "run_id": record.run_id,
                "alpha_id": record.alpha_id,
                "region": record.region,
                "regime_key": record.regime_key,
                "global_regime_key": record.global_regime_key,
                "metric_source": record.metric_source,
                "family_signature": record.family_signature,
                "genome_hash": record.genome_hash,
                "genome_json": record.genome_json,
                "motif": record.motif,
                "field_families_json": record.field_families_json,
                "operator_path_json": record.operator_path_json,
                "complexity_bucket": record.complexity_bucket,
                "turnover_bucket": record.turnover_bucket,
                "horizon_bucket": record.horizon_bucket,
                "mutation_mode": record.mutation_mode,
                "parent_family_signatures_json": record.parent_family_signatures_json,
                "fail_tags_json": record.fail_tags_json,
                "success_tags_json": record.success_tags_json,
                "objective_vector_json": record.objective_vector_json,
                "outcome_score": record.outcome_score,
                "created_at": record.created_at,
            },
            structural_signature=signature,
        )
    finally:
        connection.close()

    assert payload["_case_memory_context"]["neutralization"] == "sector"
    assert payload["_case_memory_context"]["decay"] == 5
    assert restored.neutralization == "sector"
    assert restored.decay == 5


def _make_pattern(pattern_id: str) -> PatternScore:
    return PatternScore(
        pattern_id=pattern_id,
        pattern_kind="subexpression",
        pattern_value=pattern_id,
        support=3,
        success_count=2,
        failure_count=1,
        avg_outcome=0.25,
        avg_behavioral_novelty=0.5,
        fail_tag_counts={},
        pattern_score=0.4,
    )


def _make_genome() -> Genome:
    return Genome(
        feature_gene=FeatureGene(primary_field="close", primary_family="price"),
        transform_gene=TransformGene(motif="momentum", primitive_transform="ts_delta"),
        horizon_gene=HorizonGene(fast_window=5, slow_window=10),
        wrapper_gene=WrapperGene(post_wrappers=("rank",)),
        regime_gene=RegimeGene(),
        turnover_gene=TurnoverGene(),
        complexity_gene=ComplexityGene(),
    )


def _make_case_record(
    *,
    alpha_id: str,
    outcome_score: float,
    neutralization: str,
    decay: int,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> CaseMemoryRecord:
    signature = PatternMemoryService().extract_signature("rank(ts_delta(close,5))", generation_metadata={"motif": "momentum"})
    genome = _make_genome()
    return CaseMemoryRecord(
        run_id="run-1",
        alpha_id=alpha_id,
        region="USA",
        regime_key="local-regime",
        global_regime_key="shared-global",
        metric_source="local_backtest",
        family_signature=signature.family_signature,
        structural_signature=signature,
        genome_hash=genome.stable_hash,
        genome=genome,
        motif="momentum",
        field_families=("price",),
        operator_path=("rank", "ts_delta"),
        complexity_bucket=signature.complexity_bucket,
        turnover_bucket=signature.turnover_bucket,
        horizon_bucket=signature.horizon_bucket,
        mutation_mode="guided_exploit",
        parent_family_signatures=(),
        fail_tags=(),
        success_tags=("passed_validation_filters",) if outcome_score > 0 else (),
        objective_vector=ObjectiveVector(fitness=outcome_score, sharpe=outcome_score),
        outcome_score=outcome_score,
        created_at=created_at,
        neutralization=neutralization,
        decay=decay,
    )


def _insert_alpha_history_row(
    store: AlphaHistoryStore,
    *,
    run_id: str,
    alpha_id: str,
    outcome_score: float,
    created_at: str,
) -> None:
    signature = store.memory_service.extract_signature("rank(close)", generation_metadata={})
    store.connection.execute(
        """
        INSERT INTO alpha_history
        (run_id, alpha_id, region, regime_key, global_regime_key, market_regime_key, effective_regime_key,
         regime_label, regime_confidence, expression, normalized_expression, generation_mode, generation_metadata_json,
         parent_refs_json, structural_signature_json, gene_ids_json, train_metrics_json, validation_metrics_json,
         test_metrics_json, validation_signal_json, validation_returns_json, outcome_score, behavioral_novelty_score,
         passed_filters, selected, submission_pass_count, diagnosis_summary_json, rejection_reasons_json, metric_source, created_at)
        VALUES (?, ?, '', 'legacy', '', '', 'legacy', 'unknown', 0.0, ?, ?, 'template', '{}', '[]', ?, '[]', '{}', '{}', '{}', '{}', '{}', ?, 0.5, 1, 1, 1, ?, '[]', 'external_brain', ?)
        """,
        (
            run_id,
            alpha_id,
            "rank(close)",
            "rank(close)",
            json.dumps(signature.to_dict(), sort_keys=True),
            outcome_score,
            json.dumps({"fail_tags": [], "success_tags": []}, sort_keys=True),
            created_at,
        ),
    )
    store.connection.commit()
