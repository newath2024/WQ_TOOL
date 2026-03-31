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
