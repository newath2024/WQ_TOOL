from __future__ import annotations

from core.config import (
    AdaptiveGenerationConfig,
    AppConfig,
    AuxDataConfig,
    BacktestConfig,
    DataConfig,
    EvaluationConfig,
    GenerationConfig,
    PeriodConfig,
    RuntimeConfig,
    SimulationConfig,
    SplitConfig,
    StorageConfig,
    SubmissionTestConfig,
)
from memory.pattern_memory import FAIL_TAG_PENALTIES, PatternMemoryService, PatternMemorySnapshot, PatternScore


def build_app_config(*, delay_mode: str = "d1", holding_period: int = 2) -> AppConfig:
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
