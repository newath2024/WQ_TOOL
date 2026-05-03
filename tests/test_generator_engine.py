from __future__ import annotations

import pytest

from alpha.ast_nodes import ExprNode, FunctionCallNode
from alpha.parser import parse_expression
from core.config import AdaptiveGenerationConfig, GenerationConfig, OperatorDiversityBoostConfig
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import build_registry
from generator.engine import AlphaGenerationEngine, OperatorDiversityState


def _spec(name: str, category: str, *, field_type: str = "matrix", dataset: str = "runtime") -> FieldSpec:
    return FieldSpec(
        name=name,
        dataset=dataset,
        field_type=field_type,
        coverage=1.0,
        alpha_usage_count=0,
        category=category,
        runtime_available=True,
        field_score=1.0,
        category_weight=1.0,
    )


def _field_registry() -> FieldRegistry:
    return FieldRegistry(
        fields={
            "returns": _spec("returns", "price"),
            "close": _spec("close", "price"),
            "volume": _spec("volume", "volume"),
            "anl69_eps_best_eeps_nxt_yr": _spec("anl69_eps_best_eeps_nxt_yr", "analyst", dataset="anl69"),
            "anl46_sentiment": _spec("anl46_sentiment", "analyst", dataset="anl46"),
            "anl39_epschngin": _spec("anl39_epschngin", "fundamental", dataset="anl39"),
            "anl69_roa_best_cur_fiscal_year_period": _spec(
                "anl69_roa_best_cur_fiscal_year_period",
                "analyst",
                dataset="anl69",
            ),
            "anl69_eps_expected_report_dt": _spec(
                "anl69_eps_expected_report_dt",
                "analyst",
                field_type="event",
                dataset="anl69",
            ),
            "anl69_roe_expected_report_dt": _spec(
                "anl69_roe_expected_report_dt",
                "analyst",
                field_type="event",
                dataset="anl69",
            ),
            "sector": _spec("sector", "group", field_type="vector"),
        }
    )


def _generation_config(*, allowed_operators: list[str] | None = None) -> GenerationConfig:
    return GenerationConfig(
        allowed_fields=[
            "returns",
            "close",
            "volume",
            "anl69_eps_best_eeps_nxt_yr",
            "anl46_sentiment",
            "anl39_epschngin",
            "anl69_roa_best_cur_fiscal_year_period",
            "anl69_eps_expected_report_dt",
            "anl69_roe_expected_report_dt",
        ],
        allowed_operators=allowed_operators
        or [
            "rank",
            "zscore",
            "ts_delta",
            "ts_mean",
            "ts_std_dev",
            "ts_corr",
            "ts_covariance",
            "ts_decay_linear",
            "ts_rank",
            "ts_sum",
            "ts_av_diff",
            "ts_arg_max",
            "ts_arg_min",
            "days_from_last_change",
            "inverse",
            "reverse",
            "group_neutralize",
            "abs",
        ],
        lookbacks=[5, 10, 20],
        max_depth=6,
        complexity_limit=24,
        template_count=4,
        grammar_count=4,
        mutation_count=2,
        normalization_wrappers=["rank", "zscore"],
        random_seed=17,
    )


def _adaptive_config(*, enabled: bool = True, seed_corr_probability: float = 0.0) -> AdaptiveGenerationConfig:
    return AdaptiveGenerationConfig(
        operator_diversity_boost=OperatorDiversityBoostConfig(
            enabled=enabled,
            seed_corr_pair_probability=seed_corr_probability,
        )
    )


def _engine(*, allowed_operators: list[str] | None = None, enabled: bool = True) -> AlphaGenerationEngine:
    generation = _generation_config(allowed_operators=allowed_operators)
    return AlphaGenerationEngine(
        config=generation,
        registry=build_registry(generation.allowed_operators),
        field_registry=_field_registry(),
        adaptive_config=_adaptive_config(enabled=enabled),
    )


def _find_call(node: ExprNode, name: str) -> FunctionCallNode | None:
    if isinstance(node, FunctionCallNode) and node.name == name:
        return node
    children = ()
    if isinstance(node, FunctionCallNode):
        children = node.args
    elif hasattr(node, "left") and hasattr(node, "right"):
        children = (node.left, node.right)
    elif hasattr(node, "operand"):
        children = (node.operand,)
    for child in children:
        found = _find_call(child, name)
        if found is not None:
            return found
    return None


def test_dominant_operator_weight_decreases_with_usage() -> None:
    state = OperatorDiversityState(
        config=OperatorDiversityBoostConfig(enabled=True),
        allowed_operators={"ts_mean"},
    )
    state.record_operators(["ts_mean"] * 50)

    assert state.adjusted_weight("ts_mean", 1.0) < 1.0


def test_underused_operator_gets_boosted() -> None:
    state = OperatorDiversityState(
        config=OperatorDiversityBoostConfig(enabled=True),
        allowed_operators={"ts_corr", "days_from_last_change"},
    )

    assert state.adjusted_weight("ts_corr", 1.0) > 1.0
    assert state.adjusted_weight("days_from_last_change", 1.0) > 1.0


def test_underused_boost_decays_after_use() -> None:
    state = OperatorDiversityState(
        config=OperatorDiversityBoostConfig(enabled=True),
        allowed_operators={"ts_corr"},
    )

    first = state.adjusted_weight("ts_corr", 1.0)
    state.record_operators(["ts_corr"])
    second = state.adjusted_weight("ts_corr", 1.0)
    state.record_operators(["ts_corr"])
    third = state.adjusted_weight("ts_corr", 1.0)

    assert first > second > third > 0.0


def test_corr_operator_selects_two_distinct_fields() -> None:
    engine = _engine()

    expression = engine._render_correlation_diversity_expression("ts_corr")  # noqa: SLF001

    assert expression is not None
    candidate = engine.build_candidate(expression, "test", ())
    assert candidate is not None
    call = _find_call(parse_expression(expression), "ts_corr")
    assert call is not None
    assert len(call.args) == 3
    assert str(call.args[0]) != str(call.args[1])


def test_days_from_last_change_selects_date_field() -> None:
    engine = _engine()

    expression = engine._render_days_from_last_change_expression()  # noqa: SLF001

    assert expression is not None
    assert "expected_report_dt" in expression
    assert engine.build_candidate(expression, "test", ()) is not None


def test_diversity_boost_disabled_when_config_off() -> None:
    state = OperatorDiversityState(
        config=OperatorDiversityBoostConfig(enabled=False),
        allowed_operators={"ts_corr"},
    )
    state.record_operators(["ts_corr"] * 5)

    assert state.adjusted_weight("ts_corr", 1.0) == pytest.approx(1.0)


def test_lane_allowlist_respected_with_diversity_boost() -> None:
    engine = _engine(allowed_operators=["rank", "ts_mean"], enabled=True)
    state = engine._new_operator_diversity_state()  # noqa: SLF001
    validation_ctx = engine.prepare_validation_context()

    candidate = engine._try_operator_diversity_candidate(  # noqa: SLF001
        session=None,
        operator_diversity=state,
        existing_normalized=set(),
        validation_ctx=validation_ctx,
        mode="test",
        mutation_mode="test",
    )

    assert state.adjusted_weight("ts_corr", 1.0) == 0.0
    assert candidate is None
