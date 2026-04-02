from __future__ import annotations

from dataclasses import replace

from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import build_registry
from generator.genome_builder import GenomeBuilder
from memory.case_memory import CaseAggregate, CaseMemorySnapshot


def _build_generation_config() -> GenerationConfig:
    return GenerationConfig(
        allowed_fields=["close", "volume", "pe_ratio"],
        allowed_operators=[
            "rank",
            "zscore",
            "sign",
            "ts_delta",
            "ts_mean",
            "ts_std_dev",
            "ts_corr",
            "ts_covariance",
            "ts_decay_linear",
            "group_neutralize",
            "abs",
            "log",
        ],
        lookbacks=[2, 5, 10, 15],
        max_depth=6,
        complexity_limit=24,
        template_count=12,
        grammar_count=12,
        mutation_count=8,
        normalization_wrappers=["rank", "zscore", "sign"],
        random_seed=13,
    )


def _build_field_registry() -> FieldRegistry:
    def spec(name: str, category: str) -> FieldSpec:
        return FieldSpec(
            name=name,
            dataset="runtime",
            field_type="matrix",
            coverage=1.0,
            alpha_usage_count=0,
            category=category,
            runtime_available=True,
            field_score=1.0,
            category_weight=1.0,
        )

    return FieldRegistry(
        fields={
            "close": spec("close", "price"),
            "volume": spec("volume", "volume"),
            "pe_ratio": spec("pe_ratio", "fundamental"),
            "sector": FieldSpec(
                name="sector",
                dataset="runtime",
                field_type="vector",
                coverage=1.0,
                alpha_usage_count=0,
                category="group",
                runtime_available=True,
                field_score=1.0,
                category_weight=1.0,
            ),
        }
    )


def _aggregate(
    *,
    support: int,
    avg_outcome: float,
    failure_rate: float,
) -> CaseAggregate:
    success_rate = max(0.0, min(1.0, 1.0 - failure_rate))
    return CaseAggregate(
        support=support,
        avg_outcome=avg_outcome,
        avg_fitness=avg_outcome,
        avg_sharpe=avg_outcome,
        avg_eligibility=success_rate,
        avg_robustness=success_rate,
        avg_novelty=0.5,
        avg_turnover_cost=0.3,
        avg_complexity_cost=0.2,
        success_rate=success_rate,
        failure_rate=failure_rate,
    )


def test_pick_motif_uses_neutralization_and_decay_combo_stats(monkeypatch) -> None:
    generation = replace(_build_generation_config(), sim_neutralization="sector", sim_decay=3)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(generation.allowed_operators),
        field_registry=_build_field_registry(),
        seed=65,
    )
    captured: dict[str, list[float]] = {}
    snapshot = CaseMemorySnapshot(
        regime_key="combo-regime",
        motif_neutralization_stats={
            "momentum|sector": _aggregate(support=8, avg_outcome=2.0, failure_rate=0.1),
            "spread|sector": _aggregate(support=8, avg_outcome=-1.0, failure_rate=0.9),
        },
        motif_decay_stats={
            "momentum|3": _aggregate(support=8, avg_outcome=1.5, failure_rate=0.2),
            "spread|3": _aggregate(support=8, avg_outcome=-1.0, failure_rate=0.9),
        },
    )

    def fake_choices(labels, weights, k):
        del k
        captured["labels"] = list(labels)
        captured["weights"] = list(weights)
        return [labels[0]]

    monkeypatch.setattr(builder.random, "choices", fake_choices)

    builder._pick_motif(  # noqa: SLF001
        novelty_bias=False,
        case_snapshot=snapshot,
        diversity_tracker=None,
    )

    labels = captured["labels"]
    weights = captured["weights"]
    assert weights[labels.index("momentum")] > weights[labels.index("spread")]


def test_pick_primitive_uses_field_operator_combo_stats(monkeypatch) -> None:
    generation = _build_generation_config()
    field_registry = _build_field_registry()
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(generation.allowed_operators),
        field_registry=field_registry,
        seed=66,
    )
    captured: dict[str, list[float]] = {}
    primary = field_registry.get("close")
    snapshot = CaseMemorySnapshot(
        regime_key="field-operator-regime",
        field_operator_stats={
            "close|ts_delta": _aggregate(support=10, avg_outcome=2.0, failure_rate=0.1),
            "close|ts_mean": _aggregate(support=10, avg_outcome=-1.0, failure_rate=0.8),
        },
    )

    def fake_choices(labels, weights, k):
        del k
        captured["labels"] = list(labels)
        captured["weights"] = list(weights)
        return [labels[0]]

    monkeypatch.setattr(builder.random, "choices", fake_choices)

    builder._pick_primitive(  # noqa: SLF001
        "momentum",
        primary=primary,
        novelty_bias=False,
        case_snapshot=snapshot,
        diversity_tracker=None,
    )

    labels = captured["labels"]
    weights = captured["weights"]
    assert weights[labels.index("ts_delta")] > weights[labels.index("ts_mean")]
