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
    def spec(name: str, category: str, field_type: str = "matrix") -> FieldSpec:
        return FieldSpec(
            name=name,
            dataset="runtime",
            field_type=field_type,
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
            "sector": spec("sector", "group", field_type="vector"),
        }
    )


def _aggregate(*, support: int, avg_outcome: float, failure_rate: float) -> CaseAggregate:
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


def _build_builder(config: GenerationConfig | None = None, *, seed: int = 21) -> GenomeBuilder:
    generation = config or _build_generation_config()
    return GenomeBuilder(
        generation_config=generation,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(generation.allowed_operators),
        field_registry=_build_field_registry(),
        seed=seed,
    )


def test_wrapper_selection_uses_combo_stats(monkeypatch) -> None:
    builder = _build_builder(seed=31)
    snapshot = CaseMemorySnapshot(
        regime_key="combo-wrapper",
        motif_neutralization_stats={
            "momentum|rank": _aggregate(support=12, avg_outcome=2.0, failure_rate=0.1),
            "momentum|zscore": _aggregate(support=12, avg_outcome=-0.5, failure_rate=0.9),
        },
    )

    monkeypatch.setattr(builder.random, "randint", lambda low, high: high)

    counts = {"rank": 0, "zscore": 0, "sign": 0}
    for _ in range(50):
        genome = builder.build_parent_seeded_genome(
            motif="momentum",
            primary_family="price",
            source_mode="combo-wrapper",
            case_snapshot=snapshot,
        )
        for wrapper in genome.wrapper_gene.all_wrappers():
            counts[wrapper] = counts.get(wrapper, 0) + 1

    assert counts["rank"] > counts["zscore"]
    assert counts["rank"] >= counts["zscore"] * 2


def test_failure_soft_blacklist_reduces_motif_weight() -> None:
    builder = _build_builder(seed=41)
    snapshot = CaseMemorySnapshot(
        regime_key="combo-failures",
        failure_combo_counts={
            "family-a|liquidity_conditioned_signal|exploit_local": 12,
            "family-b|liquidity_conditioned_signal|repair": 8,
        },
    )

    counts: dict[str, int] = {}
    for _ in range(100):
        genome = builder.build_guided_genome(
            case_snapshot=snapshot,
            explore=False,
            diversity_tracker=None,
        )
        counts[genome.motif] = counts.get(genome.motif, 0) + 1

    blacklisted = counts.get("liquidity_conditioned_signal", 0)
    others = [count for motif, count in counts.items() if motif != "liquidity_conditioned_signal"]

    assert others
    assert blacklisted < (sum(others) / len(others))


def test_no_combo_data_falls_back_gracefully() -> None:
    builder = _build_builder(seed=51)
    snapshot = CaseMemorySnapshot(regime_key="empty-combos")

    genomes = [
        builder.build_guided_genome(
            case_snapshot=snapshot,
            explore=False,
            diversity_tracker=None,
        )
        for _ in range(10)
    ]

    assert len(genomes) == 10
    assert all(genome.motif for genome in genomes)
