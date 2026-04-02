from __future__ import annotations

from dataclasses import replace

from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import build_registry
from generator.genome_builder import GenomeBuilder
from memory.case_memory import CaseMemorySnapshot


def _build_generation_config(**overrides: object) -> GenerationConfig:
    config = GenerationConfig(
        allowed_fields=["close", "vwap", "volume", "assets", "analyst_score", "model_score"],
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
        random_seed=23,
    )
    return replace(config, **overrides)


def _build_field_registry() -> FieldRegistry:
    def spec(name: str, category: str, *, field_type: str = "matrix") -> FieldSpec:
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
            "vwap": spec("vwap", "price"),
            "volume": spec("volume", "volume"),
            "assets": spec("assets", "fundamental"),
            "analyst_score": spec("analyst_score", "analyst"),
            "model_score": spec("model_score", "model"),
            "sector": spec("sector", "group", field_type="vector"),
        }
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


def test_unit_safety_spread_motif(monkeypatch) -> None:
    config = _build_generation_config(sim_neutralization="sector")
    builder = _build_builder(config, seed=31)
    primary = _build_field_registry().get("close")

    monkeypatch.setattr(builder, "_pick_numeric_field", lambda **kwargs: primary)

    genomes = [
        builder.build_parent_seeded_genome(
            motif="spread",
            primary_family="price",
            source_mode="unit-safety",
        )
        for _ in range(30)
    ]

    assert all(genome.feature_gene.secondary_field == "vwap" for genome in genomes)
    assert all(genome.feature_gene.secondary_family == "price" for genome in genomes)


def test_mean_reversion_penalty() -> None:
    config = _build_generation_config(sim_neutralization="sector")
    builder = _build_builder(config, seed=41)

    genomes = [builder.build_random_genome(source_mode="brain-evidence") for _ in range(200)]
    mean_reversion_count = sum(1 for genome in genomes if genome.motif == "mean_reversion")

    assert (mean_reversion_count / len(genomes)) < 0.10


def test_soft_blacklist_reduces_bad_motif() -> None:
    builder = _build_builder(seed=51)
    snapshot = CaseMemorySnapshot(
        regime_key="failure-evidence",
        failure_combo_counts={
            "family-a|liquidity_conditioned_signal|repair": 15,
        },
        global_failure_combo_counts={
            "family-global|liquidity_conditioned_signal|guided_exploit": 30,
        },
    )

    genomes = [
        builder.build_guided_genome(
            case_snapshot=snapshot,
            explore=False,
            diversity_tracker=None,
        )
        for _ in range(200)
    ]
    blacklisted_count = sum(1 for genome in genomes if genome.motif == "liquidity_conditioned_signal")

    assert (blacklisted_count / len(genomes)) < 0.03


def test_unit_safety_no_effect_on_non_arithmetic(monkeypatch) -> None:
    config = _build_generation_config(sim_neutralization="sector")
    field_registry = _build_field_registry()
    builder = _build_builder(config, seed=61)
    primary = field_registry.get("close")
    captured: dict[str, float] = {}

    def fake_choices(candidates, weights, k):
        del k
        captured.update({candidate.name: weight for candidate, weight in zip(candidates, weights, strict=True)})
        return [candidates[0]]

    monkeypatch.setattr(builder.random, "choices", fake_choices)

    selected = builder._pick_secondary_field(  # noqa: SLF001
        primary,
        motif="momentum",
        novelty_bias=False,
        case_snapshot=None,
        diversity_tracker=None,
    )

    assert selected is not None
    assert {"vwap", "volume", "assets", "analyst_score", "model_score"} <= set(captured)
