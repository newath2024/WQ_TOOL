from __future__ import annotations

import random
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha.ast_nodes import node_depth, to_expression
from alpha.parser import parse_expression
from alpha.validator import validate_expression
from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry, FieldScoreWeights, FieldSpec, build_field_registry
from evaluation.alpha_distance import structural_distance
from features.registry import build_registry
from generator.crossover import GenomeCrossover
from generator.diversity_tracker import GenerationDiversityTracker
from generator.engine import AlphaGenerationEngine
from generator.genome import FeatureGene, Genome, HorizonGene, TransformGene
from generator.genome_builder import COMPATIBLE_CATEGORIES, GenomeBuilder
from generator.grammar import MOTIF_LIBRARY, MotifGrammar
from generator.novelty import NoveltySearch
from generator.repair_policy import RepairPolicy
from services.candidate_selection_service import CandidateSelectionService
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot
from tests.conftest import build_sample_market_frame


def build_configs() -> tuple[GenerationConfig, AdaptiveGenerationConfig]:
    generation = GenerationConfig(
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
    adaptive = AdaptiveGenerationConfig()
    adaptive.crossover_rate = 1.0
    return generation, adaptive


def build_runtime_field_registry():
    market = build_sample_market_frame()
    close = market.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    volume = market.pivot(index="timestamp", columns="symbol", values="volume").sort_index()
    group = pd.DataFrame(
        [["technology", "technology", "financials", "financials"]],
        index=[close.index[0]],
        columns=close.columns,
    ).reindex(index=close.index, method="ffill")
    return build_field_registry(
        catalog_paths=[],
        runtime_numeric_fields={"close": close, "volume": volume, "pe_ratio": close * 0.1},
        runtime_group_fields={"sector": group},
        category_weights={"price": 1.0, "volume": 0.8, "fundamental": 0.9, "group": 0.6, "other": 0.5},
        score_weights=FieldScoreWeights(),
    )


def build_compatibility_field_registry() -> FieldRegistry:
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
            "vwap": spec("vwap", "price"),
            "volume": spec("volume", "volume"),
            "assets": spec("assets", "fundamental"),
            "sales": spec("sales", "fundamental"),
            "analyst_score": spec("analyst_score", "analyst"),
            "model_score": spec("model_score", "model"),
            "beta": spec("beta", "risk"),
            "adv20": spec("adv20", "liquidity"),
        }
    )


def test_genome_rendering_is_consistent_and_valid() -> None:
    generation, adaptive = build_configs()
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=21,
    )
    grammar = MotifGrammar()

    genome = builder.build_random_genome(source_mode="test")
    rendered = grammar.render(genome)
    reparsed = parse_expression(rendered.expression)

    assert rendered.normalized_expression == to_expression(reparsed)
    validation = validate_expression(
        node=reparsed,
        registry=registry,
        allowed_fields=field_registry.allowed_runtime_fields(generation.allowed_fields) | {"sector"},
        max_depth=generation.max_depth,
        group_fields={"sector"},
        field_types=field_registry.field_types(allowed=field_registry.allowed_runtime_fields(generation.allowed_fields) | {"sector"}),
        complexity_limit=generation.complexity_limit,
    )
    assert validation.is_valid


def test_secondary_field_prefers_compatible_categories_for_price_primary(monkeypatch) -> None:
    generation, adaptive = build_configs()
    generation = replace(
        generation,
        allowed_fields=["close", "vwap", "volume", "assets", "sales", "analyst_score", "model_score", "beta", "adv20"],
    )
    field_registry = build_compatibility_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=17,
    )
    primary = field_registry.get("close")
    captured: dict[str, float] = {}

    def fake_choices(candidates, weights, k):
        del k
        captured.update({candidate.name: weight for candidate, weight in zip(candidates, weights, strict=True)})
        return [candidates[0]]

    monkeypatch.setattr(builder.random, "choices", fake_choices)

    builder._pick_secondary_field(  # noqa: SLF001
        primary,
        motif="spread",
        novelty_bias=False,
        case_snapshot=None,
        diversity_tracker=None,
    )

    assert COMPATIBLE_CATEGORIES["price"] >= {"price", "volume", "fundamental"}
    assert captured["vwap"] == pytest.approx(1.25 * 1.5)
    assert captured["volume"] == pytest.approx(1.5)
    assert captured["assets"] == pytest.approx(1.5)
    assert captured["adv20"] == pytest.approx(0.3)
    assert captured["volume"] / captured["adv20"] == pytest.approx(5.0)
    assert captured["assets"] / captured["adv20"] == pytest.approx(5.0)


def test_secondary_field_prefers_compatible_categories_for_fundamental_primary(monkeypatch) -> None:
    generation, adaptive = build_configs()
    generation = replace(
        generation,
        allowed_fields=["close", "vwap", "volume", "assets", "sales", "analyst_score", "model_score", "beta", "adv20"],
    )
    field_registry = build_compatibility_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=18,
    )
    primary = field_registry.get("assets")
    captured: dict[str, float] = {}

    def fake_choices(candidates, weights, k):
        del k
        captured.update({candidate.name: weight for candidate, weight in zip(candidates, weights, strict=True)})
        return [candidates[0]]

    monkeypatch.setattr(builder.random, "choices", fake_choices)

    builder._pick_secondary_field(  # noqa: SLF001
        primary,
        motif="spread",
        novelty_bias=False,
        case_snapshot=None,
        diversity_tracker=None,
    )

    assert COMPATIBLE_CATEGORIES["fundamental"] >= {"fundamental", "analyst", "price"}
    assert captured["sales"] == pytest.approx(1.25 * 1.5)
    assert captured["analyst_score"] == pytest.approx(1.5)
    assert captured["close"] == pytest.approx(1.5)
    assert captured["model_score"] == pytest.approx(1.5)
    assert captured["volume"] == pytest.approx(0.3)
    assert captured["analyst_score"] / captured["volume"] == pytest.approx(5.0)
    assert captured["close"] / captured["volume"] == pytest.approx(5.0)


def test_auxiliary_field_uses_soft_compatibility_without_hard_block(monkeypatch) -> None:
    generation, adaptive = build_configs()
    generation = replace(generation, allowed_fields=["close", "volume", "analyst_score", "adv20"])
    field_registry = FieldRegistry(
        fields={
            "close": FieldSpec(
                name="close",
                dataset="runtime",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="price",
                runtime_available=True,
                field_score=1.0,
                category_weight=1.0,
            ),
            "volume": FieldSpec(
                name="volume",
                dataset="runtime",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="volume",
                runtime_available=True,
                field_score=1.0,
                category_weight=1.0,
            ),
            "analyst_score": FieldSpec(
                name="analyst_score",
                dataset="runtime",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="analyst",
                runtime_available=True,
                field_score=1.0,
                category_weight=1.0,
            ),
            "adv20": FieldSpec(
                name="adv20",
                dataset="runtime",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="liquidity",
                runtime_available=True,
                field_score=1.0,
                category_weight=1.0,
            ),
        }
    )
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=19,
    )
    primary = field_registry.get("volume")
    secondary = field_registry.get("close")
    captured: dict[str, float] = {}

    def fake_choices(candidates, weights, k):
        del k
        captured.update({candidate.name: weight for candidate, weight in zip(candidates, weights, strict=True)})
        return [next(candidate for candidate in candidates if candidate.name == "analyst_score")]

    monkeypatch.setattr(builder.random, "choices", fake_choices)

    selected = builder._pick_auxiliary_field(primary, secondary, novelty_bias=False)  # noqa: SLF001

    assert selected.name == "analyst_score"
    assert captured["adv20"] == pytest.approx(1.5)
    assert captured["analyst_score"] == pytest.approx(0.3)
    assert captured["analyst_score"] > 0.0


def test_motif_library_includes_new_entries() -> None:
    grammar = MotifGrammar()

    assert len(MOTIF_LIBRARY) == 12
    assert {"quality_score", "price_volume_divergence", "conditional_momentum"}.issubset(MOTIF_LIBRARY)
    assert grammar.specs["quality_score"].required_fields == 2
    assert grammar.specs["price_volume_divergence"].required_fields == 2
    assert grammar.specs["conditional_momentum"].required_fields == 2


@pytest.mark.parametrize(
    ("motif", "transform_gene", "expected_depth", "expected_fragment"),
    [
        (
            "quality_score",
            TransformGene(
                motif="quality_score",
                primitive_transform="",
                secondary_transform="",
            ),
            5,
            "rank((close/(abs(volume)+1)))",
        ),
        (
            "price_volume_divergence",
            TransformGene(
                motif="price_volume_divergence",
                primitive_transform="ts_corr",
                secondary_transform="",
            ),
            3,
            "rank(ts_corr(close,volume,10))",
        ),
        (
            "conditional_momentum",
            TransformGene(
                motif="conditional_momentum",
                primitive_transform="ts_delta",
                secondary_transform="ts_mean",
            ),
            4,
            "(ts_delta(close,2)*rank(ts_mean(volume,10)))",
        ),
    ],
)
def test_new_motifs_render_to_parseable_expressions_and_match_expected_depth(
    motif: str,
    transform_gene: TransformGene,
    expected_depth: int,
    expected_fragment: str,
) -> None:
    generation, adaptive = build_configs()
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=77,
    )
    grammar = MotifGrammar()
    genome = Genome(
        feature_gene=FeatureGene(
            primary_field="close",
            primary_family="price",
            secondary_field="volume",
            secondary_family="volume",
            auxiliary_field="pe_ratio",
            auxiliary_family="other",
        ),
        transform_gene=transform_gene,
        horizon_gene=HorizonGene(fast_window=2, slow_window=10, context_window=15),
        source_mode=f"motif-{motif}",
    )

    rendered = grammar.render(genome)
    reparsed = parse_expression(rendered.expression)

    assert rendered.expression == expected_fragment
    assert node_depth(reparsed) == expected_depth
    assert builder._base_render_depth(motif) == expected_depth  # noqa: SLF001


def test_new_motif_builder_operator_defaults_are_stable() -> None:
    generation, adaptive = build_configs()
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=78,
    )

    assert builder._pick_primitive("quality_score", diversity_tracker=None) == ""  # noqa: SLF001
    assert builder._pick_primitive("price_volume_divergence", diversity_tracker=None) == "ts_corr"  # noqa: SLF001
    assert builder._pick_primitive("conditional_momentum", diversity_tracker=None) == "ts_delta"  # noqa: SLF001
    assert builder._pick_secondary_transform("quality_score", "", diversity_tracker=None) == ""  # noqa: SLF001
    assert builder._pick_secondary_transform("price_volume_divergence", "ts_corr", diversity_tracker=None) == ""  # noqa: SLF001
    assert builder._pick_secondary_transform("conditional_momentum", "ts_delta", diversity_tracker=None) == "ts_mean"  # noqa: SLF001


def test_builder_defaults_keep_existing_sim_behavior() -> None:
    generation, adaptive = build_configs()
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=79,
    )
    genome = builder.build_parent_seeded_genome(
        motif="momentum",
        primary_family="price",
        source_mode="sim-defaults",
    )

    assert generation.sim_neutralization == "none"
    assert generation.sim_decay == 0
    assert builder.sim_neutralization_active is False
    assert builder.sim_decay_active is False
    assert builder._smoothing_activation_threshold() == 0.15  # noqa: SLF001
    assert builder._pick_conditioning_mode("group_relative_signal") == "group_neutralize"  # noqa: SLF001
    assert genome.to_dict()["sim_neutralization_active"] is False
    assert genome.to_dict()["sim_decay_active"] is False


def test_group_relative_signal_conditioning_weight_drops_when_sim_neutralization_active(monkeypatch) -> None:
    generation, adaptive = build_configs()
    generation = replace(generation, sim_neutralization="sector")
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=80,
    )
    captured: dict[str, list[float] | list[str]] = {}

    def fake_choices(labels, weights, k):
        del k
        captured["labels"] = list(labels)
        captured["weights"] = list(weights)
        return ["none"]

    monkeypatch.setattr(builder.random, "choices", fake_choices)

    selected = builder._pick_conditioning_mode("group_relative_signal")  # noqa: SLF001
    labels = captured["labels"]
    weights = captured["weights"]
    group_index = labels.index("group_neutralize")

    assert builder.sim_neutralization_active is True
    assert selected == "none"
    assert weights[group_index] == 0.05


def test_sim_decay_raises_smoothing_threshold_and_tags_genome(monkeypatch) -> None:
    generation, adaptive = build_configs()
    decay_generation = replace(generation, sim_decay=5)
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)

    def build_builder(config: GenerationConfig, seed: int) -> GenomeBuilder:
        return GenomeBuilder(
            generation_config=config,
            adaptive_config=adaptive,
            registry=registry,
            field_registry=field_registry,
            seed=seed,
        )

    default_builder = build_builder(generation, 81)
    decay_builder = build_builder(decay_generation, 82)
    primary = field_registry.get("close")
    secondary = field_registry.get("volume")
    auxiliary = field_registry.get("pe_ratio")

    def patch_builder(builder: GenomeBuilder) -> None:
        monkeypatch.setattr(builder, "_pick_numeric_field", lambda **kwargs: primary)
        monkeypatch.setattr(builder, "_pick_secondary_field", lambda *args, **kwargs: secondary)
        monkeypatch.setattr(builder, "_pick_auxiliary_field", lambda *args, **kwargs: auxiliary)
        monkeypatch.setattr(builder, "_pick_windows", lambda **kwargs: (2, 10, 15))
        monkeypatch.setattr(builder, "_pick_primitive", lambda *args, **kwargs: "ts_delta")
        monkeypatch.setattr(builder, "_pick_secondary_transform", lambda *args, **kwargs: "")
        monkeypatch.setattr(builder, "_pick_conditioning_mode", lambda *args, **kwargs: "none")
        monkeypatch.setattr(builder, "_pick_smoothing_operator", lambda: "ts_mean")
        monkeypatch.setattr(builder, "_pick_wrappers", lambda **kwargs: ())
        monkeypatch.setattr(builder, "_estimate_turnover_hint", lambda *args, **kwargs: 0.20)
        monkeypatch.setattr(builder, "_constrain_by_actual_depth", lambda genome: genome)

    patch_builder(default_builder)
    patch_builder(decay_builder)

    default_genome = default_builder.build_parent_seeded_genome(
        motif="momentum",
        primary_family="price",
        source_mode="default-smoothing",
    )
    decay_genome = decay_builder.build_parent_seeded_genome(
        motif="momentum",
        primary_family="price",
        source_mode="decay-smoothing",
    )

    assert default_builder._smoothing_activation_threshold() == 0.15  # noqa: SLF001
    assert decay_builder._smoothing_activation_threshold() == 0.30  # noqa: SLF001
    assert default_genome.turnover_gene.smoothing_operator == "ts_mean"
    assert decay_genome.turnover_gene.smoothing_operator == ""
    assert decay_genome.to_dict()["sim_neutralization_active"] is False
    assert decay_genome.to_dict()["sim_decay_active"] is True


def test_genome_builder_actual_depth_preflight_strips_underestimated_layers() -> None:
    generation, adaptive = build_configs()
    generation = replace(generation, max_depth=4)
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=21,
    )
    grammar = MotifGrammar()

    base = builder.build_parent_seeded_genome(
        motif="group_relative_signal",
        primary_family="price",
        source_mode="depth-regression",
    )
    stressed = replace(
        base,
        wrapper_gene=replace(base.wrapper_gene, pre_wrappers=(), post_wrappers=()),
        regime_gene=replace(base.regime_gene, conditioning_mode="group_neutralize", conditioning_field=""),
        turnover_gene=replace(base.turnover_gene, smoothing_operator="ts_mean", smoothing_window=10),
    )

    assert builder._estimate_render_depth(stressed) == generation.max_depth  # noqa: SLF001
    assert node_depth(parse_expression(grammar.render(stressed).expression)) > generation.max_depth

    constrained = builder._enforce_actual_depth_limit(stressed)  # noqa: SLF001

    assert constrained.turnover_gene.smoothing_operator == ""
    assert node_depth(parse_expression(grammar.render(constrained).expression)) <= generation.max_depth


def test_genome_builder_actual_depth_preflight_falls_back_on_render_error(monkeypatch) -> None:
    generation, adaptive = build_configs()
    generation = replace(generation, max_depth=4)
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=31,
    )
    genome = builder.build_parent_seeded_genome(
        motif="group_relative_signal",
        primary_family="price",
        source_mode="depth-fallback",
    )

    monkeypatch.setattr(builder._grammar, "render", lambda genome: (_ for _ in ()).throw(RuntimeError("boom")))

    assert builder._enforce_actual_depth_limit(genome) is genome  # noqa: SLF001


def test_crossover_produces_renderable_child() -> None:
    generation, adaptive = build_configs()
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=33,
    )
    grammar = MotifGrammar()
    crossover = GenomeCrossover(random.Random(5))

    left = builder.build_parent_seeded_genome(motif="momentum", primary_family="price", source_mode="left")
    right = builder.build_parent_seeded_genome(motif="group_relative_signal", primary_family="price", source_mode="right")
    child = crossover.crossover(left, right)
    rendered = grammar.render(child)

    assert child.motif in {"momentum", "group_relative_signal"}
    assert rendered.expression
    assert rendered.genome.stable_hash


def test_novelty_scoring_and_distance_are_symmetric() -> None:
    generation, adaptive = build_configs()
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=44,
    )
    grammar = MotifGrammar()
    novelty = NoveltySearch()
    memory_service = PatternMemoryService()

    first = grammar.render(builder.build_parent_seeded_genome(motif="momentum", primary_family="price", source_mode="a"))
    second = grammar.render(builder.build_parent_seeded_genome(motif="spread", primary_family="price", source_mode="b"))
    first_signature = memory_service.extract_signature(first.expression, generation_metadata={"motif": first.genome.motif})
    second_signature = memory_service.extract_signature(second.expression, generation_metadata={"motif": second.genome.motif})

    assert structural_distance(first_signature, first_signature) == 0.0
    assert structural_distance(first_signature, second_signature) == structural_distance(second_signature, first_signature)
    score = novelty.score(first_signature, [second_signature])
    assert 0.0 <= score.score <= 1.0
    assert score.min_distance > 0.0


def test_repair_policy_reduces_turnover_and_complexity_pressure() -> None:
    generation, adaptive = build_configs()
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=55,
    )
    grammar = MotifGrammar()
    repair = RepairPolicy(
        generation_config=generation,
        repair_config=adaptive.repair_policy,
        field_registry=field_registry,
        registry=registry,
    )

    base = builder.build_parent_seeded_genome(motif="volatility_adjusted_momentum", primary_family="price", source_mode="repair")
    stressed = replace(
        base,
        wrapper_gene=replace(base.wrapper_gene, pre_wrappers=("rank", "rank"), post_wrappers=("zscore", "sign")),
        turnover_gene=replace(base.turnover_gene, smoothing_operator="", smoothing_window=0, turnover_hint=0.90),
        complexity_gene=replace(base.complexity_gene, target_depth=6, binary_branching=2, wrapper_budget=3),
    )
    repaired, actions = repair.repair(stressed, fail_tags=("high_turnover", "excessive_complexity"))
    rendered = grammar.render(repaired)

    assert actions
    assert repaired.turnover_gene.turnover_hint <= 0.0
    assert repaired.complexity_gene.binary_branching == 1
    assert rendered.expression


def test_diversity_selection_keeps_exploration_quota() -> None:
    generation, adaptive = build_configs()
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    engine = AlphaGenerationEngine(
        config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
    )
    selector = CandidateSelectionService()

    exploit = engine.generate(count=4)
    novelty = []
    for candidate in engine.generate(count=2):
        candidate.generation_metadata["mutation_mode"] = "novelty"
        novelty.append(replace(candidate, generation_mode="novelty"))
    diversity = replace(
        adaptive.diversity,
        max_family_fraction=1.0,
        max_field_category_fraction=1.0,
        max_horizon_bucket_fraction=1.0,
        max_operator_path_fraction=1.0,
        exploration_quota_fraction=0.34,
    )
    selected, _ = selector.select_for_simulation(
        [*exploit, *novelty],
        snapshot=PatternMemorySnapshot(regime_key="selection"),
        field_registry=field_registry,
        batch_size=3,
        min_pattern_support=1,
        rejection_filters=[],
        diversity_config=diversity,
    )

    assert len(selected) == 3
    assert any(item.candidate.generation_mode == "novelty" for item in selected)


def test_engine_generation_avoids_excessive_duplicates() -> None:
    generation, adaptive = build_configs()
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    engine = AlphaGenerationEngine(
        config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
    )

    candidates = engine.generate(count=15)
    family_signatures = [candidate.generation_metadata.get("family_signature", "") for candidate in candidates]

    assert len(candidates) == 15
    assert len({candidate.normalized_expression for candidate in candidates}) == len(candidates)
    assert len({signature for signature in family_signatures if signature}) >= 5


def test_pick_motif_applies_minimum_weight_floor(monkeypatch) -> None:
    generation, adaptive = build_configs()
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=63,
    )
    tracker = GenerationDiversityTracker()
    captured: dict[str, list[float]] = {}

    class Snapshot:
        def stats_for_scope(self, category: str, scope: str = "blended") -> dict[str, SimpleNamespace]:
            assert category == "motif"
            return {
                "spread": SimpleNamespace(support=100, avg_outcome=2.0, success_rate=0.9),
            }

    def fake_choices(labels, weights, k):
        del k
        captured["weights"] = list(weights)
        return [labels[0]]

    monkeypatch.setattr(builder.random, "choices", fake_choices)

    builder._pick_motif(  # noqa: SLF001
        novelty_bias=False,
        case_snapshot=Snapshot(),
        diversity_tracker=tracker,
    )

    assert captured["weights"]
    assert min(captured["weights"]) >= (max(captured["weights"]) * 0.05) - 1e-12
    assert sum(tracker.sampled_motifs.values()) == 1


def test_pick_motif_repicks_after_domination_penalty(monkeypatch) -> None:
    generation, adaptive = build_configs()
    field_registry = build_runtime_field_registry()
    registry = build_registry(generation.allowed_operators)
    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=64,
    )
    tracker = GenerationDiversityTracker()
    motifs = list(MOTIF_LIBRARY)
    dominant = motifs[0]
    alternate = motifs[1]
    tracker.sampled_motifs[dominant] = len(motifs) * 3 + 1
    calls: list[list[float]] = []

    def fake_choices(labels, weights, k):
        del k
        calls.append(list(weights))
        return [dominant if len(calls) == 1 else alternate]

    monkeypatch.setattr(builder.random, "choices", fake_choices)

    selected = builder._pick_motif(  # noqa: SLF001
        novelty_bias=False,
        case_snapshot=None,
        diversity_tracker=tracker,
    )

    dominant_index = motifs.index(dominant)
    assert selected == alternate
    assert len(calls) == 2
    assert abs(calls[1][dominant_index] - (calls[0][dominant_index] * 0.3)) < 1e-12
    assert tracker.sampled_motif_count(dominant) == len(motifs) * 3 + 1
    assert tracker.sampled_motif_count(alternate) == 1
