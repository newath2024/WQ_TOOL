from __future__ import annotations

import random
from dataclasses import replace

import pandas as pd

from alpha.ast_nodes import to_expression
from alpha.parser import parse_expression
from alpha.validator import validate_expression
from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldScoreWeights, build_field_registry
from evaluation.alpha_distance import structural_distance
from features.registry import build_registry
from generator.crossover import GenomeCrossover
from generator.engine import AlphaGenerationEngine
from generator.genome_builder import GenomeBuilder
from generator.grammar import MotifGrammar
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
