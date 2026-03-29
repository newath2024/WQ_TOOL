from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from alpha.parser import parse_expression
from alpha.validator import validate_expression
from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldScoreWeights, build_field_registry
from features.registry import build_registry
from generator.repair_policy import RepairPolicy
from tests.conftest import build_sample_market_frame


def _write_operator_catalog(path: Path) -> Path:
    payload = {
        "operator_count": 3,
        "operators": [
            {
                "signature": "ts_decay_linear(x, d, dense = false)",
                "name": "ts_decay_linear",
                "tier": "base",
                "scope": "Combo, Regular",
                "summary": "Applies a linear decay to time-series data over a set number of days, smoothing recent values.",
                "details": "This operator may help reduce turnover and make your strategy more stable across days.",
            },
            {
                "signature": "log(x)",
                "name": "log",
                "tier": "base",
                "scope": "Combo, Regular, Selection",
                "summary": "Calculates the natural logarithm of the input value.",
                "details": "The input x should be positive, as the logarithm is undefined for zero or negative values.",
            },
            {
                "signature": "group_rank(x, group)",
                "name": "group_rank",
                "tier": "expert",
                "scope": "Regular",
                "summary": "Ranks values within each group.",
                "details": "Useful for sector-relative ranking and other group-relative signals.",
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build_field_registry():
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
        runtime_numeric_fields={"close": close, "volume": volume},
        runtime_group_fields={"sector": group},
        category_weights={"price": 1.0, "volume": 0.8, "group": 0.6, "other": 0.5},
        score_weights=FieldScoreWeights(),
    )


def test_registry_enriches_specs_from_operator_catalog(tmp_path: Path) -> None:
    catalog_path = _write_operator_catalog(tmp_path / "operators.json")
    registry = build_registry(
        ["ts_decay_linear", "log", "group_rank"],
        operator_catalog_paths=[str(catalog_path)],
    )

    decay = registry.get("ts_decay_linear")
    assert decay.description.startswith("Applies a linear decay")
    assert decay.has_tag("smoothing")
    assert decay.has_tag("reduces_turnover")
    assert decay.prefers_motif("mean_reversion")
    assert decay.parameter_requirements["window"] == "positive_int"

    log_spec = registry.get("log")
    assert log_spec.has_tag("requires_positive_input")
    assert "positive" in log_spec.catalog_details.lower()

    group_rank = registry.get("group_rank")
    assert group_rank.has_tag("group_aware")
    assert group_rank.prefers_motif("group_relative_signal")


def test_repair_policy_prefers_catalog_aware_smoothing_operator(tmp_path: Path) -> None:
    catalog_path = _write_operator_catalog(tmp_path / "operators.json")
    generation = GenerationConfig(
        allowed_fields=["close", "volume"],
        allowed_operators=["ts_delta", "ts_mean", "ts_decay_linear", "rank", "group_neutralize"],
        lookbacks=[2, 5, 10, 15],
        max_depth=6,
        complexity_limit=24,
        template_count=4,
        grammar_count=4,
        mutation_count=2,
        normalization_wrappers=["rank"],
        random_seed=17,
        operator_catalog_paths=[str(catalog_path)],
    )
    adaptive = AdaptiveGenerationConfig()
    field_registry = _build_field_registry()
    registry = build_registry(
        generation.allowed_operators,
        operator_catalog_paths=generation.operator_catalog_paths,
    )

    from generator.genome_builder import GenomeBuilder

    builder = GenomeBuilder(
        generation_config=generation,
        adaptive_config=adaptive,
        registry=registry,
        field_registry=field_registry,
        seed=99,
    )
    repair = RepairPolicy(
        generation_config=generation,
        repair_config=adaptive.repair_policy,
        field_registry=field_registry,
        registry=registry,
    )

    stressed = builder.build_parent_seeded_genome(
        motif="momentum",
        primary_family="price",
        source_mode="repair-test",
    )
    stressed = replace(
        stressed,
        turnover_gene=replace(stressed.turnover_gene, smoothing_operator="", smoothing_window=0, turnover_hint=0.95),
    )
    repaired, actions = repair.repair(stressed, fail_tags=("high_turnover",))

    assert "reduce_turnover" in actions
    assert repaired.turnover_gene.smoothing_operator == "ts_decay_linear"


def test_validator_uses_catalog_positive_input_constraint(tmp_path: Path) -> None:
    catalog_path = _write_operator_catalog(tmp_path / "operators.json")
    registry = build_registry(["log"], operator_catalog_paths=[str(catalog_path)])
    validation = validate_expression(
        node=parse_expression("log(-1)"),
        registry=registry,
        allowed_fields=set(),
        max_depth=5,
        group_fields=set(),
        field_types={},
        complexity_limit=10,
    )

    assert any("positive input" in error.lower() for error in validation.errors)
