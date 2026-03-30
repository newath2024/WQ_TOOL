from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha.parser import parse_expression
from alpha.validator import validate_expression
from core.config import GenerationConfig
from data.field_registry import FieldScoreWeights, build_field_registry, load_field_catalog, load_runtime_field_values
from features.registry import build_registry
from generator.engine import AlphaGenerationEngine
from tests.conftest import build_sample_market_frame


def test_field_catalog_scoring_and_runtime_merge(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id": "pe_ratio",
                        "dataset": {"name": "Fundamental"},
                        "category": {"name": "Fundamental"},
                        "type": "MATRIX",
                        "coverage": 0.95,
                        "alphaCount": 100,
                    },
                    {
                        "id": "close",
                        "dataset": {"name": "Price Volume"},
                        "category": {"name": "Price"},
                        "type": "MATRIX",
                        "coverage": 1.0,
                        "alphaCount": 1000,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    market = build_sample_market_frame()
    close = market.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    registry = build_field_registry(
        catalog_paths=[str(catalog_path)],
        runtime_numeric_fields={"close": close, "pe_ratio": close * 0.1},
        runtime_group_fields={},
        category_weights={"price": 1.0, "fundamental": 0.9, "other": 0.5},
        score_weights=FieldScoreWeights(),
    )

    assert registry.contains("close")
    assert registry.contains("pe_ratio")
    assert registry.get("close").runtime_available
    assert registry.get("pe_ratio").runtime_available
    assert registry.get("close").field_score >= registry.get("pe_ratio").field_score


def test_long_csv_field_values_are_loaded_as_numeric_and_group_matrices(tmp_path: Path) -> None:
    values_path = tmp_path / "field_values.csv"
    rows = [
        {"timestamp": "2021-01-01", "symbol": "AAA", "field": "pe_ratio", "value": 10.5, "field_type": "matrix", "category": "fundamental"},
        {"timestamp": "2021-01-01", "symbol": "BBB", "field": "pe_ratio", "value": 12.0, "field_type": "matrix", "category": "fundamental"},
        {"timestamp": "2021-01-01", "symbol": "AAA", "field": "sector_label", "value": "technology", "field_type": "vector", "category": "group"},
        {"timestamp": "2021-01-01", "symbol": "BBB", "field": "sector_label", "value": "financials", "field_type": "vector", "category": "group"},
    ]
    pd.DataFrame(rows).to_csv(values_path, index=False)

    bundle = load_runtime_field_values([str(values_path)], default_timeframe="1d")
    numeric_fields, group_fields = bundle.for_timeframe("1d")

    assert "pe_ratio" in numeric_fields
    assert "sector_label" in group_fields
    assert float(numeric_fields["pe_ratio"].loc[pd.Timestamp("2021-01-01"), "AAA"]) == 10.5
    assert group_fields["sector_label"].loc[pd.Timestamp("2021-01-01"), "BBB"] == "financials"


def test_alpha_factory_generates_structured_candidates_with_metadata() -> None:
    market = build_sample_market_frame()
    close = market.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    volume = market.pivot(index="timestamp", columns="symbol", values="volume").sort_index()
    group = pd.DataFrame(
        [["technology", "technology", "financials", "financials"]],
        index=[close.index[0]],
        columns=close.columns,
    ).reindex(index=close.index, method="ffill")
    field_registry = build_field_registry(
        catalog_paths=[],
        runtime_numeric_fields={"close": close, "volume": volume, "pe_ratio": close * 0.1},
        runtime_group_fields={"sector": group},
        category_weights={"price": 1.0, "volume": 0.8, "fundamental": 0.9, "group": 0.6, "other": 0.5},
        score_weights=FieldScoreWeights(),
    )
    config = GenerationConfig(
        allowed_fields=["close", "volume", "pe_ratio"],
        allowed_operators=[
            "rank",
            "ts_delta",
            "ts_mean",
            "ts_std_dev",
            "group_neutralize",
            "ts_decay_linear",
            "ts_corr",
            "ts_covariance",
        ],
        lookbacks=[2, 5, 10],
        max_depth=5,
        complexity_limit=20,
        template_count=6,
        grammar_count=0,
        mutation_count=4,
        normalization_wrappers=["rank", "zscore", "sign"],
        random_seed=7,
        template_pool_size=20,
    )
    engine = AlphaGenerationEngine(
        config=config,
        registry=build_registry(config.allowed_operators),
        field_registry=field_registry,
    )

    candidates = engine.generate(count=5)

    assert len(candidates) == 5
    assert all(candidate.template_name for candidate in candidates)
    assert all(candidate.fields_used for candidate in candidates)
    assert all(candidate.operators_used for candidate in candidates)
    assert all(candidate.depth > 0 for candidate in candidates)
    assert all({"delta", "correlation", "covariance", "decay_linear", "ts_std"} & set(candidate.operators_used) == set() for candidate in candidates)


def test_field_catalog_directory_loads_and_prefers_matching_brain_market(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    pd.DataFrame(
        [
            {
                "id": "close",
                "dataset": "Price Volume",
                "category": "Price",
                "type": "MATRIX",
                "coverage": 1.0,
                "alphaCount": 100,
                "region": "IND",
                "universe": "TOP500",
                "delay": 1,
            },
            {
                "id": "close",
                "dataset": "Price Volume",
                "category": "Price",
                "type": "MATRIX",
                "coverage": 0.9,
                "alphaCount": 50,
                "region": "GLB",
                "universe": "TOP3000",
                "delay": 1,
            },
            {
                "id": "cap",
                "dataset": "Price Volume",
                "category": "Price",
                "type": "MATRIX",
                "coverage": 1.0,
                "alphaCount": 10,
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
            },
        ]
    ).to_csv(catalog_dir / "market.csv", index=False)

    fields = load_field_catalog(
        [str(catalog_dir)],
        category_weights={"price": 1.0, "other": 0.5},
        score_weights=FieldScoreWeights(),
        preferred_region="USA",
        preferred_universe="TOP3000",
        preferred_delay=1,
    )

    assert fields["close"].region == "GLB"
    assert fields["close"].universe == "TOP3000"
    assert fields["cap"].region == "USA"


def test_alpha_factory_can_generate_catalog_only_candidates_when_enabled(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        [
            {
                "id": "close",
                "dataset": "Price Volume",
                "category": "Price",
                "type": "MATRIX",
                "coverage": 1.0,
                "alphaCount": 1000,
                "region": "GLB",
                "universe": "TOP3000",
                "delay": 1,
            },
            {
                "id": "adv20",
                "dataset": "Price Volume",
                "category": "Volume",
                "type": "MATRIX",
                "coverage": 1.0,
                "alphaCount": 800,
                "region": "GLB",
                "universe": "TOP3000",
                "delay": 1,
            },
            {
                "id": "cap",
                "dataset": "Price Volume",
                "category": "Price",
                "type": "MATRIX",
                "coverage": 1.0,
                "alphaCount": 600,
                "region": "GLB",
                "universe": "TOP3000",
                "delay": 1,
            },
            {
                "id": "sector",
                "dataset": "Universe",
                "category": "Group",
                "type": "VECTOR",
                "coverage": 1.0,
                "alphaCount": 50,
                "region": "GLB",
                "universe": "TOP3000",
                "delay": 1,
            },
        ]
    ).to_csv(catalog_path, index=False)

    field_registry = build_field_registry(
        catalog_paths=[str(catalog_path)],
        runtime_numeric_fields={},
        runtime_group_fields={},
        category_weights={"price": 1.0, "volume": 0.8, "group": 0.6, "other": 0.5},
        score_weights=FieldScoreWeights(),
        preferred_region="USA",
        preferred_universe="TOP3000",
        preferred_delay=1,
    )
    config = GenerationConfig(
        allowed_fields=[],
        allowed_operators=[
            "rank",
            "ts_delta",
            "ts_mean",
            "ts_std_dev",
            "group_neutralize",
            "ts_decay_linear",
            "ts_corr",
            "ts_covariance",
        ],
        lookbacks=[2, 5, 10],
        max_depth=5,
        complexity_limit=20,
        template_count=4,
        grammar_count=0,
        mutation_count=2,
        normalization_wrappers=["rank", "zscore", "sign"],
        random_seed=11,
        template_pool_size=12,
        allow_catalog_fields_without_runtime=True,
    )
    engine = AlphaGenerationEngine(
        config=config,
        registry=build_registry(config.allowed_operators),
        field_registry=field_registry,
    )

    candidates = engine.generate(count=3)

    assert len(candidates) == 3
    assert all(candidate.fields_used for candidate in candidates)
    assert all(
        field_registry.contains(field_name)
        for candidate in candidates
        for field_name in candidate.fields_used
    )
    assert all(not field_registry.get(field_name).runtime_available for candidate in candidates for field_name in candidate.fields_used)


def test_typed_validator_rejects_redundant_wrapper_chain() -> None:
    registry = build_registry(["rank"])
    node = parse_expression("rank(rank(close))")

    result = validate_expression(
        node=node,
        registry=registry,
        allowed_fields={"close"},
        max_depth=4,
        field_types={"close": "matrix"},
    )

    assert not result.is_valid
    assert any("Redundant nested wrapper" in error for error in result.errors)
