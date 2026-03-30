from __future__ import annotations

import json
from dataclasses import replace

from core.config import AdaptiveGenerationConfig, GenerationConfig, load_config
from core.run_context import RunContext
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import build_registry
from generator.engine import AlphaCandidate, AlphaGenerationEngine, CandidateBuildResult
from generator.guided_generator import GuidedGenerator
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot
from services.brain_batch_service import BrainBatchService
from services.data_service import CachedResearchContextProvider
from services.runtime_service import build_command_environment, init_run
from storage.repository import SQLiteRepository


def _build_generation_config() -> GenerationConfig:
    return GenerationConfig(
        allowed_fields=["close", "volume", "returns"],
        allowed_operators=["rank", "ts_mean", "ts_delta", "zscore", "sign"],
        lookbacks=[2, 3, 5],
        max_depth=5,
        complexity_limit=20,
        template_count=6,
        grammar_count=6,
        mutation_count=4,
        normalization_wrappers=["rank", "zscore", "sign"],
        random_seed=11,
    )


def _build_field_registry() -> FieldRegistry:
    return FieldRegistry(
        fields={
            name: FieldSpec(
                name=name,
                dataset="test",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=10,
                category="price" if name != "volume" else "volume",
                runtime_available=True,
                field_score=0.8,
                category_weight=0.8,
            )
            for name in ["close", "volume", "returns"]
        }
    )


def _build_guided_generator(adaptive_config: AdaptiveGenerationConfig | None = None) -> GuidedGenerator:
    config = _build_generation_config()
    adaptive = adaptive_config or AdaptiveGenerationConfig()
    return GuidedGenerator(
        generation_config=config,
        adaptive_config=adaptive,
        registry=build_registry(config.allowed_operators),
        memory_service=PatternMemoryService(),
        field_registry=_build_field_registry(),
    )


def _environment(config_path: str, run_id: str):
    context = RunContext.create(seed=7, config_path=config_path, run_id=run_id)
    return build_command_environment(config_path=config_path, command_name="benchmark-generation", context=context)


def test_engine_validation_context_cache_reuses_field_resolution(monkeypatch) -> None:
    config = _build_generation_config()
    field_registry = _build_field_registry()
    registry = build_registry(config.allowed_operators)
    engine = AlphaGenerationEngine(
        config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=registry,
        field_registry=field_registry,
    )
    counts = {"numeric": 0, "group": 0, "types": 0}
    field_registry_type = type(field_registry)
    original_numeric = field_registry_type.generation_numeric_fields
    original_group = field_registry_type.generation_group_fields
    original_types = field_registry_type.field_types

    def counted_numeric(self, *args, **kwargs):
        counts["numeric"] += 1
        return original_numeric(self, *args, **kwargs)

    def counted_group(self, *args, **kwargs):
        counts["group"] += 1
        return original_group(self, *args, **kwargs)

    def counted_types(self, *args, **kwargs):
        counts["types"] += 1
        return original_types(self, *args, **kwargs)

    monkeypatch.setattr(field_registry_type, "generation_numeric_fields", counted_numeric)
    monkeypatch.setattr(field_registry_type, "generation_group_fields", counted_group)
    monkeypatch.setattr(field_registry_type, "field_types", counted_types)

    assert engine.build_candidate("rank(close)", mode="test", parent_ids=()) is not None
    assert engine.build_candidate("rank(volume)", mode="test", parent_ids=()) is not None
    assert counts == {"numeric": 1, "group": 1, "types": 1}


def test_engine_validation_context_cache_invalidates_when_field_registry_changes() -> None:
    config = _build_generation_config()
    engine = AlphaGenerationEngine(
        config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(config.allowed_operators),
        field_registry=_build_field_registry(),
    )

    first = engine.prepare_validation_context()
    second_registry = _build_field_registry()
    second_registry.fields["beta"] = FieldSpec(
        name="beta",
        dataset="test",
        field_type="matrix",
        coverage=1.0,
        alpha_usage_count=1,
        category="risk",
        runtime_available=True,
        field_score=0.6,
        category_weight=0.6,
    )
    engine.field_registry = second_registry
    second = engine.prepare_validation_context()

    assert first is not second
    assert "beta" in second.field_categories


def test_candidate_build_result_classifies_failure_reasons() -> None:
    config = _build_generation_config()
    engine = AlphaGenerationEngine(
        config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(config.allowed_operators),
        field_registry=_build_field_registry(),
    )

    parse_failed = engine._build_candidate_result("rank(", mode="test", parent_ids=())  # noqa: SLF001
    disallowed = engine._build_candidate_result("rank(beta)", mode="test", parent_ids=())  # noqa: SLF001
    redundant = engine._build_candidate_result("rank(rank(close))", mode="test", parent_ids=())  # noqa: SLF001

    assert parse_failed.failure_reason == "parse_failed"
    assert disallowed.failure_reason == "disallowed_field"
    assert redundant.failure_reason == "redundant_expression"


def test_guided_generator_stops_on_time_budget(monkeypatch) -> None:
    adaptive = AdaptiveGenerationConfig(
        max_generation_seconds=1.0,
        max_attempt_multiplier=50,
        max_consecutive_failures=100,
        min_candidates_before_early_exit=1,
    )
    generator = _build_guided_generator(adaptive)
    fake_render = type("FakeRender", (), {"expression": "rank(close)"})()
    monotonic_values = iter([0.0, 0.0, 0.2, 1.5, 1.5, 1.5])

    def fake_monotonic():
        return next(monotonic_values, 1.5)

    monkeypatch.setattr("generator.guided_generator.time.monotonic", fake_monotonic)
    monkeypatch.setattr("generator.engine.time.monotonic", fake_monotonic)
    monkeypatch.setattr(generator.genome_builder, "build_guided_genome", lambda **kwargs: object())
    monkeypatch.setattr(generator.repair_policy, "repair", lambda genome: (genome, ()))
    monkeypatch.setattr(generator.grammar, "render", lambda genome: fake_render)
    monkeypatch.setattr(generator.base_engine, "_render_metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        generator.base_engine,
        "_build_candidate_result",
        lambda *args, **kwargs: CandidateBuildResult(candidate=None, failure_reason="expression_validation_failed"),
    )

    candidates = generator.generate(
        count=5,
        snapshot=PatternMemorySnapshot(regime_key="budget"),
    )

    assert candidates == []
    assert generator.last_generation_stats is not None
    assert generator.last_generation_stats.timeout_stop is True
    assert generator.last_generation_stats.attempt_count == 1


def test_guided_generator_stops_after_consecutive_failures_and_returns_partial_candidates(monkeypatch) -> None:
    adaptive = AdaptiveGenerationConfig(
        max_generation_seconds=20.0,
        max_attempt_multiplier=50,
        max_consecutive_failures=4,
        min_candidates_before_early_exit=1,
    )
    generator = _build_guided_generator(adaptive)
    fake_render = type("FakeRender", (), {"expression": "rank(close)"})()
    valid_candidate = AlphaCandidate(
        alpha_id="alpha-1",
        expression="rank(close)",
        normalized_expression="rank(close)",
        generation_mode="guided_exploit",
        parent_ids=(),
        complexity=2,
        created_at="2026-03-30T00:00:00+00:00",
    )
    results = iter(
        [
            CandidateBuildResult(candidate=valid_candidate),
            CandidateBuildResult(candidate=None, failure_reason="expression_validation_failed"),
            CandidateBuildResult(candidate=None, failure_reason="expression_validation_failed"),
            CandidateBuildResult(candidate=None, failure_reason="expression_validation_failed"),
        ]
    )

    monkeypatch.setattr(generator.genome_builder, "build_guided_genome", lambda **kwargs: object())
    monkeypatch.setattr(generator.repair_policy, "repair", lambda genome: (genome, ()))
    monkeypatch.setattr(generator.grammar, "render", lambda genome: fake_render)
    monkeypatch.setattr(generator.base_engine, "_render_metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(generator.base_engine, "_build_candidate_result", lambda *args, **kwargs: next(results))

    candidates = generator.generate(
        count=5,
        snapshot=PatternMemorySnapshot(regime_key="fail-fast"),
    )

    assert len(candidates) == 1
    assert generator.last_generation_stats is not None
    assert generator.last_generation_stats.consecutive_failure_stop is True
    assert generator.last_generation_stats.attempt_count == 3


def test_research_context_provider_hits_cache_and_expires_with_ttl() -> None:
    config = load_config("config/dev.yaml")
    config.storage.path = ":memory:"
    provider = CachedResearchContextProvider(enabled=True, ttl_seconds=1)
    environment = _environment("config/dev.yaml", "cache-hit")

    first = provider.load(config, environment, stage="brain-sim-data")
    second = provider.load(config, environment, stage="brain-sim-data")

    assert first.profile.cache_hit is False
    assert second.profile.cache_hit is True

    provider._entries[first.profile.cache_key].cached_at -= 2  # noqa: SLF001
    third = provider.load(config, environment, stage="brain-sim-data")

    assert third.profile.cache_hit is False
    assert third.profile.cache_reason == "ttl_expired"


def test_research_context_provider_persist_metadata_skips_rewrites_on_cache_hit() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        environment = _environment("config/dev.yaml", "cache-persist")
        init_run(repository, config, environment, status="running")
        provider = CachedResearchContextProvider(enabled=True, ttl_seconds=0)

        first = provider.load(config, environment, stage="brain-sim-data")
        first_persist = provider.persist_metadata(repository, config, environment, first, round_index=1)
        first_run_score_created_at = repository.connection.execute(
            "SELECT created_at FROM run_field_scores WHERE run_id = ? ORDER BY field_name ASC LIMIT 1",
            (environment.context.run_id,),
        ).fetchone()[0]
        first_catalog_updated_at = repository.connection.execute(
            "SELECT updated_at FROM field_catalog ORDER BY field_name ASC LIMIT 1"
        ).fetchone()[0]

        second = provider.load(config, environment, stage="brain-sim-data")
        second_persist = provider.persist_metadata(repository, config, environment, second, round_index=2)
        second_run_score_created_at = repository.connection.execute(
            "SELECT created_at FROM run_field_scores WHERE run_id = ? ORDER BY field_name ASC LIMIT 1",
            (environment.context.run_id,),
        ).fetchone()[0]
        second_catalog_updated_at = repository.connection.execute(
            "SELECT updated_at FROM field_catalog ORDER BY field_name ASC LIMIT 1"
        ).fetchone()[0]
    finally:
        repository.close()

    assert first_persist == {
        "dataset_summary_persisted": True,
        "field_catalog_persisted": True,
        "run_field_scores_persisted": True,
    }
    assert second_persist == {
        "dataset_summary_persisted": False,
        "field_catalog_persisted": False,
        "run_field_scores_persisted": False,
    }
    assert first_run_score_created_at == second_run_score_created_at
    assert first_catalog_updated_at == second_catalog_updated_at


def test_brain_batch_service_persists_generation_stage_metrics() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        environment = _environment("config/dev.yaml", "generation-stage")
        init_run(repository, config, environment, status="running")

        result = BrainBatchService(repository).prepare_service_batch(
            config=config,
            environment=environment,
            count=6,
            mutation_parent_ids=None,
            round_index=1,
        )
        stage_metrics = repository.get_stage_metrics(environment.context.run_id)
    finally:
        repository.close()

    generation_rows = [row for row in stage_metrics if row["stage"] == "generation"]
    assert generation_rows
    metrics = json.loads(generation_rows[-1]["metrics_json"])
    assert metrics["generated"] == len(result.candidates)
    assert metrics["selected_for_simulation"] == len(result.selected)
    assert "load_research_context_ms" in metrics
    assert "resolve_field_registry_ms" in metrics
    assert "top_fail_reasons" in metrics
