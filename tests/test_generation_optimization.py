from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from alpha.parser import parse_expression
from alpha.validator import validate_expression
from core.config import AdaptiveGenerationConfig, GenerationConfig, load_config
from core.run_context import RunContext
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import build_registry
from generator.diversity_tracker import GenerationDiversityTracker
from generator.engine import AlphaCandidate, AlphaGenerationEngine, CandidateBuildResult, GenerationSessionStats
from generator.guided_generator import GuidedGenerator
from generator.mutation_policy import MutationPolicy
from generator.seed_utils import derive_generation_seed
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot
from services.brain_batch_service import BrainBatchService
from services.data_service import (
    CachedResearchContextProvider,
    apply_local_validation_field_penalties_to_registry,
    build_local_validation_field_penalties,
)
from services.recipe_guided_generator import RecipeGuidedStats
from services.runtime_service import build_command_environment, init_run
from storage.models import StageMetricRecord
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
    fields = {
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
    fields["sector"] = FieldSpec(
        name="sector",
        dataset="test",
        field_type="vector",
        coverage=1.0,
        alpha_usage_count=6,
        category="group",
        runtime_available=True,
        field_score=0.9,
        category_weight=0.8,
    )
    fields["forecast_currency_tangible_book_value_per_share"] = FieldSpec(
        name="forecast_currency_tangible_book_value_per_share",
        dataset="test",
        field_type="vector",
        coverage=1.0,
        alpha_usage_count=4,
        category="analyst",
        runtime_available=True,
        field_score=0.7,
        category_weight=0.8,
    )
    return FieldRegistry(fields=fields)


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
    original_group = field_registry_type.generation_group_key_fields
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
    monkeypatch.setattr(field_registry_type, "generation_group_key_fields", counted_group)
    monkeypatch.setattr(field_registry_type, "field_types", counted_types)

    assert engine.build_candidate("rank(close)", mode="test", parent_ids=()) is not None
    assert engine.build_candidate("rank(volume)", mode="test", parent_ids=()) is not None
    assert counts == {"numeric": 1, "group": 1, "types": 1}


def test_engine_validation_context_uses_only_true_group_keys() -> None:
    config = _build_generation_config()
    engine = AlphaGenerationEngine(
        config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(["group_neutralize", *config.allowed_operators]),
        field_registry=_build_field_registry(),
    )

    context = engine.prepare_validation_context()

    assert context.group_fields == {"sector"}
    assert "sector" in context.allowed_generation_fields
    assert "forecast_currency_tangible_book_value_per_share" not in context.group_fields
    assert "forecast_currency_tangible_book_value_per_share" not in context.allowed_generation_fields


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
    assert disallowed.failure_reason == "validation_disallowed_field"
    assert disallowed.failure_fields == ("beta",)
    assert redundant.failure_reason == "redundant_expression"


def test_validator_records_disallowed_field_name() -> None:
    registry = build_registry(["rank"])
    validation = validate_expression(
        node=parse_expression("rank(foo)"),
        registry=registry,
        allowed_fields={"close"},
        max_depth=5,
    )

    assert validation.primary_reason_code == "validation_disallowed_field"
    assert validation.issues[0].field_name == "foo"


def test_candidate_build_result_maps_validation_sub_reasons() -> None:
    config = _build_generation_config()
    engine = AlphaGenerationEngine(
        config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(["rank", "group_neutralize", "ts_mean"]),
        field_registry=_build_field_registry(),
    )

    unknown_operator = engine._build_candidate_result("mystery(close)", mode="test", parent_ids=())  # noqa: SLF001
    arity_mismatch = engine._build_candidate_result("rank(close, 1)", mode="test", parent_ids=())  # noqa: SLF001
    invalid_group = engine._build_candidate_result(
        "group_neutralize(close, forecast_currency_tangible_book_value_per_share)",
        mode="test",
        parent_ids=(),
    )  # noqa: SLF001
    unsupported = engine._build_candidate_result("group_neutralize(sector, sector)", mode="test", parent_ids=())  # noqa: SLF001
    semantic_invalid = engine._build_candidate_result("ts_mean(close, 0)", mode="test", parent_ids=())  # noqa: SLF001

    assert unknown_operator.failure_reason == "validation_unknown_operator"
    assert arity_mismatch.failure_reason == "validation_operator_arity_mismatch"
    assert invalid_group.failure_reason == "validation_invalid_group_field"
    assert unsupported.failure_reason == "validation_unsupported_combination"
    assert semantic_invalid.failure_reason == "validation_semantic_invalid"


def test_generation_session_stats_remaps_legacy_validation_bucket() -> None:
    session = GenerationSessionStats()

    session.record_failure("expression_validation_failed")
    session.record_failure("validation_disallowed_field")
    session.record_failure("duplicate_normalized_expression")

    metrics = session.to_metrics(generated_count=0, selected_for_simulation=0)

    assert metrics["validate_fail_count"] == 2
    assert "expression_validation_failed" not in metrics["failure_reason_counts"]
    assert metrics["failure_reason_counts"]["validation_unknown_error"] == 1
    assert "expression_validation_failed" not in metrics["top_fail_reasons"]


def test_generation_session_stats_omits_failure_samples_at_info() -> None:
    session = GenerationSessionStats()
    session.record_failure("validation_disallowed_field", expression="rank(foo)")
    session.record_failure("duplicate_normalized_expression", expression="rank(close)")

    metrics = session.to_metrics(generated_count=0, selected_for_simulation=0, include_debug_samples=False)

    assert "failure_samples" not in metrics


def test_generation_session_stats_tracks_validation_field_counts_and_samples() -> None:
    session = GenerationSessionStats()

    session.record_failure("validation_disallowed_field", expression="rank(foo)", fields=("foo",))
    session.record_failure("validation_disallowed_field", expression="rank(foo + bar)", fields=("foo", "bar"))
    for index in range(60):
        field_name = f"field_{index:02d}"
        session.record_failure("validation_disallowed_field", expression=f"rank({field_name})", fields=(field_name,))

    metrics = session.to_metrics(generated_count=0, selected_for_simulation=0, include_debug_samples=False)

    assert metrics["failure_reason_counts"]["validation_disallowed_field"] == 62
    assert metrics["validation_disallowed_field_counts"]["foo"] == 2
    assert metrics["validation_disallowed_field_counts"]["bar"] == 1
    assert len(metrics["validation_disallowed_field_counts"]) == 50
    assert len(metrics["validation_disallowed_field_samples"]) == 10
    assert {"field": "foo", "expression": "rank(foo)"} in metrics["validation_disallowed_field_samples"]


def test_generation_session_stats_persists_redundant_samples_at_info() -> None:
    session = GenerationSessionStats()
    session.record_failure("redundant_expression", expression="rank(rank(close))")
    session.record_failure("validation_disallowed_field", expression="rank(foo)")

    metrics = session.to_metrics(generated_count=0, selected_for_simulation=0, include_debug_samples=False)

    assert metrics["redundant_expression_samples"] == ["rank(rank(close))"]
    assert "failure_samples" not in metrics


def test_generation_session_stats_limits_failure_samples_in_debug() -> None:
    session = GenerationSessionStats()
    for expression in ("rank(foo)", "rank(bar)", "rank(baz)", "rank(qux)"):
        session.record_failure("validation_disallowed_field", expression=expression)

    metrics = session.to_metrics(generated_count=0, selected_for_simulation=0, include_debug_samples=True)

    assert metrics["failure_samples"]["validation_disallowed_field"] == [
        "rank(foo)",
        "rank(bar)",
        "rank(baz)",
    ]


def test_generation_session_stats_reports_duplicate_breakdowns() -> None:
    session = GenerationSessionStats()

    session.record_duplicate(
        "structural_duplicate_expression",
        expression="rank(close)",
        mutation_mode="exploit_local",
        motif="momentum",
        operator_path=("rank", "ts_mean"),
        pre_dedup=True,
    )
    session.record_duplicate(
        "duplicate_normalized_expression",
        expression="rank(volume)",
        mutation_mode="novelty",
        motif="mean_reversion",
        operator_path=("rank", "ts_delta"),
        pre_dedup=False,
    )

    metrics = session.to_metrics(generated_count=0, selected_for_simulation=0)

    assert metrics["pre_dedup_reject_count"] == 1
    assert metrics["structural_duplicate_count"] == 1
    assert metrics["normalized_duplicate_count"] == 1
    assert metrics["duplicate_by_mutation_mode"] == {"exploit_local": 1, "novelty": 1}
    assert metrics["duplicate_by_motif"] == {"momentum": 1, "mean_reversion": 1}
    assert metrics["duplicate_by_operator_path"] == {"rank>ts_mean": 1, "rank>ts_delta": 1}


def test_local_validation_field_penalty_builder_and_registry_downrank() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        penalty_config = config.adaptive_generation.local_validation_field_penalty
        penalty_config.enabled = True
        penalty_config.lookback_rounds = 2
        penalty_config.min_count = 2
        penalty_config.max_fields = 10
        penalty_config.penalty_strength = 1.0
        penalty_config.min_multiplier = 0.10
        run_id = "local-validation-penalty"
        repository.save_stage_metrics(
            [
                StageMetricRecord(
                    run_id=run_id,
                    round_index=1,
                    stage="generation",
                    metrics_json=json.dumps({"validation_disallowed_field_counts": {"volume": 5}}),
                    created_at="2026-04-22T00:00:00+00:00",
                ),
                StageMetricRecord(
                    run_id=run_id,
                    round_index=2,
                    stage="generation",
                    metrics_json=json.dumps({"validation_disallowed_field_counts": {"close": 1, "missing": 5}}),
                    created_at="2026-04-22T00:01:00+00:00",
                ),
                StageMetricRecord(
                    run_id=run_id,
                    round_index=3,
                    stage="generation",
                    metrics_json=json.dumps({"validation_disallowed_field_counts": {"close": 1, "returns": 1}}),
                    created_at="2026-04-22T00:02:00+00:00",
                ),
            ]
        )

        penalty_result = build_local_validation_field_penalties(repository, config, run_id)
        penalized_registry, applied = apply_local_validation_field_penalties_to_registry(
            _build_field_registry(),
            penalty_result,
        )
    finally:
        repository.close()

    assert penalty_result.counts == {"missing": 5, "close": 2}
    assert penalty_result.multipliers["close"] == 0.5
    assert "volume" not in penalty_result.counts
    assert applied.applied_field_count == 1
    assert penalized_registry.get("close").field_score == 0.4
    assert penalized_registry.contains("missing") is False


def test_mutation_policy_downranks_parent_with_penalized_field() -> None:
    class RandomProbe:
        def __init__(self) -> None:
            self.weights: list[float] = []

        def choice(self, sequence):
            return sequence[0]

        def choices(self, population, weights, k):
            self.weights = list(weights)
            return [population[0]]

        def random(self) -> float:
            return 0.5

    class Parent:
        def __init__(self, alpha_id: str, fields: tuple[str, ...]) -> None:
            self.alpha_id = alpha_id
            self.family_signature = alpha_id
            self.generation_metadata = {"fields_used": list(fields)}
            self.fields_used = fields

    randomizer = RandomProbe()
    config = _build_generation_config()
    policy = MutationPolicy(
        config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        memory_service=PatternMemoryService(),
        randomizer=randomizer,
        field_registry=_build_field_registry(),
        registry=build_registry(config.allowed_operators),
        field_penalty_multipliers={"close": 0.2},
    )

    policy._select_parent(  # noqa: SLF001
        [Parent("bad", ("close",)), Parent("good", ("volume",))],
        diversity_tracker=None,
    )

    assert randomizer.weights == [0.2, 1.0]


def test_build_candidates_rejects_structural_duplicates_before_validation(monkeypatch) -> None:
    config = _build_generation_config()
    engine = AlphaGenerationEngine(
        config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(config.allowed_operators),
        field_registry=_build_field_registry(),
    )
    session = GenerationSessionStats()
    valid_candidate = AlphaCandidate(
        alpha_id="alpha-1",
        expression="rank(close)",
        normalized_expression="rank(close)",
        generation_mode="test",
        parent_ids=(),
        complexity=2,
        created_at="2026-03-30T00:00:00+00:00",
        generation_metadata={
            "motif": "momentum",
            "mutation_mode": "exploit_local",
            "operator_path": ["rank", "ts_mean"],
            "field_families": ["price"],
        },
    )
    calls: list[str] = []

    def fake_build_candidate_result(*args, **kwargs):
        calls.append(str(kwargs["expression"]))
        return CandidateBuildResult(candidate=valid_candidate)

    monkeypatch.setattr(engine, "_build_candidate_result", fake_build_candidate_result)

    candidates = engine._build_candidates(  # noqa: SLF001
        [
            (
                "rank(close)",
                "test",
                (),
                {
                    "motif": "momentum",
                    "mutation_mode": "exploit_local",
                    "operator_path": ["rank", "ts_mean"],
                    "pre_normalized_expression": "rank(close)",
                    "genome_hash": "g-1",
                },
            ),
            (
                "rank(close)",
                "test",
                (),
                {
                    "motif": "momentum",
                    "mutation_mode": "exploit_local",
                    "operator_path": ["rank", "ts_mean"],
                    "pre_normalized_expression": "rank(close)",
                    "genome_hash": "g-2",
                },
            ),
        ],
        existing_normalized=set(),
        session=session,
    )

    metrics = session.to_metrics(generated_count=len(candidates), selected_for_simulation=0)
    assert len(calls) == 1
    assert len(candidates) == 1
    assert metrics["pre_dedup_reject_count"] == 1
    assert metrics["structural_duplicate_count"] == 1


def test_build_candidates_rejects_history_duplicates_before_validation(monkeypatch) -> None:
    config = _build_generation_config()
    engine = AlphaGenerationEngine(
        config=config,
        adaptive_config=AdaptiveGenerationConfig(),
        registry=build_registry(config.allowed_operators),
        field_registry=_build_field_registry(),
    )
    session = GenerationSessionStats()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("_build_candidate_result should not run for pre-dedup history duplicates")

    monkeypatch.setattr(engine, "_build_candidate_result", fail_if_called)

    candidates = engine._build_candidates(  # noqa: SLF001
        [
            (
                "rank(close)",
                "test",
                (),
                {
                    "motif": "momentum",
                    "mutation_mode": "exploit_local",
                    "operator_path": ["rank", "ts_mean"],
                    "pre_normalized_expression": "rank(close)",
                    "genome_hash": "g-1",
                },
            )
        ],
        existing_normalized={"rank(close)"},
        session=session,
    )

    metrics = session.to_metrics(generated_count=len(candidates), selected_for_simulation=0)
    assert candidates == []
    assert metrics["pre_dedup_reject_count"] == 1
    assert metrics["normalized_duplicate_count"] == 1


def test_round_scoped_generation_seed_changes_by_round_and_scope() -> None:
    round_one_fresh = derive_generation_seed(11, run_id="run-1", round_index=1, scope="fresh")
    round_two_fresh = derive_generation_seed(11, run_id="run-1", round_index=2, scope="fresh")
    round_one_mutation = derive_generation_seed(11, run_id="run-1", round_index=1, scope="mutation")

    assert round_one_fresh != round_two_fresh
    assert round_one_fresh != round_one_mutation


def test_engine_generate_switches_to_explore_phase_when_novelty_tail_starts(monkeypatch) -> None:
    config = _build_generation_config()
    adaptive = AdaptiveGenerationConfig(
        exploration_ratio=0.5,
        max_consecutive_failures=100,
        min_candidates_before_early_exit=1,
    )
    engine = AlphaGenerationEngine(
        config=config,
        adaptive_config=adaptive,
        registry=build_registry(config.allowed_operators),
        field_registry=_build_field_registry(),
    )
    phases: list[str] = []
    render_expressions = iter(["rank(close)", "rank(volume)"])
    fake_genome = type("FakeGenome", (), {"stable_hash": "g-1"})()

    class FakeRender:
        def __init__(self, expression: str) -> None:
            self.expression = expression
            self.normalized_expression = expression
            self.operator_path = ()

    original_should_stop = engine._should_stop_generation  # noqa: SLF001

    def record_phase(**kwargs):
        phases.append(str(kwargs["phase"]))
        return original_should_stop(**kwargs)

    monkeypatch.setattr(engine, "_should_stop_generation", record_phase)
    monkeypatch.setattr(engine.genome_builder, "build_random_genome", lambda **kwargs: fake_genome)
    monkeypatch.setattr(engine.repair_policy, "repair", lambda genome: (genome, ()))
    monkeypatch.setattr(engine.grammar, "render", lambda genome: FakeRender(next(render_expressions)))
    monkeypatch.setattr(engine, "_render_metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(engine, "_reject_pre_dedup_candidate", lambda **kwargs: False)

    candidates = engine.generate(count=2)

    assert len(candidates) == 2
    assert phases[:2] == ["exploit", "explore"]
    assert engine.last_generation_stats is not None
    assert engine.last_generation_stats.explore_phase_entered is True
    assert engine.last_generation_stats.exploit_attempt_count == 1
    assert engine.last_generation_stats.explore_attempt_count == 1


def test_engine_explore_failure_limit_prefers_explicit_override() -> None:
    config = _build_generation_config()
    adaptive = AdaptiveGenerationConfig(
        max_consecutive_failures=4,
        explore_max_consecutive_failures=10,
        min_candidates_before_early_exit=1,
    )
    engine = AlphaGenerationEngine(
        config=config,
        adaptive_config=adaptive,
        registry=build_registry(config.allowed_operators),
        field_registry=_build_field_registry(),
    )

    assert engine._resolve_consecutive_failure_limit(phase="exploit", candidate_count=1, target_count=5) == 2  # noqa: SLF001
    assert engine._resolve_consecutive_failure_limit(phase="explore", candidate_count=1, target_count=5) == 5  # noqa: SLF001


def test_engine_generate_skips_unparseable_render_before_candidate_build(monkeypatch) -> None:
    config = _build_generation_config()
    adaptive = AdaptiveGenerationConfig(
        max_consecutive_failures=1,
        min_candidates_before_early_exit=1,
    )
    engine = AlphaGenerationEngine(
        config=config,
        adaptive_config=adaptive,
        registry=build_registry(config.allowed_operators),
        field_registry=_build_field_registry(),
    )
    fake_genome = type("FakeGenome", (), {"stable_hash": "g-parse"})()
    fake_render = type("FakeRender", (), {"expression": "rank("})()

    monkeypatch.setattr(engine.genome_builder, "build_random_genome", lambda **kwargs: fake_genome)
    monkeypatch.setattr(engine.repair_policy, "repair", lambda genome: (genome, ()))
    monkeypatch.setattr(engine.grammar, "render", lambda genome: fake_render)
    monkeypatch.setattr(
        engine,
        "_render_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_render_metadata should not run")),
    )
    monkeypatch.setattr(
        engine,
        "_build_candidate_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_build_candidate_result should not run")),
    )

    candidates = engine.generate(count=1)

    assert candidates == []
    assert engine.last_generation_stats is not None
    metrics = engine.last_generation_stats.to_metrics(generated_count=0, selected_for_simulation=0)
    assert metrics["parse_fail_count"] == 1


def test_guided_generator_stops_on_time_budget(monkeypatch) -> None:
    adaptive = AdaptiveGenerationConfig(
        max_generation_seconds=1.0,
        max_attempt_multiplier=50,
        exploit_budget_ratio=0.6,
        explore_budget_ratio=0.4,
        min_explore_attempts=1,
        min_explore_seconds=0.5,
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
        lambda *args, **kwargs: CandidateBuildResult(candidate=None, failure_reason="validation_unknown_error"),
    )

    candidates = generator.generate(
        count=5,
        snapshot=PatternMemorySnapshot(regime_key="budget"),
    )

    assert candidates == []
    assert generator.last_generation_stats is not None
    assert generator.last_generation_stats.timeout_stop is True
    assert generator.last_generation_stats.explore_phase_entered is False
    assert generator.last_generation_stats.exploit_attempt_count == 1
    assert generator.last_generation_stats.explore_attempt_count == 0
    assert generator.last_generation_stats.attempt_count == 1


def test_guided_generator_explore_runs_after_exploit_failure_break_and_returns_partial_candidates(monkeypatch) -> None:
    adaptive = AdaptiveGenerationConfig(
        max_generation_seconds=20.0,
        max_attempt_multiplier=50,
        exploit_budget_ratio=0.6,
        explore_budget_ratio=0.4,
        min_explore_attempts=2,
        min_explore_seconds=1.0,
        max_consecutive_failures=4,
        explore_max_consecutive_failures=4,
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
            CandidateBuildResult(candidate=None, failure_reason="validation_unknown_error"),
            CandidateBuildResult(candidate=None, failure_reason="validation_unknown_error"),
            CandidateBuildResult(candidate=None, failure_reason="validation_unknown_error"),
            CandidateBuildResult(candidate=None, failure_reason="validation_unknown_error"),
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
    assert generator.last_generation_stats.exploit_consecutive_failure_stop is True
    assert generator.last_generation_stats.explore_phase_entered is True
    assert generator.last_generation_stats.explore_consecutive_failure_stop is True
    assert generator.last_generation_stats.exploit_attempt_count == 3
    assert generator.last_generation_stats.explore_attempt_count == 2
    assert generator.last_generation_stats.attempt_count == 5
    metrics = generator.last_generation_stats.to_metrics(generated_count=len(candidates), selected_for_simulation=0)
    assert metrics["exploit_attempt_count"] == 3
    assert metrics["explore_attempt_count"] == 2
    assert metrics["explore_phase_entered"] is True
    assert metrics["exploit_consecutive_failure_stop"] is True
    assert metrics["explore_consecutive_failure_stop"] is True


def test_guided_generator_explore_uses_doubled_failure_limit_by_default(monkeypatch) -> None:
    adaptive = AdaptiveGenerationConfig(
        max_generation_seconds=20.0,
        max_attempt_multiplier=50,
        exploit_budget_ratio=0.6,
        explore_budget_ratio=0.4,
        min_explore_attempts=2,
        min_explore_seconds=1.0,
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
            CandidateBuildResult(candidate=None, failure_reason="validation_unknown_error"),
            CandidateBuildResult(candidate=None, failure_reason="validation_unknown_error"),
            CandidateBuildResult(candidate=None, failure_reason="validation_unknown_error"),
            CandidateBuildResult(candidate=None, failure_reason="validation_unknown_error"),
            CandidateBuildResult(candidate=None, failure_reason="validation_unknown_error"),
            CandidateBuildResult(candidate=None, failure_reason="validation_unknown_error"),
        ]
    )

    monkeypatch.setattr(generator.genome_builder, "build_guided_genome", lambda **kwargs: object())
    monkeypatch.setattr(generator.repair_policy, "repair", lambda genome: (genome, ()))
    monkeypatch.setattr(generator.grammar, "render", lambda genome: fake_render)
    monkeypatch.setattr(generator.base_engine, "_render_metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(generator.base_engine, "_build_candidate_result", lambda *args, **kwargs: next(results))

    candidates = generator.generate(
        count=5,
        snapshot=PatternMemorySnapshot(regime_key="fail-slower"),
    )

    assert len(candidates) == 1
    assert generator.last_generation_stats is not None
    assert generator.last_generation_stats.exploit_attempt_count == 3
    assert generator.last_generation_stats.explore_attempt_count == 4
    assert generator.last_generation_stats.exploit_consecutive_failure_stop is True
    assert generator.last_generation_stats.explore_consecutive_failure_stop is True


def test_guided_generator_skips_unparseable_render_before_candidate_build(monkeypatch) -> None:
    adaptive = AdaptiveGenerationConfig(
        max_consecutive_failures=1,
        explore_max_consecutive_failures=1,
        min_candidates_before_early_exit=1,
    )
    generator = _build_guided_generator(adaptive)
    fake_render = type("FakeRender", (), {"expression": "rank("})()

    monkeypatch.setattr(generator.genome_builder, "build_guided_genome", lambda **kwargs: object())
    monkeypatch.setattr(generator.repair_policy, "repair", lambda genome: (genome, ()))
    monkeypatch.setattr(generator.grammar, "render", lambda genome: fake_render)
    monkeypatch.setattr(
        generator.base_engine,
        "_render_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_render_metadata should not run")),
    )
    monkeypatch.setattr(
        generator.base_engine,
        "_build_candidate_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_build_candidate_result should not run")),
    )

    candidates = generator.generate(
        count=1,
        snapshot=PatternMemorySnapshot(regime_key="parse-guard"),
    )

    assert candidates == []
    assert generator.last_generation_stats is not None
    metrics = generator.last_generation_stats.to_metrics(generated_count=0, selected_for_simulation=0)
    assert metrics["parse_fail_count"] == 1
    assert metrics["explore_attempt_count"] == 1


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


def test_brain_batch_service_persists_generation_stage_metrics(tmp_path: Path) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        config.runtime.progress_log_dir = str(tmp_path / "progress")
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
        log_path = tmp_path / "progress" / f"{environment.context.run_id}.jsonl"
        progress_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    finally:
        repository.close()

    generation_rows = [row for row in stage_metrics if row["stage"] == "generation"]
    assert generation_rows
    assert any(row["event"] == "batch_prepared" for row in progress_rows)
    metrics = json.loads(generation_rows[-1]["metrics_json"])
    assert metrics["generated"] == len(result.candidates)
    assert metrics["selected_for_simulation"] == len(result.selected)
    assert "load_research_context_ms" in metrics
    assert "resolve_field_registry_ms" in metrics
    assert "top_fail_reasons" in metrics
    assert "validation_disallowed_field_counts" in metrics
    assert "validation_disallowed_field_samples" in metrics
    assert "local_validation_penalized_field_count" in metrics
    assert "local_validation_penalty_counts" in metrics
    assert "local_validation_penalty_multipliers" in metrics


def test_brain_batch_service_includes_recipe_guided_metrics(tmp_path: Path, monkeypatch) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        config.runtime.progress_log_dir = str(tmp_path / "progress")
        environment = _environment("config/dev.yaml", "generation-recipe-guided")
        init_run(repository, config, environment, status="running")

        service = BrainBatchService(repository)
        recipe_candidate = AlphaCandidate(
            alpha_id="recipe-1",
            expression="rank(ts_mean(close,5))",
            normalized_expression="rank(ts_mean(close,5))",
            generation_mode="recipe_guided",
            parent_ids=(),
            complexity=3,
            created_at="2026-04-23T00:00:00+00:00",
            template_name="recipe_fundamental_quality",
            fields_used=("close",),
            operators_used=("rank", "ts_mean"),
            depth=3,
            generation_metadata={
                "search_bucket_id": "fundamental_quality|fundamental|balanced",
                "recipe_bucket_prior": 0.08,
            },
        )
        recipe_stats = RecipeGuidedStats(enabled=True)
        recipe_stats.generated_count = 1
        recipe_stats.attempt_count = 2
        recipe_stats.success_count = 1
        recipe_stats.bucket_counts["fundamental_quality|fundamental|balanced"] = 1
        recipe_stats.template_counts["recipe_fundamental_quality"] = 1
        recipe_stats.parentless_count = 1

        monkeypatch.setattr(service, "_generate_mutation_candidates", lambda **kwargs: ([], GenerationSessionStats()))
        monkeypatch.setattr(service, "_generate_quality_polish_candidates", lambda **kwargs: ([], type("Stats", (), {"selected_count": 0, "to_metrics": lambda self: {}})()))
        monkeypatch.setattr(
            service,
            "_generate_recipe_guided_candidates",
            lambda **kwargs: ([recipe_candidate], recipe_stats),
        )
        monkeypatch.setattr(service, "_generate_fresh_candidates", lambda **kwargs: ([], GenerationSessionStats()))

        result = service.prepare_service_batch(
            config=config,
            environment=environment,
            count=1,
            mutation_parent_ids=None,
            round_index=1,
        )
        stage_metrics = repository.get_stage_metrics(environment.context.run_id)
    finally:
        repository.close()

    metrics = json.loads([row for row in stage_metrics if row["stage"] == "generation"][-1]["metrics_json"])
    assert any(candidate.generation_mode == "recipe_guided" for candidate in result.candidates)
    assert metrics["recipe_guided_generated"] == 1
    assert metrics["recipe_guided_attempt_count"] == 2
    assert metrics["recipe_guided_success_count"] == 1
    assert metrics["recipe_guided_bucket_counts"] == {"fundamental_quality|fundamental|balanced": 1}
    assert metrics["recipe_guided_template_counts"] == {"recipe_fundamental_quality": 1}
    assert "recipe_guided_selected_by_bucket" in metrics


def test_brain_batch_service_persists_dynamic_source_budget_metrics(tmp_path: Path, monkeypatch) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        config.runtime.progress_log_dir = str(tmp_path / "progress")
        environment = _environment("config/dev.yaml", "generation-source-budget")
        init_run(repository, config, environment, status="running")

        service = BrainBatchService(repository)
        fresh_candidate = AlphaCandidate(
            alpha_id="fresh-1",
            expression="rank(ts_mean(close,5))",
            normalized_expression="rank(ts_mean(close,5))",
            generation_mode="guided_explore",
            parent_ids=(),
            complexity=3,
            created_at="2026-04-23T00:00:00+00:00",
            template_name="template",
            fields_used=("close",),
            operators_used=("rank", "ts_mean"),
            depth=3,
            generation_metadata={},
        )

        class _PolishStats:
            selected_count = 0
            turnover_repair_selected = 0

            def to_metrics(self):
                return {
                    "quality_polish_generated": 0,
                    "quality_polish_attempt_count": 0,
                    "quality_polish_success_count": 0,
                    "quality_polish_selected": 0,
                    "quality_polish_failure_reason_counts": {},
                    "turnover_repair_generated": 0,
                    "turnover_repair_attempt_count": 0,
                    "turnover_repair_success_count": 0,
                    "turnover_repair_selected": 0,
                    "turnover_repair_transform_counts": {},
                    "quality_polish_generation_total_ms": 0.0,
                }

        monkeypatch.setattr(service, "_generate_mutation_candidates", lambda **kwargs: ([], GenerationSessionStats()))
        monkeypatch.setattr(service, "_generate_quality_polish_candidates", lambda **kwargs: ([], _PolishStats()))
        monkeypatch.setattr(service, "_generate_recipe_guided_candidates", lambda **kwargs: ([], RecipeGuidedStats(enabled=True)))
        monkeypatch.setattr(service, "_generate_fresh_candidates", lambda **kwargs: ([fresh_candidate], GenerationSessionStats()))
        monkeypatch.setattr(
            service,
            "_plan_generation_source_budgets",
            lambda **kwargs: (
                {"quality_polish": 1, "recipe_guided": 1, "fresh": 2},
                {"quality_polish": 0.55, "recipe_guided": 0.65, "fresh": 0.50},
            ),
        )

        service.prepare_service_batch(
            config=config,
            environment=environment,
            count=2,
            mutation_parent_ids=None,
            round_index=1,
        )
        stage_metrics = repository.get_stage_metrics(environment.context.run_id)
    finally:
        repository.close()

    metrics = json.loads([row for row in stage_metrics if row["stage"] == "generation"][-1]["metrics_json"])
    assert metrics["source_budget_allocations"] == {"quality_polish": 1, "recipe_guided": 1, "fresh": 2}
    assert metrics["source_yield_scores"] == {
        "quality_polish": 0.55,
        "recipe_guided": 0.65,
        "fresh": 0.5,
    }
    assert metrics["source_generated_counts"]["fresh"] == 1


def test_brain_batch_service_persists_redundant_expression_samples(tmp_path: Path, monkeypatch) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        config.runtime.progress_log_dir = str(tmp_path / "progress")
        environment = _environment("config/dev.yaml", "generation-redundant-samples")
        init_run(repository, config, environment, status="running")

        service = BrainBatchService(repository)
        redundant_stats = GenerationSessionStats()
        redundant_stats.record_failure("redundant_expression", expression="rank(rank(close))")

        monkeypatch.setattr(
            service,
            "_generate_mutation_candidates",
            lambda **kwargs: ([], GenerationSessionStats()),
        )
        monkeypatch.setattr(
            service,
            "_generate_fresh_candidates",
            lambda **kwargs: ([], redundant_stats),
        )

        service.prepare_service_batch(
            config=config,
            environment=environment,
            count=1,
            mutation_parent_ids=None,
            round_index=1,
        )
        stage_metrics = repository.get_stage_metrics(environment.context.run_id)
        log_path = tmp_path / "progress" / f"{environment.context.run_id}.jsonl"
        progress_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    finally:
        repository.close()

    generation_rows = [row for row in stage_metrics if row["stage"] == "generation"]
    assert generation_rows
    metrics = json.loads(generation_rows[-1]["metrics_json"])
    assert metrics["redundant_expression_samples"] == ["rank(rank(close))"]
    prepared_rows = [row for row in progress_rows if row["event"] == "batch_prepared"]
    assert prepared_rows
    assert prepared_rows[-1]["payload"]["generation_stage_metrics"]["redundant_expression_samples"] == [
        "rank(rank(close))"
    ]


def test_brain_batch_service_applies_local_validation_penalty_from_previous_round(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        config.runtime.progress_log_dir = str(tmp_path / "progress")
        penalty_config = config.adaptive_generation.local_validation_field_penalty
        penalty_config.enabled = True
        penalty_config.lookback_rounds = 20
        penalty_config.min_count = 2
        penalty_config.penalty_strength = 1.0
        environment = _environment("config/dev.yaml", "generation-local-penalty")
        init_run(repository, config, environment, status="running")
        repository.save_stage_metrics(
            [
                StageMetricRecord(
                    run_id=environment.context.run_id,
                    round_index=1,
                    stage="generation",
                    metrics_json=json.dumps({"validation_disallowed_field_counts": {"close": 5}}),
                    created_at="2026-04-22T00:00:00+00:00",
                )
            ]
        )

        service = BrainBatchService(repository)
        captured: dict[str, float] = {}

        monkeypatch.setattr(
            service,
            "_generate_mutation_candidates",
            lambda **kwargs: ([], GenerationSessionStats()),
        )

        def fake_fresh_candidates(**kwargs):
            field_registry = kwargs["field_registry"]
            captured["close_score"] = field_registry.get("close").field_score
            captured["close_multiplier"] = kwargs["field_penalty_multipliers"]["close"]
            return [], GenerationSessionStats()

        monkeypatch.setattr(service, "_generate_fresh_candidates", fake_fresh_candidates)

        service.prepare_service_batch(
            config=config,
            environment=environment,
            count=1,
            mutation_parent_ids=None,
            round_index=2,
        )
        stage_metrics = repository.get_stage_metrics(environment.context.run_id)
    finally:
        repository.close()

    metrics = json.loads([row for row in stage_metrics if row["stage"] == "generation"][-1]["metrics_json"])
    assert metrics["local_validation_penalized_field_count"] >= 1
    assert metrics["local_validation_penalty_counts"]["close"] == 5
    assert metrics["local_validation_penalty_multipliers"]["close"] == captured["close_multiplier"]
    assert captured["close_score"] > 0
    assert captured["close_multiplier"] < 1.0
