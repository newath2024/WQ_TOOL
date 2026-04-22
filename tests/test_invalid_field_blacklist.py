from __future__ import annotations

from types import SimpleNamespace

import pytest

import services.generation_service as generation_service
import services.mutation_service as mutation_service
from core.brain_rejections import extract_invalid_field_from_rejection
from core.config import load_config
from core.run_context import RunContext
from data.field_registry import FieldRegistry, FieldSpec
from generator.engine import GenerationSessionStats
from memory.case_memory import CaseMemoryRecord, CaseMemoryService, ObjectiveVector
from memory.pattern_memory import MemoryParent, PatternMemoryService, PatternMemorySnapshot, RegionLearningContext
from services.brain_batch_service import BrainBatchService
from services.brain_service import BrainService
from services.closed_loop_service import ClosedLoopService
from services.data_service import (
    CachedResearchContextResult,
    ResearchContextLoadProfile,
    persist_research_metadata,
    resolve_generation_field_registry,
    sanitize_generation_research_context,
)
from services.models import DedupBatchResult, PreSimulationSelectionResult, ResearchContext, SimulationJob
from services.runtime_service import build_command_environment, init_run
from storage.models import AlphaRecord, BrainResultRecord, SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository


def _build_field_registry(*names: str) -> FieldRegistry:
    categories = {"close": "price", "volume": "volume", "returns": "price", "sector": "group"}
    field_types = {"sector": "vector"}
    return FieldRegistry(
        fields={
            name: FieldSpec(
                name=name,
                dataset="test",
                field_type=field_types.get(name, "matrix"),
                coverage=1.0,
                alpha_usage_count=1,
                category=categories.get(name, "other"),
                runtime_available=True,
                field_score=1.0,
                category_weight=1.0,
            )
            for name in names
        }
    )


def _research_context(field_registry: FieldRegistry) -> ResearchContext:
    regime_context = RegionLearningContext(region="USA", regime_key="regime", global_regime_key="global-regime")
    return ResearchContext(
        bundle=SimpleNamespace(fingerprint="dataset-fingerprint"),
        matrices=SimpleNamespace(),
        region="USA",
        regime_key="regime",
        global_regime_key="global-regime",
        region_learning_context=regime_context,
        memory_service=PatternMemoryService(),
        field_registry=field_registry,
        effective_regime_key="regime",
    )


def _config():
    config = load_config("config/dev.yaml")
    config.storage.path = ":memory:"
    config.generation.allowed_fields = ["close", "volume"]
    config.generation.allow_catalog_fields_without_runtime = False
    config.adaptive_generation.enabled = False
    config.loop.rounds = 1
    config.loop.generation_batch_size = 1
    config.loop.simulation_batch_size = 1
    config.brain.region = "USA"
    config.brain.universe = "TOP3000"
    config.brain.delay = 1
    return config


def _environment(command_name: str, run_id: str):
    context = RunContext.create(seed=7, config_path="config/dev.yaml", run_id=run_id)
    return build_command_environment(config_path="config/dev.yaml", command_name=command_name, context=context)


def _save_result(
    repository: SQLiteRepository,
    *,
    run_id: str,
    job_id: str,
    rejection_reason: str,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
) -> None:
    batch_id = f"batch-{job_id}"
    submitted_at = "2026-04-04T00:00:00+00:00"
    repository.submissions.upsert_batch(
        SubmissionBatchRecord(
            batch_id=batch_id,
            run_id=run_id,
            round_index=1,
            backend="api",
            status="completed",
            candidate_count=1,
            sim_config_snapshot="{}",
            export_path=None,
            notes_json="{}",
            created_at=submitted_at,
            updated_at=submitted_at,
        )
    )
    repository.submissions.upsert_submissions(
        [
            SubmissionRecord(
                job_id=job_id,
                batch_id=batch_id,
                run_id=run_id,
                round_index=1,
                candidate_id=f"candidate-{job_id}",
                expression="rank(close)",
                backend="api",
                status="failed",
                sim_config_snapshot="{}",
                submitted_at=submitted_at,
                updated_at=submitted_at,
                completed_at=submitted_at,
                export_path=None,
                raw_submission_json="{}",
                error_message=rejection_reason,
            )
        ]
    )
    repository.brain_results.save_results(
        [
            BrainResultRecord(
                job_id=job_id,
                run_id=run_id,
                round_index=1,
                batch_id=batch_id,
                candidate_id=f"candidate-{job_id}",
                expression="rank(close)",
                status="failed",
                region=region,
                universe=universe,
                delay=delay,
                neutralization="SUBINDUSTRY",
                decay=5,
                sharpe=None,
                fitness=None,
                turnover=None,
                drawdown=None,
                returns=None,
                margin=None,
                submission_eligible=None,
                rejection_reason=rejection_reason,
                raw_result_json="{}",
                metric_source="external_brain",
                simulated_at="2026-04-04T00:00:00+00:00",
                created_at="2026-04-04T00:00:00+00:00",
            )
        ]
    )


class _CaptureEngine:
    last_field_registry: FieldRegistry | None = None

    def __init__(self, *args, field_registry: FieldRegistry, **kwargs) -> None:
        del args, kwargs
        type(self).last_field_registry = field_registry

    def generate(self, *args, **kwargs):
        del args, kwargs
        return []

    def generate_mutations(self, *args, **kwargs):
        del args, kwargs
        return []


class _CaptureGuidedGenerator:
    last_parent_pool: list[MemoryParent] | None = None
    last_case_snapshot = None
    last_guardrails = None

    def __init__(self, *args, generation_guardrails=None, **kwargs) -> None:
        del args, kwargs
        type(self).last_guardrails = generation_guardrails

    def generate(self, *args, case_snapshot=None, **kwargs):
        del args, kwargs
        type(self).last_case_snapshot = case_snapshot
        return []

    def generate_mutations(self, *args, parent_pool=None, case_snapshot=None, **kwargs):
        del args, kwargs
        type(self).last_parent_pool = list(parent_pool or [])
        type(self).last_case_snapshot = case_snapshot
        return []


def _memory_parent(alpha_id: str, expression: str) -> MemoryParent:
    signature = PatternMemoryService().extract_signature(expression, generation_metadata={"motif": "momentum"})
    return MemoryParent(
        run_id="test-run",
        alpha_id=alpha_id,
        expression=expression,
        normalized_expression=expression,
        generation_mode="template",
        generation_metadata={"motif": "momentum", "parent_refs": []},
        parent_refs=(),
        family_signature=signature.family_signature,
        outcome_score=0.1,
        behavioral_novelty_score=0.1,
        fail_tags=(),
        success_tags=(),
        mutation_hints=(),
        structural_signature=signature,
    )


def _case_snapshot(*expressions: str):
    service = PatternMemoryService()
    case_memory_service = CaseMemoryService()
    records = [
        CaseMemoryRecord(
            run_id="test-run",
            alpha_id=f"case-{index}",
            region="USA",
            regime_key="regime",
            global_regime_key="global-regime",
            metric_source="external_brain",
            family_signature=service.extract_signature(expression, generation_metadata={"motif": "momentum"}).family_signature,
            structural_signature=service.extract_signature(expression, generation_metadata={"motif": "momentum"}),
            genome_hash="",
            genome=None,
            motif="momentum",
            field_families=("price",),
            operator_path=("rank",),
            complexity_bucket="moderate",
            turnover_bucket="balanced",
            horizon_bucket="medium",
            mutation_mode="exploit_local",
            parent_family_signatures=(),
            fail_tags=(),
            success_tags=(),
            objective_vector=ObjectiveVector(),
            outcome_score=0.1,
            created_at=f"2026-04-04T00:00:0{index}+00:00",
        )
        for index, expression in enumerate(expressions, start=1)
    ]
    return case_memory_service.build_snapshot("regime", records, region="USA", global_regime_key="global-regime")


class _FakeSelectionService:
    def __init__(self) -> None:
        self.last_field_registry: FieldRegistry | None = None

    def configure_runtime(self, **kwargs) -> None:
        del kwargs

    def run_pre_sim_pipeline(self, candidates, **kwargs) -> PreSimulationSelectionResult:
        del candidates
        self.last_field_registry = kwargs["field_registry"]
        return PreSimulationSelectionResult(
            selected=(),
            archived=(),
            dedup_result=DedupBatchResult(kept_candidates=(), blocked_candidates=(), decisions=()),
        )


class _FakeResearchContextProvider:
    def __init__(self, research_context: ResearchContext) -> None:
        self.research_context = research_context

    def load(self, config, environment, *, stage: str) -> CachedResearchContextResult:
        del config, environment, stage
        return CachedResearchContextResult(
            research_context=self.research_context,
            profile=ResearchContextLoadProfile(
                cache_hit=False,
                cache_reason="test",
                cache_key="test-cache-key",
                config_fingerprint="cfg",
                input_fingerprint="input",
                load_research_context_ms=1.0,
                build_field_registry_ms=1.0,
                prepare_context_ms=1.0,
                field_registry_fingerprint="field-registry",
            ),
        )

    def persist_metadata(
        self,
        repository,
        config,
        environment,
        cache_result,
        *,
        round_index: int = 0,
        research_context_override=None,
        removed_field_names=(),
    ) -> dict[str, bool]:
        del repository, config, environment, cache_result, round_index, research_context_override, removed_field_names
        return {
            "dataset_summary_persisted": False,
            "field_catalog_persisted": False,
            "run_field_scores_persisted": False,
        }


def test_extract_invalid_field_from_rejection_parses_supported_errors() -> None:
    assert extract_invalid_field_from_rejection('Attempted to use unknown variable "beta".') == "beta"
    assert extract_invalid_field_from_rejection("Invalid data field anl46_indicator") == "anl46_indicator"
    assert extract_invalid_field_from_rejection("poll_timeout") is None
    assert extract_invalid_field_from_rejection("poll_timeout_live") is None
    assert extract_invalid_field_from_rejection("poll_timeout_after_downtime") is None


def test_brain_result_store_lists_invalid_generation_fields_by_profile() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _config()
        environment = _environment("service-status", "seed-run")
        init_run(repository, config, environment, status="running")
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-close-1",
            rejection_reason='Attempted to use unknown variable "close".',
        )
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-close-2",
            rejection_reason='Attempted to use unknown variable "close".',
        )
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-beta",
            rejection_reason="Invalid data field beta",
        )
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-timeout",
            rejection_reason="poll_timeout",
        )
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-timeout-live",
            rejection_reason="poll_timeout_live",
        )
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-timeout-downtime",
            rejection_reason="poll_timeout_after_downtime",
        )
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-other-profile",
            rejection_reason='Attempted to use unknown variable "volume".',
            universe="TOP500",
        )

        blocked = repository.brain_results.list_invalid_generation_fields(
            region="USA",
            universe="TOP3000",
            delay=1,
        )
    finally:
        repository.close()

    assert blocked == {"close", "beta"}


def test_resolve_generation_field_registry_filters_blacklisted_fields_and_fails_when_empty() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _config()
        environment = _environment("generate", "blacklist-empty")
        init_run(repository, config, environment, status="running")
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-close",
            rejection_reason='Attempted to use unknown variable "close".',
        )
        research_context = _research_context(_build_field_registry("close", "volume", "sector"))

        filtered = resolve_generation_field_registry(
            repository,
            config,
            research_context,
            environment,
            stage="generate",
        )

        assert filtered.contains("volume")
        assert filtered.contains("sector")
        assert not filtered.contains("close")
        assert research_context.field_registry.contains("close")

        config.generation.allowed_fields = ["close"]
        with pytest.raises(ValueError, match="removed all numeric fields"):
            resolve_generation_field_registry(
                repository,
                config,
                research_context,
                environment,
                stage="generate",
            )
    finally:
        repository.close()


def test_persist_research_metadata_prunes_invalid_fields_from_catalog_tables() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _config()
        environment = _environment("generate", "persist-blacklist")
        init_run(repository, config, environment, status="running")
        research_context = _research_context(_build_field_registry("close", "volume", "sector"))

        persist_research_metadata(
            repository,
            config,
            environment,
            research_context,
            persist_dataset_summary=False,
        )
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-close",
            rejection_reason='Attempted to use unknown variable "close".',
        )
        sanitized_context, blocked_fields = sanitize_generation_research_context(
            repository,
            config,
            research_context,
            environment,
            stage="generate",
        )
        persist_research_metadata(
            repository,
            config,
            environment,
            sanitized_context,
            persist_dataset_summary=False,
            removed_field_names=blocked_fields,
        )
        catalog_rows = repository.connection.execute("SELECT field_name FROM field_catalog").fetchall()
        score_rows = repository.connection.execute(
            "SELECT field_name FROM run_field_scores WHERE run_id = ?",
            (environment.context.run_id,),
        ).fetchall()
    finally:
        repository.close()

    assert {row["field_name"] for row in catalog_rows} == {"volume", "sector"}
    assert {row["field_name"] for row in score_rows} == {"volume", "sector"}


def test_brain_service_prunes_invalid_field_metadata_on_unknown_variable_failure() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _config()
        environment = _environment("run-service", "brain-prune")
        init_run(repository, config, environment, status="running")
        research_context = _research_context(_build_field_registry("close", "volume"))
        persist_research_metadata(
            repository,
            config,
            environment,
            research_context,
            persist_dataset_summary=False,
        )
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-close",
                run_id=environment.context.run_id,
                round_index=1,
                backend="api",
                status="submitted",
                candidate_count=1,
                sim_config_snapshot="{}",
                export_path=None,
                notes_json="{}",
                created_at="2026-04-15T00:00:00+00:00",
                updated_at="2026-04-15T00:00:00+00:00",
            )
        )
        repository.submissions.upsert_submissions(
            [
                SubmissionRecord(
                    job_id="job-close",
                    batch_id="batch-close",
                    run_id=environment.context.run_id,
                    round_index=1,
                    candidate_id="candidate-close",
                    expression="rank(close)",
                    backend="api",
                    status="submitted",
                    sim_config_snapshot="{}",
                    submitted_at="2026-04-15T00:00:00+00:00",
                    updated_at="2026-04-15T00:00:00+00:00",
                    completed_at=None,
                    export_path=None,
                    raw_submission_json="{}",
                    error_message=None,
                )
            ]
        )

        service = BrainService(repository, config.brain, adapter=SimpleNamespace())
        service._failed_job(
            SimulationJob(
                job_id="job-close",
                candidate_id="candidate-close",
                expression="rank(close)",
                backend="api",
                status="submitted",
                submitted_at="2026-04-15T00:00:00+00:00",
                sim_config_snapshot={"region": "USA", "universe": "TOP3000", "delay": 1},
                run_id=environment.context.run_id,
                batch_id="batch-close",
                round_index=1,
            ),
            updated_at="2026-04-15T00:01:00+00:00",
            error_message='Attempted to use unknown variable "close".',
        )
        catalog_rows = repository.connection.execute("SELECT field_name FROM field_catalog").fetchall()
        score_rows = repository.connection.execute(
            "SELECT field_name FROM run_field_scores WHERE run_id = ?",
            (environment.context.run_id,),
        ).fetchall()
    finally:
        repository.close()

    assert {row["field_name"] for row in catalog_rows} == {"volume"}
    assert {row["field_name"] for row in score_rows} == {"volume"}


def test_generate_and_persist_uses_filtered_generation_field_registry(monkeypatch) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _config()
        environment = _environment("generate", "generate-blacklist")
        init_run(repository, config, environment, status="running")
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-close",
            rejection_reason='Attempted to use unknown variable "close".',
        )
        research_context = _research_context(_build_field_registry("close", "volume"))
        _CaptureEngine.last_field_registry = None
        monkeypatch.setattr(generation_service, "load_research_context", lambda *args, **kwargs: research_context)
        monkeypatch.setattr(generation_service, "persist_research_metadata", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            generation_service,
            "export_generated_alphas",
            lambda repository, environment: {"generated_alphas_latest_csv": "generated.csv"},
        )
        monkeypatch.setattr(generation_service, "AlphaGenerationEngine", _CaptureEngine)

        generation_service.generate_and_persist(repository, config, environment, count=1)
    finally:
        repository.close()

    assert _CaptureEngine.last_field_registry is not None
    assert not _CaptureEngine.last_field_registry.contains("close")
    assert _CaptureEngine.last_field_registry.contains("volume")


def test_mutate_and_persist_uses_filtered_generation_field_registry(monkeypatch) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _config()
        environment = _environment("mutate", "mutate-blacklist")
        init_run(repository, config, environment, status="running")
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-close",
            rejection_reason='Attempted to use unknown variable "close".',
        )
        research_context = _research_context(_build_field_registry("close", "volume"))
        parent_record = AlphaRecord(
            run_id=environment.context.run_id,
            alpha_id="parent-1",
            expression="rank(volume)",
            normalized_expression="rank(volume)",
            generation_mode="template",
            template_name="momentum",
            fields_used_json='["volume"]',
            operators_used_json='["rank"]',
            depth=2,
            generation_metadata="{}",
            complexity=3,
            created_at=environment.context.started_at,
            status="selected",
        )
        _CaptureEngine.last_field_registry = None
        monkeypatch.setattr(mutation_service, "load_research_context", lambda *args, **kwargs: research_context)
        monkeypatch.setattr(mutation_service, "persist_research_metadata", lambda *args, **kwargs: None)
        monkeypatch.setattr(repository, "get_top_alpha_records", lambda run_id, limit: [parent_record])
        monkeypatch.setattr(mutation_service, "AlphaGenerationEngine", _CaptureEngine)

        mutation_service.mutate_and_persist(repository, config, environment, from_top=1, count=1)
    finally:
        repository.close()

    assert _CaptureEngine.last_field_registry is not None
    assert not _CaptureEngine.last_field_registry.contains("close")
    assert _CaptureEngine.last_field_registry.contains("volume")


def test_mutate_and_persist_filters_blacklisted_memory_parents(monkeypatch) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _config()
        config.adaptive_generation.enabled = True
        environment = _environment("mutate", "mutate-memory-blacklist")
        init_run(repository, config, environment, status="running")
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-close",
            rejection_reason='Attempted to use unknown variable "close".',
        )
        research_context = _research_context(_build_field_registry("close", "volume", "sector"))
        parent_record = AlphaRecord(
            run_id=environment.context.run_id,
            alpha_id="parent-safe",
            expression="rank(volume)",
            normalized_expression="rank(volume)",
            generation_mode="template",
            template_name="momentum",
            fields_used_json='["volume"]',
            operators_used_json='["rank"]',
            depth=2,
            generation_metadata="{}",
            complexity=3,
            created_at=environment.context.started_at,
            status="selected",
        )
        monkeypatch.setattr(mutation_service, "load_research_context", lambda *args, **kwargs: research_context)
        monkeypatch.setattr(mutation_service, "persist_research_metadata", lambda *args, **kwargs: None)
        monkeypatch.setattr(repository, "get_top_alpha_records", lambda run_id, limit: [parent_record])
        monkeypatch.setattr(
            repository.alpha_history,
            "load_snapshot",
            lambda **kwargs: PatternMemorySnapshot(
                regime_key="regime",
                global_regime_key="global-regime",
                region="USA",
                top_parents=(
                    _memory_parent("parent-bad", "rank(close)"),
                    _memory_parent("parent-safe", "rank(volume)"),
                ),
            ),
        )
        monkeypatch.setattr(
            repository.alpha_history,
            "load_case_snapshot",
            lambda *args, **kwargs: _case_snapshot("rank(close)", "rank(volume)"),
        )
        _CaptureGuidedGenerator.last_parent_pool = None
        _CaptureGuidedGenerator.last_case_snapshot = None
        monkeypatch.setattr(mutation_service, "GuidedGenerator", _CaptureGuidedGenerator)

        mutation_service.mutate_and_persist(repository, config, environment, from_top=1, count=1)
    finally:
        repository.close()

    assert _CaptureGuidedGenerator.last_parent_pool is not None
    assert [parent.alpha_id for parent in _CaptureGuidedGenerator.last_parent_pool] == ["parent-safe"]
    assert _CaptureGuidedGenerator.last_case_snapshot is not None
    assert all("close" not in case.structural_signature.fields for case in _CaptureGuidedGenerator.last_case_snapshot.cases)


def test_brain_batch_service_uses_filtered_generation_field_registry(monkeypatch) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _config()
        environment = _environment("run-service", "service-blacklist")
        init_run(repository, config, environment, status="running")
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-close",
            rejection_reason='Attempted to use unknown variable "close".',
        )
        research_context = _research_context(_build_field_registry("close", "volume"))
        selection_service = _FakeSelectionService()
        service = BrainBatchService(repository, selection_service=selection_service)
        monkeypatch.setattr(service, "_get_research_context_provider", lambda config: _FakeResearchContextProvider(research_context))

        def fake_generate_fresh(*, field_registry, **kwargs):
            del kwargs
            assert not field_registry.contains("close")
            assert field_registry.contains("volume")
            return [], GenerationSessionStats()

        monkeypatch.setattr(service, "_generate_fresh_candidates", fake_generate_fresh)

        result = service.prepare_service_batch(
            config=config,
            environment=environment,
            count=1,
            mutation_parent_ids=None,
            round_index=1,
        )
    finally:
        repository.close()

    assert result.candidates == ()
    assert selection_service.last_field_registry is not None
    assert not selection_service.last_field_registry.contains("close")
    assert selection_service.last_field_registry.contains("volume")


def test_brain_batch_service_filters_blacklisted_memory_inputs(monkeypatch) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _config()
        config.adaptive_generation.enabled = True
        environment = _environment("run-service", "service-memory-blacklist")
        init_run(repository, config, environment, status="running")
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-close",
            rejection_reason='Attempted to use unknown variable "close".',
        )
        research_context = _research_context(_build_field_registry("close", "volume", "sector"))
        selection_service = _FakeSelectionService()
        service = BrainBatchService(repository, selection_service=selection_service)
        monkeypatch.setattr(service, "_get_research_context_provider", lambda config: _FakeResearchContextProvider(research_context))
        monkeypatch.setattr(
            repository.alpha_history,
            "load_snapshot",
            lambda **kwargs: PatternMemorySnapshot(
                regime_key="regime",
                global_regime_key="global-regime",
                region="USA",
                top_parents=(
                    _memory_parent("parent-bad", "rank(close)"),
                    _memory_parent("parent-safe", "rank(volume)"),
                ),
            ),
        )
        monkeypatch.setattr(
            repository.alpha_history,
            "load_case_snapshot",
            lambda *args, **kwargs: _case_snapshot("rank(close)", "rank(volume)"),
        )

        def fake_generate_mutation_candidates(*, snapshot, case_snapshot, **kwargs):
            assert [parent.alpha_id for parent in snapshot.top_parents] == ["parent-safe"]
            assert all("close" not in case.structural_signature.fields for case in case_snapshot.cases)
            return [], GenerationSessionStats()

        monkeypatch.setattr(service, "_generate_mutation_candidates", fake_generate_mutation_candidates)
        monkeypatch.setattr(service, "_generate_fresh_candidates", lambda **kwargs: ([], GenerationSessionStats()))

        service.prepare_service_batch(
            config=config,
            environment=environment,
            count=1,
            mutation_parent_ids={"parent-bad", "parent-safe"},
            round_index=1,
        )
    finally:
        repository.close()


def test_closed_loop_service_uses_filtered_generation_field_registry(monkeypatch, tmp_path) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = _config()
        config.runtime.progress_log_dir = str(tmp_path / "progress")
        environment = _environment("run-closed-loop", "closed-loop-blacklist")
        init_run(repository, config, environment, status="running")
        _save_result(
            repository,
            run_id=environment.context.run_id,
            job_id="job-close",
            rejection_reason='Attempted to use unknown variable "close".',
        )
        research_context = _research_context(_build_field_registry("close", "volume"))
        service = ClosedLoopService(repository)
        monkeypatch.setattr("services.closed_loop_service.load_research_context", lambda *args, **kwargs: research_context)
        monkeypatch.setattr("services.closed_loop_service.persist_research_metadata", lambda *args, **kwargs: None)

        def fake_generate_fresh(self, *, field_registry, **kwargs):
            del kwargs
            assert not field_registry.contains("close")
            assert field_registry.contains("volume")
            return []

        monkeypatch.setattr(ClosedLoopService, "_generate_fresh_candidates", fake_generate_fresh)

        summary = service.run(config=config, environment=environment)
    finally:
        repository.close()

    assert summary.status == "stopped_no_candidates"
