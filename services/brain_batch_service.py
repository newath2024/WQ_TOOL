from __future__ import annotations

from core.config import AppConfig
from features.registry import build_registry
from generator.engine import AlphaCandidate, AlphaGenerationEngine
from generator.guided_generator import GuidedGenerator
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot
from services.candidate_selection_service import CandidateSelectionService
from services.data_service import load_research_context, persist_research_metadata, resolve_field_registry
from services.evaluation_service import alpha_candidate_from_record
from services.models import CandidateScore, CommandEnvironment
from storage.repository import SQLiteRepository


class BrainBatchService:
    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        selection_service: CandidateSelectionService | None = None,
    ) -> None:
        self.repository = repository
        self.selection_service = selection_service

    def prepare_next_batch(
        self,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        count: int | None = None,
    ) -> tuple[list[AlphaCandidate], list[CandidateScore]]:
        candidates, selected, _ = self.prepare_service_batch(
            config=config,
            environment=environment,
            count=count,
            mutation_parent_ids=None,
        )
        return candidates, selected

    def prepare_service_batch(
        self,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        count: int | None = None,
        mutation_parent_ids: set[str] | None = None,
        round_index: int = 0,
    ) -> tuple[list[AlphaCandidate], list[CandidateScore], str]:
        research_context = load_research_context(config, environment, stage="brain-sim-data")
        active_regime_key = research_context.effective_regime_key or research_context.regime_key
        persist_research_metadata(self.repository, config, environment, research_context, round_index=round_index)
        field_registry = resolve_field_registry(config, research_context)
        registry = build_registry(
            config.generation.allowed_operators,
            operator_catalog_paths=config.generation.operator_catalog_paths,
        )
        snapshot = self.repository.alpha_history.load_snapshot(
            regime_key=active_regime_key,
            region=research_context.region,
            global_regime_key=research_context.global_regime_key,
            parent_pool_size=config.adaptive_generation.parent_pool_size,
            region_learning_config=config.adaptive_generation.region_learning,
            pattern_decay=config.adaptive_generation.pattern_decay,
            prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
        )
        case_snapshot = self.repository.alpha_history.load_case_snapshot(
            active_regime_key,
            region=research_context.region,
            global_regime_key=research_context.global_regime_key,
            region_learning_config=config.adaptive_generation.region_learning,
        )
        existing_normalized = self.repository.list_existing_normalized_expressions(environment.context.run_id)
        generation_count = count or config.loop.generation_batch_size
        mutation_candidates = self._generate_mutation_candidates(
            config=config,
            registry=registry,
            field_registry=field_registry,
            snapshot=snapshot,
            case_snapshot=case_snapshot,
            region_learning_context=research_context.region_learning_context,
            run_id=environment.context.run_id,
            mutation_parent_ids=mutation_parent_ids or set(),
            existing_normalized=existing_normalized,
        )
        fresh_budget = max(0, generation_count - len(mutation_candidates))
        fresh_candidates = self._generate_fresh_candidates(
            config=config,
            registry=registry,
            field_registry=field_registry,
            snapshot=snapshot,
            case_snapshot=case_snapshot,
            count=fresh_budget,
            existing_normalized=existing_normalized
            | {candidate.normalized_expression for candidate in mutation_candidates},
            memory_service=research_context.memory_service,
            region_learning_context=research_context.region_learning_context,
        )
        candidates = [*mutation_candidates, *fresh_candidates]
        self.repository.save_alpha_candidates(environment.context.run_id, candidates)
        selector = self.selection_service or CandidateSelectionService(
            research_context.memory_service,
            repository=self.repository,
            adaptive_config=config.adaptive_generation,
        )
        selector.configure_runtime(repository=self.repository, adaptive_config=config.adaptive_generation)
        pre_sim_result = selector.run_pre_sim_pipeline(
            candidates,
            snapshot=snapshot,
            field_registry=field_registry,
            batch_size=config.loop.simulation_batch_size,
            min_pattern_support=config.adaptive_generation.min_pattern_support,
            rejection_filters=config.loop.rejection_filters,
            case_snapshot=case_snapshot,
            diversity_config=config.adaptive_generation.diversity,
            run_id=environment.context.run_id,
            round_index=round_index,
            legacy_regime_key=research_context.regime_key,
            global_regime_key=research_context.global_regime_key,
            effective_regime_key=active_regime_key,
        )
        return candidates, list(pre_sim_result.selected), active_regime_key

    def _generate_fresh_candidates(
        self,
        *,
        config: AppConfig,
        registry,
        field_registry,
        snapshot: PatternMemorySnapshot,
        case_snapshot,
        count: int,
        existing_normalized: set[str],
        memory_service,
        region_learning_context,
    ) -> list[AlphaCandidate]:
        if count <= 0:
            return []
        if config.adaptive_generation.enabled:
            engine = GuidedGenerator(
                generation_config=config.generation,
                adaptive_config=config.adaptive_generation,
                registry=registry,
                memory_service=memory_service,
                field_registry=field_registry,
                region_learning_context=region_learning_context,
            )
            return engine.generate(
                count=count,
                snapshot=snapshot,
                existing_normalized=existing_normalized,
                case_snapshot=case_snapshot,
            )
        engine = AlphaGenerationEngine(
            config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=region_learning_context,
        )
        return engine.generate(count=count, existing_normalized=existing_normalized, case_snapshot=case_snapshot)

    def _generate_mutation_candidates(
        self,
        *,
        config: AppConfig,
        registry,
        field_registry,
        snapshot: PatternMemorySnapshot,
        case_snapshot,
        region_learning_context,
        run_id: str,
        mutation_parent_ids: set[str],
        existing_normalized: set[str],
    ) -> list[AlphaCandidate]:
        if not mutation_parent_ids:
            return []
        mutation_learning_records = self.repository.list_mutation_outcomes(effective_regime_key=snapshot.regime_key)
        mutation_budget = max(
            1,
            min(
                config.generation.mutation_count,
                len(mutation_parent_ids) * config.loop.max_children_per_parent,
            ),
        )
        if config.adaptive_generation.enabled:
            parent_pool = [parent for parent in snapshot.top_parents if parent.alpha_id in mutation_parent_ids]
            if not parent_pool:
                return []
            engine = GuidedGenerator(
                generation_config=config.generation,
                adaptive_config=config.adaptive_generation,
                registry=registry,
                memory_service=self.selection_service.memory_service if self.selection_service else PatternMemoryService(),
                field_registry=field_registry,
                region_learning_context=region_learning_context,
                mutation_learning_records=mutation_learning_records,
            )
            return engine.generate_mutations(
                count=mutation_budget,
                snapshot=snapshot,
                parent_pool=parent_pool,
                existing_normalized=existing_normalized,
                case_snapshot=case_snapshot,
            )

        parent_refs_map = self.repository.get_parent_refs(run_id)
        parent_records = [
            record
            for record in self.repository.list_alpha_records(run_id)
            if record.alpha_id in mutation_parent_ids
        ]
        parents = [alpha_candidate_from_record(record, parent_refs=parent_refs_map.get(record.alpha_id)) for record in parent_records]
        if not parents:
            return []
        engine = AlphaGenerationEngine(
            config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=region_learning_context,
            mutation_learning_records=mutation_learning_records,
        )
        return engine.generate_mutations(
            parents=parents,
            count=mutation_budget,
            existing_normalized=existing_normalized,
            case_snapshot=case_snapshot,
        )
