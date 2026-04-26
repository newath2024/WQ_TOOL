from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime

from core.config import AppConfig
from core.quality_score import MultiObjectiveQualityScorer
from features.registry import build_registry
from generator.engine import AlphaCandidate, AlphaGenerationEngine, GenerationSessionStats
from generator.guided_generator import GuidedGenerator
from generator.seed_utils import derive_generation_seed
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot
from services.candidate_selection_service import CandidateSelectionService
from services.data_service import (
    CachedResearchContextProvider,
    apply_local_validation_field_penalties,
    build_generation_guardrails,
    filter_generation_alpha_records,
    filter_generation_case_snapshot,
    filter_generation_pattern_snapshot,
    resolve_generation_field_registry,
    sanitize_generation_research_context,
)
from services.evaluation_service import alpha_candidate_from_record
from services.models import BatchPreparationResult, CandidateScore, CommandEnvironment
from services.progress_log import append_progress_event
from services.quality_polisher import QualityPolishStats, QualityPolisher
from services.recipe_guided_generator import RecipeGuidedGenerator, RecipeGuidedStats
from storage.models import StageMetricRecord
from storage.repository import SQLiteRepository


class BrainBatchService:
    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        selection_service: CandidateSelectionService | None = None,
        research_context_provider: CachedResearchContextProvider | None = None,
    ) -> None:
        self.repository = repository
        self.selection_service = selection_service
        self._research_context_provider = research_context_provider
        self._research_context_provider_signature: tuple[bool, int] | None = None

    def prepare_next_batch(
        self,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        count: int | None = None,
    ) -> tuple[list[AlphaCandidate], list[CandidateScore]]:
        result = self.prepare_service_batch(
            config=config,
            environment=environment,
            count=count,
            mutation_parent_ids=None,
        )
        return list(result.candidates), list(result.selected)

    def prepare_service_batch(
        self,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        count: int | None = None,
        mutation_parent_ids: set[str] | None = None,
        round_index: int = 0,
    ) -> BatchPreparationResult:
        provider = self._get_research_context_provider(config)
        cache_result = provider.load(config, environment, stage="brain-sim-data")
        research_context, blocked_fields = sanitize_generation_research_context(
            self.repository,
            config,
            cache_result.research_context,
            environment,
            stage="brain-sim-data",
        )
        research_context, local_validation_penalty = apply_local_validation_field_penalties(
            self.repository,
            config,
            research_context,
            environment,
            stage="brain-sim-data",
            before_round_index=round_index,
        )
        active_regime_key = research_context.effective_regime_key or research_context.regime_key
        persistence = provider.persist_metadata(
            self.repository,
            config,
            environment,
            cache_result,
            round_index=round_index,
            research_context_override=research_context,
            removed_field_names=blocked_fields,
        )
        resolve_field_registry_started = time.perf_counter()
        field_registry = resolve_generation_field_registry(
            self.repository,
            config,
            research_context,
            environment,
            stage="brain-sim-data",
        )
        resolve_field_registry_ms = (time.perf_counter() - resolve_field_registry_started) * 1000.0
        blocked_field_set = set(blocked_fields)
        generation_guardrails = build_generation_guardrails(self.repository, config, field_registry)
        registry = build_registry(
            config.generation.allowed_operators,
            operator_catalog_paths=config.generation.operator_catalog_paths,
        )
        snapshot = filter_generation_pattern_snapshot(
            self.repository.alpha_history.load_snapshot(
                regime_key=active_regime_key,
                region=research_context.region,
                global_regime_key=research_context.global_regime_key,
                parent_pool_size=config.adaptive_generation.parent_pool_size,
                region_learning_config=config.adaptive_generation.region_learning,
                pattern_decay=config.adaptive_generation.pattern_decay,
                prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
            ),
            blocked_fields=blocked_field_set,
        )
        case_snapshot = filter_generation_case_snapshot(
            self.repository.alpha_history.load_case_snapshot(
                active_regime_key,
                region=research_context.region,
                global_regime_key=research_context.global_regime_key,
                region_learning_config=config.adaptive_generation.region_learning,
            ),
            blocked_fields=blocked_field_set,
        )
        existing_normalized = self.repository.list_existing_normalized_expressions(environment.context.run_id)
        generation_count = count or config.loop.generation_batch_size
        mutation_candidates, mutation_stats = self._generate_mutation_candidates(
            config=config,
            registry=registry,
            field_registry=field_registry,
            snapshot=snapshot,
            case_snapshot=case_snapshot,
            region_learning_context=research_context.region_learning_context,
            run_id=environment.context.run_id,
            mutation_parent_ids=mutation_parent_ids or set(),
            existing_normalized=existing_normalized,
            round_index=round_index,
            generation_guardrails=generation_guardrails,
            blocked_fields=blocked_field_set,
            field_penalty_multipliers=local_validation_penalty.multipliers,
        )
        self._tag_generation_source(mutation_candidates, source="mutation")
        remaining_after_mutation = max(0, generation_count - len(mutation_candidates))
        source_budget_plan, source_yield_scores = self._plan_generation_source_budgets(
            config=config,
            run_id=environment.context.run_id,
            round_index=round_index,
            generation_count=generation_count,
            mutation_count=len(mutation_candidates),
        )
        source_budget_caps = self._source_budget_capacities(
            config=config,
            generation_count=generation_count,
            mutation_count=len(mutation_candidates),
        )
        quality_polish_budget = min(
            remaining_after_mutation,
            int(source_budget_plan.get("quality_polish", 0)),
        )
        quality_polish_candidates, quality_polish_stats = self._generate_quality_polish_candidates(
            config=config,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=research_context.region_learning_context,
            run_id=environment.context.run_id,
            existing_normalized=existing_normalized
            | {candidate.normalized_expression for candidate in mutation_candidates},
            round_index=round_index,
            generation_guardrails=generation_guardrails,
            blocked_fields=blocked_field_set,
            field_penalty_multipliers=local_validation_penalty.multipliers,
            count=quality_polish_budget,
        )
        quality_polish_shortfall = max(0, int(quality_polish_budget) - len(quality_polish_candidates))
        remaining_after_polish = max(0, generation_count - len(mutation_candidates) - len(quality_polish_candidates))
        planned_recipe_budget = int(source_budget_plan.get("recipe_guided", 0))
        recipe_guided_headroom = max(0, int(source_budget_caps.get("recipe_guided", 0)) - planned_recipe_budget)
        recipe_guided_budget = min(
            remaining_after_polish,
            planned_recipe_budget + min(quality_polish_shortfall, recipe_guided_headroom),
        )
        recipe_guided_candidates, recipe_guided_stats = self._generate_recipe_guided_candidates(
            config=config,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=research_context.region_learning_context,
            run_id=environment.context.run_id,
            existing_normalized=existing_normalized
            | {candidate.normalized_expression for candidate in mutation_candidates}
            | {candidate.normalized_expression for candidate in quality_polish_candidates},
            round_index=round_index,
            generation_guardrails=generation_guardrails,
            blocked_fields=blocked_field_set,
            field_penalty_multipliers=local_validation_penalty.multipliers,
            count=recipe_guided_budget,
        )
        remaining_after_recipe = max(
            0,
            generation_count
            - len(mutation_candidates)
            - len(quality_polish_candidates)
            - len(recipe_guided_candidates),
        )
        planned_fresh_budget = int(source_budget_plan.get("fresh", 0))
        fresh_spillover_allowed = int(
            generation_count * float(config.adaptive_generation.recipe_generation.fresh_spillover_fraction)
        )
        fresh_budget = min(
            remaining_after_recipe,
            int(source_budget_caps.get("fresh", 0)),
            planned_fresh_budget + max(0, fresh_spillover_allowed),
        )
        fresh_spillover_used = max(0, int(fresh_budget) - min(int(planned_fresh_budget), remaining_after_recipe))
        fresh_candidates, fresh_stats = self._generate_fresh_candidates(
            config=config,
            registry=registry,
            field_registry=field_registry,
            snapshot=snapshot,
            case_snapshot=case_snapshot,
            count=fresh_budget,
            existing_normalized=(
                existing_normalized
                | {candidate.normalized_expression for candidate in mutation_candidates}
                | {candidate.normalized_expression for candidate in quality_polish_candidates}
                | {candidate.normalized_expression for candidate in recipe_guided_candidates}
            ),
            memory_service=research_context.memory_service,
            region_learning_context=research_context.region_learning_context,
            run_id=environment.context.run_id,
            round_index=round_index,
            generation_guardrails=generation_guardrails,
            field_penalty_multipliers=local_validation_penalty.multipliers,
        )
        self._tag_generation_source(fresh_candidates, source="fresh")
        candidates = [*mutation_candidates, *quality_polish_candidates, *recipe_guided_candidates, *fresh_candidates]
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
        generation_stats = self._merge_generation_stats(mutation_stats, fresh_stats)
        generation_metrics = generation_stats.to_metrics(
            generated_count=len(candidates),
            selected_for_simulation=len(pre_sim_result.selected),
            include_debug_samples=logging.getLogger(__name__).isEnabledFor(logging.DEBUG),
        )
        quality_polish_stats.selected_count = sum(
            1
            for score in pre_sim_result.selected
            if getattr(score.candidate, "generation_mode", "") == "quality_polish"
        )
        quality_polish_stats.turnover_repair_selected = sum(
            1
            for score in pre_sim_result.selected
            if getattr(score.candidate, "generation_metadata", {}).get("repair_reason") == "turnover"
        )
        recipe_guided_stats.selected_count = sum(
            1
            for score in pre_sim_result.selected
            if getattr(score.candidate, "generation_mode", "") == "recipe_guided"
        )
        for score in pre_sim_result.selected:
            if getattr(score.candidate, "generation_mode", "") != "recipe_guided":
                continue
            bucket_id = str(score.candidate.generation_metadata.get("search_bucket_id") or "").strip()
            if bucket_id:
                recipe_guided_stats.selected_by_bucket[bucket_id] += 1
        self._merge_quality_polish_metrics(generation_metrics, quality_polish_stats)
        self._merge_recipe_guided_metrics(generation_metrics, recipe_guided_stats)
        generation_metrics["source_budget_allocations"] = {
            key: int(value)
            for key, value in source_budget_plan.items()
            if int(value) > 0
        }
        generation_metrics["source_yield_scores"] = {
            key: round(float(value), 6)
            for key, value in source_yield_scores.items()
        }
        generation_metrics["source_budget_caps"] = {
            key: int(value)
            for key, value in source_budget_caps.items()
            if int(value) > 0
        }
        generation_metrics["fresh_spillover_used"] = int(fresh_spillover_used)
        generation_metrics["source_unfilled_budget"] = max(0, int(generation_count) - len(candidates))
        generation_metrics["source_generated_counts"] = {
            "quality_polish": len(quality_polish_candidates),
            "recipe_guided": len(recipe_guided_candidates),
            "fresh": len(fresh_candidates),
        }
        generation_metrics["source_selected_counts"] = dict(
            Counter(
                str(score.candidate.generation_metadata.get("generation_source") or "")
                for score in pre_sim_result.selected
                if str(score.candidate.generation_metadata.get("generation_source") or "")
            )
        )
        generation_metrics.update(
            {
                "load_research_context_ms": round(cache_result.profile.load_research_context_ms, 3),
                "build_field_registry_ms": round(cache_result.profile.build_field_registry_ms, 3),
                "resolve_field_registry_ms": round(resolve_field_registry_ms, 3),
                "prepare_context_ms": round(cache_result.profile.prepare_context_ms, 3),
                "research_context_cache_hit": cache_result.profile.cache_hit,
                "research_context_cache_reason": cache_result.profile.cache_reason,
                "research_context_cache_key": cache_result.profile.cache_key[:12],
                "metadata_dataset_summary_persisted": persistence["dataset_summary_persisted"],
                "metadata_field_catalog_persisted": persistence["field_catalog_persisted"],
                "metadata_run_field_scores_persisted": persistence["run_field_scores_persisted"],
                **local_validation_penalty.to_metrics(),
            }
        )
        self.repository.save_stage_metrics(
            [
                StageMetricRecord(
                    run_id=environment.context.run_id,
                    round_index=round_index,
                    stage="generation",
                    metrics_json=json.dumps(generation_metrics, sort_keys=True),
                    created_at=datetime.now(UTC).isoformat(),
                )
            ]
        )
        append_progress_event(
            config,
            environment,
            event="batch_prepared",
            stage="generation",
            status="prepared",
            round_index=round_index,
            payload={
                "candidate_count": len(candidates),
                "selected_count": len(pre_sim_result.selected),
                "archived_count": len(pre_sim_result.archived),
                "regime_key": active_regime_key,
                "generation_stage_metrics": generation_metrics,
            },
        )
        return BatchPreparationResult(
            candidates=tuple(candidates),
            selected=tuple(pre_sim_result.selected),
            regime_key=active_regime_key,
            validated_count=int(pre_sim_result.stage_metrics.get("kept_after_dedup", len(candidates))),
            archived_count=len(pre_sim_result.archived),
            mutated_children_count=len(mutation_candidates),
            generation_stage_metrics=generation_metrics,
        )

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
        run_id: str,
        round_index: int,
        generation_guardrails,
        field_penalty_multipliers: dict[str, float] | None = None,
    ) -> tuple[list[AlphaCandidate], GenerationSessionStats]:
        if count <= 0:
            return [], GenerationSessionStats()
        scoped_generation = self._generation_config_for_scope(
            config.generation,
            run_id=run_id,
            round_index=round_index,
            scope="fresh",
        )
        if config.adaptive_generation.enabled:
            engine = GuidedGenerator(
                generation_config=scoped_generation,
                adaptive_config=config.adaptive_generation,
                registry=registry,
                memory_service=memory_service,
                field_registry=field_registry,
                region_learning_context=region_learning_context,
                generation_guardrails=generation_guardrails,
                field_penalty_multipliers=field_penalty_multipliers,
            )
            candidates = engine.generate(
                count=count,
                snapshot=snapshot,
                existing_normalized=existing_normalized,
                case_snapshot=case_snapshot,
            )
            return candidates, engine.last_generation_stats or GenerationSessionStats()
        engine = AlphaGenerationEngine(
            config=scoped_generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=region_learning_context,
            generation_guardrails=generation_guardrails,
            field_penalty_multipliers=field_penalty_multipliers,
        )
        candidates = engine.generate(count=count, existing_normalized=existing_normalized, case_snapshot=case_snapshot)
        return candidates, engine.last_generation_stats or GenerationSessionStats()

    def _generate_quality_polish_candidates(
        self,
        *,
        config: AppConfig,
        registry,
        field_registry,
        region_learning_context,
        run_id: str,
        existing_normalized: set[str],
        round_index: int,
        generation_guardrails,
        blocked_fields: set[str],
        field_penalty_multipliers: dict[str, float] | None,
        count: int,
    ) -> tuple[list[AlphaCandidate], QualityPolishStats]:
        if count <= 0:
            return [], QualityPolishStats(enabled=bool(config.adaptive_generation.quality_optimization.enabled))
        scoped_generation = self._generation_config_for_scope(
            config.generation,
            run_id=run_id,
            round_index=round_index,
            scope="quality_polish",
        )
        result = QualityPolisher(self.repository).generate(
            config=config.adaptive_generation.quality_optimization,
            adaptive_config=config.adaptive_generation,
            generation_config=scoped_generation,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=region_learning_context,
            generation_guardrails=generation_guardrails,
            field_penalty_multipliers=field_penalty_multipliers,
            blocked_fields=blocked_fields,
            existing_normalized=existing_normalized,
            run_id=run_id,
            round_index=round_index,
            count=count,
        )
        return result.candidates, result.stats

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
        round_index: int,
        generation_guardrails,
        blocked_fields: set[str],
        field_penalty_multipliers: dict[str, float] | None = None,
    ) -> tuple[list[AlphaCandidate], GenerationSessionStats]:
        if not mutation_parent_ids:
            return [], GenerationSessionStats()
        mutation_learning_records = self.repository.list_mutation_outcomes(effective_regime_key=snapshot.regime_key)
        scoped_generation = self._generation_config_for_scope(
            config.generation,
            run_id=run_id,
            round_index=round_index,
            scope="mutation",
        )
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
                return [], GenerationSessionStats()
            engine = GuidedGenerator(
                generation_config=scoped_generation,
                adaptive_config=config.adaptive_generation,
                registry=registry,
                memory_service=self.selection_service.memory_service if self.selection_service else PatternMemoryService(),
                field_registry=field_registry,
                region_learning_context=region_learning_context,
                mutation_learning_records=mutation_learning_records,
                generation_guardrails=generation_guardrails,
                field_penalty_multipliers=field_penalty_multipliers,
            )
            candidates = engine.generate_mutations(
                count=mutation_budget,
                snapshot=snapshot,
                parent_pool=parent_pool,
                existing_normalized=existing_normalized,
                case_snapshot=case_snapshot,
            )
            return candidates, engine.last_generation_stats or GenerationSessionStats()

        parent_refs_map = self.repository.get_parent_refs(run_id)
        parent_records = [
            record
            for record in self.repository.list_alpha_records(run_id)
            if record.alpha_id in mutation_parent_ids
        ]
        parent_records = filter_generation_alpha_records(parent_records, blocked_fields=blocked_fields)
        parents = [alpha_candidate_from_record(record, parent_refs=parent_refs_map.get(record.alpha_id)) for record in parent_records]
        if not parents:
            return [], GenerationSessionStats()
        engine = AlphaGenerationEngine(
            config=scoped_generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=region_learning_context,
            mutation_learning_records=mutation_learning_records,
            generation_guardrails=generation_guardrails,
            field_penalty_multipliers=field_penalty_multipliers,
        )
        candidates = engine.generate_mutations(
            parents=parents,
            count=mutation_budget,
            existing_normalized=existing_normalized,
            case_snapshot=case_snapshot,
        )
        return candidates, engine.last_generation_stats or GenerationSessionStats()

    def _generate_recipe_guided_candidates(
        self,
        *,
        config: AppConfig,
        registry,
        field_registry,
        region_learning_context,
        run_id: str,
        existing_normalized: set[str],
        round_index: int,
        generation_guardrails,
        blocked_fields: set[str],
        field_penalty_multipliers: dict[str, float] | None,
        count: int,
    ) -> tuple[list[AlphaCandidate], RecipeGuidedStats]:
        if count <= 0:
            return [], RecipeGuidedStats(enabled=bool(config.adaptive_generation.recipe_generation.enabled))
        scoped_generation = self._generation_config_for_scope(
            config.generation,
            run_id=run_id,
            round_index=round_index,
            scope="recipe_guided",
        )
        result = RecipeGuidedGenerator(self.repository).generate(
            config=config.adaptive_generation.recipe_generation,
            adaptive_config=config.adaptive_generation,
            generation_config=scoped_generation,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=region_learning_context,
            generation_guardrails=generation_guardrails,
            field_penalty_multipliers=field_penalty_multipliers,
            blocked_fields=blocked_fields,
            existing_normalized=existing_normalized,
            run_id=run_id,
            round_index=round_index,
            count=count,
        )
        return result.candidates, result.stats

    def _plan_generation_source_budgets(
        self,
        *,
        config: AppConfig,
        run_id: str,
        round_index: int,
        generation_count: int,
        mutation_count: int,
    ) -> tuple[dict[str, int], dict[str, float]]:
        available_pool = max(0, int(generation_count) - int(mutation_count))
        if available_pool <= 0:
            return {"quality_polish": 0, "recipe_guided": 0, "fresh": 0}, {
                "quality_polish": _neutral_yield_score(),
                "recipe_guided": _neutral_yield_score(),
                "fresh": _neutral_yield_score(),
            }

        recipe_config = config.adaptive_generation.recipe_generation
        capacities = self._source_budget_capacities(
            config=config,
            generation_count=generation_count,
            mutation_count=mutation_count,
        )
        static_quality = min(available_pool, capacities["quality_polish"])
        static_recipe = min(max(0, available_pool - static_quality), capacities["recipe_guided"])
        static_fresh = min(
            capacities["fresh"],
            max(0, available_pool - static_quality - static_recipe),
        )
        static_plan = {
            "quality_polish": int(static_quality),
            "recipe_guided": int(static_recipe),
            "fresh": int(static_fresh),
        }
        neutral_scores = {
            "quality_polish": _neutral_yield_score(),
            "recipe_guided": _neutral_yield_score(),
            "fresh": _neutral_yield_score(),
        }
        if not bool(recipe_config.dynamic_budget_enabled):
            return static_plan, neutral_scores

        yield_scores = self._load_source_yield_scores(
            run_id=run_id,
            round_index=round_index,
            config=config,
        )
        floor_targets = self._source_floor_targets(
            total_budget=available_pool,
            capacities=capacities,
            floor_fractions=dict(recipe_config.source_exploration_floor_fractions or {}),
            weights=yield_scores,
        )
        dynamic_plan = _allocate_integer_budget(
            total_budget=available_pool,
            capacities=capacities,
            weights=yield_scores,
            floor_targets=floor_targets,
        )
        for source in ("quality_polish", "recipe_guided", "fresh"):
            dynamic_plan.setdefault(source, 0)
        return dynamic_plan, yield_scores

    @staticmethod
    def _source_budget_capacities(
        *,
        config: AppConfig,
        generation_count: int,
        mutation_count: int,
    ) -> dict[str, int]:
        available_pool = max(0, int(generation_count) - int(mutation_count))
        recipe_config = config.adaptive_generation.recipe_generation
        polish_config = config.adaptive_generation.quality_optimization
        quality_cap = min(
            available_pool,
            int(polish_config.max_polish_candidates_per_round) if bool(polish_config.enabled) else 0,
            max(
                1 if bool(polish_config.enabled) and float(polish_config.polish_budget_fraction) > 0 else 0,
                int(generation_count * float(polish_config.polish_budget_fraction)),
            ),
        )
        recipe_cap = min(
            available_pool,
            int(recipe_config.max_recipe_candidates_per_round) if bool(recipe_config.enabled) else 0,
            max(
                1 if bool(recipe_config.enabled) and float(recipe_config.recipe_budget_fraction) > 0 else 0,
                int(generation_count * float(recipe_config.recipe_budget_fraction)),
            ),
        )
        fresh_cap = min(
            available_pool,
            max(0, int(generation_count * float(recipe_config.max_fresh_budget_fraction))),
        )
        return {
            "quality_polish": max(0, int(quality_cap)),
            "recipe_guided": max(0, int(recipe_cap)),
            "fresh": max(0, int(fresh_cap)),
        }

    def _load_source_yield_scores(
        self,
        *,
        run_id: str,
        round_index: int,
        config: AppConfig,
    ) -> dict[str, float]:
        recipe_config = config.adaptive_generation.recipe_generation
        stage_rows = self.repository.list_recent_generation_stage_metrics(
            run_id,
            limit=int(recipe_config.yield_lookback_rounds),
            before_round_index=int(round_index),
        )
        result_rows = self.repository.list_generation_result_rows(
            run_id=run_id,
            before_round_index=int(round_index),
            lookback_rounds=int(recipe_config.yield_lookback_rounds),
        )
        generated_counts: Counter[str] = Counter()
        selected_counts: Counter[str] = Counter()
        for row in stage_rows:
            metrics = json.loads(row.get("metrics_json") or "{}")
            generated_counts.update(
                {
                    str(key): int(value or 0)
                    for key, value in dict(metrics.get("source_generated_counts") or {}).items()
                }
            )
            selected_counts.update(
                {
                    str(key): int(value or 0)
                    for key, value in dict(metrics.get("source_selected_counts") or {}).items()
                }
            )

        completed_counts: Counter[str] = Counter()
        positive_quality_counts: Counter[str] = Counter()
        quality_sums: Counter[str] = Counter()
        for row in result_rows:
            source = _generation_source_for_row(row)
            if source not in {"quality_polish", "recipe_guided", "fresh"}:
                continue
            if str(row.get("status") or "") != "completed":
                continue
            completed_counts[source] += 1
            quality_score = _result_quality_score(row)
            quality_sums[source] += float(quality_score)
            if float(quality_score) > 0.0:
                positive_quality_counts[source] += 1

        scores: dict[str, float] = {}
        for source in ("quality_polish", "recipe_guided", "fresh"):
            generated_support = int(generated_counts.get(source, 0))
            completed_support = int(completed_counts.get(source, 0))
            if (
                generated_support < int(recipe_config.dynamic_budget_min_generated_support)
                or completed_support < int(recipe_config.dynamic_budget_min_completed_support)
            ):
                scores[source] = _neutral_yield_score()
                continue
            selected_rate = float(selected_counts.get(source, 0)) / max(1, generated_support)
            positive_quality_rate = float(positive_quality_counts.get(source, 0)) / max(1, completed_support)
            avg_quality_score = float(quality_sums.get(source, 0.0)) / max(1, completed_support)
            raw_score = _yield_score(
                selected_rate=selected_rate,
                positive_quality_rate=positive_quality_rate,
                avg_quality_score=avg_quality_score,
            )
            scores[source] = _adjusted_yield_score(
                raw_score=raw_score,
                strength=float(recipe_config.source_reallocation_strength),
            )
        return scores

    @staticmethod
    def _source_floor_targets(
        *,
        total_budget: int,
        capacities: dict[str, int],
        floor_fractions: dict[str, float],
        weights: dict[str, float],
    ) -> dict[str, int]:
        floors = {key: 0 for key in capacities}
        eligible = [key for key, cap in capacities.items() if int(cap) > 0 and float(floor_fractions.get(key, 0.0)) > 0.0]
        if total_budget <= 0 or not eligible:
            return floors
        if total_budget < len(eligible):
            ranked = sorted(eligible, key=lambda key: (-float(floor_fractions.get(key, 0.0)), -float(weights.get(key, 0.0)), key))
            for key in ranked[:total_budget]:
                floors[key] = 1
            return floors
        for key in eligible:
            floors[key] = min(
                int(capacities.get(key, 0)),
                max(1, int(total_budget * float(floor_fractions.get(key, 0.0)))),
            )
        return floors

    @staticmethod
    def _merge_quality_polish_metrics(
        generation_metrics: dict[str, object],
        quality_polish_stats: QualityPolishStats,
    ) -> None:
        polish_metrics = quality_polish_stats.to_metrics()
        generation_metrics.update(polish_metrics)
        generation_metrics["attempt_count"] = int(generation_metrics.get("attempt_count") or 0) + int(
            polish_metrics.get("quality_polish_attempt_count") or 0
        )
        generation_metrics["attempt_success_count"] = int(generation_metrics.get("attempt_success_count") or 0) + int(
            polish_metrics.get("quality_polish_success_count") or 0
        )
        generation_metrics["generation_total_ms"] = round(
            float(generation_metrics.get("generation_total_ms") or 0.0)
            + float(polish_metrics.get("quality_polish_generation_total_ms") or 0.0),
            3,
        )
        failure_counts = Counter(dict(generation_metrics.get("failure_reason_counts") or {}))
        failure_counts.update(dict(polish_metrics.get("quality_polish_failure_reason_counts") or {}))
        generation_metrics["failure_reason_counts"] = dict(failure_counts)
        generation_metrics["top_fail_reasons"] = dict(failure_counts.most_common(5))
        generation_metrics["parse_fail_count"] = int(failure_counts.get("parse_failed", 0))
        generation_metrics["validate_fail_count"] = int(
            sum(count for reason, count in failure_counts.items() if str(reason).startswith("validation_"))
        )
        generation_metrics["complexity_fail_count"] = int(failure_counts.get("complexity_exceeded", 0))
        generation_metrics["redundancy_fail_count"] = int(failure_counts.get("redundant_expression", 0))
        generation_metrics["duplicate_fail_count"] = int(
            failure_counts.get("duplicate_normalized_expression", 0)
            + failure_counts.get("structural_duplicate_expression", 0)
        )
        generation_metrics["normalized_duplicate_count"] = int(
            failure_counts.get("duplicate_normalized_expression", 0)
        )
        generation_metrics["structural_duplicate_count"] = int(
            failure_counts.get("structural_duplicate_expression", 0)
        )

    @staticmethod
    def _merge_recipe_guided_metrics(
        generation_metrics: dict[str, object],
        recipe_guided_stats: RecipeGuidedStats,
    ) -> None:
        recipe_metrics = recipe_guided_stats.to_metrics()
        generation_metrics.update(recipe_metrics)
        generation_metrics["attempt_count"] = int(generation_metrics.get("attempt_count") or 0) + int(
            recipe_metrics.get("recipe_guided_attempt_count") or 0
        )
        generation_metrics["attempt_success_count"] = int(generation_metrics.get("attempt_success_count") or 0) + int(
            recipe_metrics.get("recipe_guided_success_count") or 0
        )
        generation_metrics["generation_total_ms"] = round(
            float(generation_metrics.get("generation_total_ms") or 0.0)
            + float(recipe_metrics.get("recipe_guided_generation_total_ms") or 0.0),
            3,
        )
        failure_counts = Counter(dict(generation_metrics.get("failure_reason_counts") or {}))
        failure_counts.update(dict(recipe_metrics.get("recipe_guided_failure_reason_counts") or {}))
        generation_metrics["failure_reason_counts"] = dict(failure_counts)
        generation_metrics["top_fail_reasons"] = dict(failure_counts.most_common(5))
        generation_metrics["parse_fail_count"] = int(failure_counts.get("parse_failed", 0))
        generation_metrics["validate_fail_count"] = int(
            sum(count for reason, count in failure_counts.items() if str(reason).startswith("validation_"))
        )
        generation_metrics["complexity_fail_count"] = int(failure_counts.get("complexity_exceeded", 0))
        generation_metrics["redundancy_fail_count"] = int(failure_counts.get("redundant_expression", 0))
        generation_metrics["duplicate_fail_count"] = int(
            failure_counts.get("duplicate_normalized_expression", 0)
            + failure_counts.get("structural_duplicate_expression", 0)
        )
        generation_metrics["normalized_duplicate_count"] = int(
            failure_counts.get("duplicate_normalized_expression", 0)
        )
        generation_metrics["structural_duplicate_count"] = int(
            failure_counts.get("structural_duplicate_expression", 0)
        )

    def _get_research_context_provider(self, config: AppConfig) -> CachedResearchContextProvider:
        signature = (
            bool(config.service.research_context_cache_enabled),
            int(config.service.research_context_cache_ttl_seconds),
        )
        if self._research_context_provider is None or self._research_context_provider_signature != signature:
            self._research_context_provider = CachedResearchContextProvider(
                enabled=config.service.research_context_cache_enabled,
                ttl_seconds=config.service.research_context_cache_ttl_seconds,
            )
            self._research_context_provider_signature = signature
        return self._research_context_provider

    def _merge_generation_stats(self, *stats: GenerationSessionStats) -> GenerationSessionStats:
        merged = GenerationSessionStats()
        for item in stats:
            merged.prepare_validation_context_ms += item.prepare_validation_context_ms
            merged.generation_total_ms += item.generation_total_ms
            merged.exploit_phase_ms += item.exploit_phase_ms
            merged.explore_phase_ms += item.explore_phase_ms
            merged.attempt_count += item.attempt_count
            merged.attempt_success_count += item.attempt_success_count
            merged.exploit_attempt_count += item.exploit_attempt_count
            merged.explore_attempt_count += item.explore_attempt_count
            merged.exploit_success_count += item.exploit_success_count
            merged.explore_success_count += item.explore_success_count
            merged.timeout_stop = merged.timeout_stop or item.timeout_stop
            merged.consecutive_failure_stop = merged.consecutive_failure_stop or item.consecutive_failure_stop
            merged.exploit_consecutive_failure_stop = (
                merged.exploit_consecutive_failure_stop or item.exploit_consecutive_failure_stop
            )
            merged.explore_consecutive_failure_stop = (
                merged.explore_consecutive_failure_stop or item.explore_consecutive_failure_stop
            )
            merged.explore_phase_entered = merged.explore_phase_entered or item.explore_phase_entered
            merged.validation_context_cache_hit = merged.validation_context_cache_hit or item.validation_context_cache_hit
            merged.pre_dedup_reject_count += item.pre_dedup_reject_count
            merged.failure_counts.update(item.failure_counts)
            for reason, field_counts in item.failure_field_counts.items():
                merged.failure_field_counts.setdefault(reason, Counter()).update(field_counts)
            for sample in item.validation_disallowed_field_samples:
                if sample in merged.validation_disallowed_field_samples:
                    continue
                if len(merged.validation_disallowed_field_samples) >= 10:
                    break
                merged.validation_disallowed_field_samples.append(dict(sample))
            merged.duplicate_by_mutation_mode.update(item.duplicate_by_mutation_mode)
            merged.duplicate_by_motif.update(item.duplicate_by_motif)
            merged.duplicate_by_operator_path.update(item.duplicate_by_operator_path)
            for reason, samples in item.failure_samples.items():
                merged_bucket = merged.failure_samples.setdefault(reason, [])
                for sample in samples:
                    if sample in merged_bucket or len(merged_bucket) >= 3:
                        continue
                    merged_bucket.append(sample)
        return merged

    @staticmethod
    def _tag_generation_source(candidates: list[AlphaCandidate], *, source: str) -> None:
        for candidate in candidates:
            candidate.generation_metadata.setdefault("generation_source", str(source))

    @staticmethod
    def _generation_config_for_scope(
        generation_config,
        *,
        run_id: str,
        round_index: int,
        scope: str,
    ):
        scoped_seed = derive_generation_seed(
            generation_config.random_seed,
            run_id=run_id,
            round_index=round_index,
            scope=scope,
        )
        return replace(generation_config, random_seed=scoped_seed)


def _neutral_yield_score() -> float:
    return 0.5


def _yield_score(*, selected_rate: float, positive_quality_rate: float, avg_quality_score: float) -> float:
    return float(
        0.50 * max(0.0, min(1.0, float(selected_rate)))
        + 0.35 * max(0.0, min(1.0, float(positive_quality_rate)))
        + 0.15 * max(0.0, float(avg_quality_score))
    )


def _adjusted_yield_score(*, raw_score: float, strength: float) -> float:
    neutral = _neutral_yield_score()
    return float(neutral + max(0.0, min(1.0, float(strength))) * (float(raw_score) - neutral))


def _allocate_integer_budget(
    *,
    total_budget: int,
    capacities: dict[str, int],
    weights: dict[str, float],
    floor_targets: dict[str, int],
) -> dict[str, int]:
    allocation = {key: 0 for key, cap in capacities.items() if int(cap) > 0}
    if total_budget <= 0 or not allocation:
        return allocation

    positive_floor_keys = [
        key for key in allocation if int(floor_targets.get(key, 0)) > 0 and int(capacities.get(key, 0)) > 0
    ]
    if total_budget < len(positive_floor_keys):
        ranked = sorted(positive_floor_keys, key=lambda key: (-float(weights.get(key, _neutral_yield_score())), key))
        for key in ranked[: int(total_budget)]:
            allocation[key] = 1
        return allocation

    for key in positive_floor_keys:
        allocation[key] = min(int(capacities.get(key, 0)), max(1, int(floor_targets.get(key, 0))))

    allocated = sum(allocation.values())
    remaining = max(0, int(total_budget) - allocated)
    capacity_remaining = {
        key: max(0, int(capacities.get(key, 0)) - int(allocation.get(key, 0)))
        for key in allocation
    }
    eligible = [key for key, cap in capacity_remaining.items() if cap > 0]
    if remaining <= 0 or not eligible:
        return allocation

    total_weight = sum(max(0.0, float(weights.get(key, _neutral_yield_score()))) for key in eligible)
    if total_weight <= 0.0:
        total_weight = float(len(eligible))
        base_weights = {key: 1.0 for key in eligible}
    else:
        base_weights = {key: max(0.0, float(weights.get(key, _neutral_yield_score()))) for key in eligible}

    extras = {key: 0 for key in eligible}
    remainders: list[tuple[float, str]] = []
    for key in eligible:
        ideal = float(remaining) * float(base_weights[key]) / float(total_weight)
        extra = min(int(capacity_remaining[key]), int(ideal))
        extras[key] = extra
        remainders.append((ideal - float(extra), key))

    used = sum(extras.values())
    leftover = max(0, remaining - used)
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if leftover <= 0:
            break
        available = max(0, capacity_remaining[key] - extras[key])
        if available <= 0:
            continue
        extras[key] += 1
        leftover -= 1

    for key, extra in extras.items():
        allocation[key] += int(extra)
    return allocation


def _decode_json_object(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _generation_source_for_row(row: dict[str, object]) -> str:
    metadata = _decode_json_object(row.get("generation_metadata"))
    explicit = str(metadata.get("generation_source") or "").strip()
    if explicit:
        return explicit
    generation_mode = str(row.get("generation_mode") or "").strip()
    if generation_mode in {"quality_polish", "recipe_guided"}:
        return generation_mode
    return "fresh"


def _result_quality_score(row: dict[str, object]) -> float:
    stored = row.get("quality_score")
    try:
        parsed = float(stored)
    except (TypeError, ValueError):
        parsed = 0.0
    if abs(parsed) > 1e-12:
        return parsed
    return float(
        MultiObjectiveQualityScorer.score(
            metrics={
                "fitness": row.get("fitness"),
                "sharpe": row.get("sharpe"),
                "turnover": row.get("turnover"),
                "drawdown": row.get("drawdown"),
                "returns": row.get("returns"),
                "margin": row.get("margin"),
            },
            submission_eligible=row.get("submission_eligible"),
            rejection_reason=row.get("rejection_reason"),
            status=str(row.get("status") or ""),
        )
    )
