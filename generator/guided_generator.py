from __future__ import annotations

import math
import random
import time

from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry
from features.registry import OperatorRegistry
from generator.guardrails import GenerationGuardrails
from generator.diversity_tracker import GenerationDiversityTracker
from generator.engine import AlphaCandidate, AlphaGenerationEngine, GenerationSessionStats
from generator.genome_builder import GenomeBuilder
from generator.grammar import MotifGrammar
from generator.mutation_policy import MutationPolicy
from generator.repair_policy import RepairPolicy
from memory.case_memory import CaseMemorySnapshot, CaseMemoryService
from memory.pattern_memory import MemoryParent, PatternMemoryService, PatternMemorySnapshot, RegionLearningContext


class GuidedGenerator:
    def __init__(
        self,
        generation_config: GenerationConfig,
        adaptive_config: AdaptiveGenerationConfig,
        registry: OperatorRegistry,
        memory_service: PatternMemoryService,
        field_registry: FieldRegistry,
        region_learning_context: RegionLearningContext | None = None,
        mutation_learning_records: list[dict] | None = None,
        generation_guardrails: GenerationGuardrails | None = None,
        field_penalty_multipliers: dict[str, float] | None = None,
    ) -> None:
        self.generation_config = generation_config
        self.adaptive_config = adaptive_config
        self.registry = registry
        self.memory_service = memory_service
        self.case_memory_service = CaseMemoryService()
        self.field_registry = field_registry
        self.base_engine = AlphaGenerationEngine(
            config=generation_config,
            adaptive_config=adaptive_config,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=region_learning_context,
            mutation_learning_records=mutation_learning_records,
            generation_guardrails=generation_guardrails,
            field_penalty_multipliers=field_penalty_multipliers,
        )
        self.random = random.Random(generation_config.random_seed)
        self.genome_builder = GenomeBuilder(
            generation_config=generation_config,
            adaptive_config=adaptive_config,
            registry=registry,
            field_registry=field_registry,
            seed=generation_config.random_seed + 101,
        )
        self.grammar = MotifGrammar()
        self.repair_policy = RepairPolicy(
            generation_config=generation_config,
            repair_config=adaptive_config.repair_policy,
            field_registry=field_registry,
            registry=registry,
        )
        self.mutation_policy = MutationPolicy(
            config=generation_config,
            adaptive_config=adaptive_config,
            memory_service=memory_service,
            mutation_learning_records=mutation_learning_records,
            randomizer_seed=generation_config.random_seed + 211,
            field_registry=field_registry,
            registry=registry,
            field_penalty_multipliers=field_penalty_multipliers,
        )
        self.last_generation_stats: GenerationSessionStats | None = None

    def generate(
        self,
        count: int,
        snapshot: PatternMemorySnapshot,
        existing_normalized: set[str] | None = None,
        case_snapshot: CaseMemorySnapshot | None = None,
    ) -> list[AlphaCandidate]:
        if not self.adaptive_config.enabled:
            candidates = self.base_engine.generate(
                count=count,
                existing_normalized=existing_normalized,
                case_snapshot=case_snapshot,
            )
            self.last_generation_stats = self.base_engine.last_generation_stats
            return candidates

        existing = set(existing_normalized or set())
        candidates: list[AlphaCandidate] = []
        exploration_count = max(1, int(math.ceil(count * self.adaptive_config.exploration_ratio)))
        exploitation_count = max(0, count - exploration_count)
        validation_ctx, prepare_ms, cache_hit = self.base_engine._get_or_build_validation_context()  # noqa: SLF001
        session = GenerationSessionStats(
            prepare_validation_context_ms=prepare_ms,
            validation_context_cache_hit=cache_hit,
        )
        diversity_tracker = GenerationDiversityTracker()
        generation_started = time.monotonic()
        max_attempts = self.base_engine._resolve_max_attempts(count, legacy_multiplier=25, minimum=80)  # noqa: SLF001
        max_generation_seconds = float(getattr(self.adaptive_config, "max_generation_seconds", 0.0) or 0.0)
        exploit_attempt_budget, reserved_explore_attempts = self._resolve_phase_attempt_budgets(max_attempts)
        exploit_time_budget = self._resolve_exploit_time_budget(max_generation_seconds)
        exploit_consecutive_failures = 0

        exploit_started = time.monotonic()
        if exploit_attempt_budget > 0 and (max_generation_seconds <= 0.0 or exploit_time_budget > 0.0):
            while len(candidates) < exploitation_count and session.exploit_attempt_count < exploit_attempt_budget:
                if self._phase_should_stop(
                    session=session,
                    phase="exploit",
                    generation_started=generation_started,
                    phase_started=exploit_started,
                    phase_attempt_count=session.exploit_attempt_count,
                    phase_attempt_budget=exploit_attempt_budget,
                    phase_time_budget=exploit_time_budget,
                    consecutive_failures=exploit_consecutive_failures,
                    candidate_count=len(candidates),
                    target_count=count,
                ):
                    break
                session.record_attempt("exploit")
                genome = self.genome_builder.build_guided_genome(
                    case_snapshot=case_snapshot,
                    explore=False,
                    diversity_tracker=diversity_tracker,
                )
                repaired, repair_actions = self.repair_policy.repair(genome)
                render = self.grammar.render(repaired)
                if not self.base_engine._is_parseable_expression(render.expression):  # noqa: SLF001
                    session.record_failure("parse_failed", expression=render.expression)
                    exploit_consecutive_failures += 1
                    continue
                metadata = self.base_engine._render_metadata(  # noqa: SLF001
                    render,
                    mutation_mode="exploit_local",
                    repair_actions=repair_actions,
                    memory_context=self.base_engine._memory_context_metadata(  # noqa: SLF001
                        pattern_snapshot=snapshot,
                        case_snapshot=case_snapshot,
                    ),
                )
                if self.base_engine._reject_pre_dedup_candidate(  # noqa: SLF001
                    session=session,
                    diversity_tracker=diversity_tracker,
                    existing_normalized=existing,
                    expression=render.expression,
                    generation_metadata=metadata,
                    operator_path=tuple(getattr(render, "operator_path", ()) or ()),
                    genome_hash=str(getattr(repaired, "stable_hash", "") or ""),
                    structural_key=str(getattr(render, "normalized_expression", render.expression) or ""),
                    normalized_expression=str(getattr(render, "normalized_expression", render.expression) or ""),
                    pre_dedup=True,
                ):
                    exploit_consecutive_failures += 1
                    continue
                result = self.base_engine._build_candidate_result(  # noqa: SLF001
                    expression=render.expression,
                    mode="guided_exploit",
                    parent_ids=(),
                    generation_metadata=metadata,
                    validation_ctx=validation_ctx,
                )
                candidate = result.candidate
                if candidate is None:
                    session.record_failure(
                        result.failure_reason,
                        expression=render.expression,
                        fields=result.failure_fields,
                    )
                    exploit_consecutive_failures += 1
                    continue
                if candidate.normalized_expression in existing:
                    self.base_engine._record_duplicate_reject(  # noqa: SLF001
                        session=session,
                        diversity_tracker=diversity_tracker,
                        reason="duplicate_normalized_expression",
                        expression=candidate.normalized_expression,
                        generation_metadata=candidate.generation_metadata,
                        operator_path=tuple(candidate.generation_metadata.get("operator_path") or candidate.operators_used),
                        pre_dedup=False,
                    )
                    exploit_consecutive_failures += 1
                    continue
                existing.add(candidate.normalized_expression)
                candidates.append(candidate)
                session.record_success("exploit")
                self.base_engine._record_diversity_success(diversity_tracker, candidate)  # noqa: SLF001
                exploit_consecutive_failures = 0
        session.exploit_phase_ms = (time.monotonic() - exploit_started) * 1000.0

        remaining_attempt_budget = max(0, max_attempts - session.attempt_count)
        remaining_time_budget = self._remaining_generation_seconds(
            max_generation_seconds=max_generation_seconds,
            generation_started=generation_started,
        )
        if (
            len(candidates) < count
            and not session.timeout_stop
            and reserved_explore_attempts > 0
            and remaining_attempt_budget > 0
            and (max_generation_seconds <= 0.0 or remaining_time_budget > 0.0)
        ):
            session.explore_phase_entered = True
            explore_started = time.monotonic()
            explore_consecutive_failures = 0
            while len(candidates) < count and session.attempt_count < max_attempts:
                if self._phase_should_stop(
                    session=session,
                    phase="explore",
                    generation_started=generation_started,
                    phase_started=explore_started,
                    phase_attempt_count=session.explore_attempt_count,
                    phase_attempt_budget=remaining_attempt_budget,
                    phase_time_budget=remaining_time_budget,
                    consecutive_failures=explore_consecutive_failures,
                    candidate_count=len(candidates),
                    target_count=count,
                ):
                    break
                session.record_attempt("explore")
                genome = self.genome_builder.build_guided_genome(
                    case_snapshot=case_snapshot,
                    explore=True,
                    diversity_tracker=diversity_tracker,
                )
                repaired, repair_actions = self.repair_policy.repair(genome)
                render = self.grammar.render(repaired)
                if not self.base_engine._is_parseable_expression(render.expression):  # noqa: SLF001
                    session.record_failure("parse_failed", expression=render.expression)
                    explore_consecutive_failures += 1
                    continue
                metadata = self.base_engine._render_metadata(  # noqa: SLF001
                    render,
                    mutation_mode="novelty",
                    repair_actions=repair_actions,
                    memory_context=self.base_engine._memory_context_metadata(  # noqa: SLF001
                        pattern_snapshot=snapshot,
                        case_snapshot=case_snapshot,
                    ),
                )
                if self.base_engine._reject_pre_dedup_candidate(  # noqa: SLF001
                    session=session,
                    diversity_tracker=diversity_tracker,
                    existing_normalized=existing,
                    expression=render.expression,
                    generation_metadata=metadata,
                    operator_path=tuple(getattr(render, "operator_path", ()) or ()),
                    genome_hash=str(getattr(repaired, "stable_hash", "") or ""),
                    structural_key=str(getattr(render, "normalized_expression", render.expression) or ""),
                    normalized_expression=str(getattr(render, "normalized_expression", render.expression) or ""),
                    pre_dedup=True,
                ):
                    explore_consecutive_failures += 1
                    continue
                result = self.base_engine._build_candidate_result(  # noqa: SLF001
                    expression=render.expression,
                    mode="guided_explore",
                    parent_ids=(),
                    generation_metadata=metadata,
                    validation_ctx=validation_ctx,
                )
                candidate = result.candidate
                if candidate is None:
                    session.record_failure(
                        result.failure_reason,
                        expression=render.expression,
                        fields=result.failure_fields,
                    )
                    explore_consecutive_failures += 1
                    continue
                if candidate.normalized_expression in existing:
                    self.base_engine._record_duplicate_reject(  # noqa: SLF001
                        session=session,
                        diversity_tracker=diversity_tracker,
                        reason="duplicate_normalized_expression",
                        expression=candidate.normalized_expression,
                        generation_metadata=candidate.generation_metadata,
                        operator_path=tuple(candidate.generation_metadata.get("operator_path") or candidate.operators_used),
                        pre_dedup=False,
                    )
                    explore_consecutive_failures += 1
                    continue
                existing.add(candidate.normalized_expression)
                candidates.append(candidate)
                session.record_success("explore")
                self.base_engine._record_diversity_success(diversity_tracker, candidate)  # noqa: SLF001
                explore_consecutive_failures = 0
            session.explore_phase_ms = (time.monotonic() - explore_started) * 1000.0

        session.generation_total_ms = (time.monotonic() - generation_started) * 1000.0
        self.last_generation_stats = session
        self.base_engine.last_generation_stats = session
        return candidates[:count]

    def generate_mutations(
        self,
        count: int,
        snapshot: PatternMemorySnapshot,
        parent_pool: list[MemoryParent],
        existing_normalized: set[str] | None = None,
        case_snapshot: CaseMemorySnapshot | None = None,
    ) -> list[AlphaCandidate]:
        existing = set(existing_normalized or set())
        candidates: list[AlphaCandidate] = []
        if not parent_pool or count <= 0:
            self.last_generation_stats = GenerationSessionStats()
            self.base_engine.last_generation_stats = self.last_generation_stats
            return []

        validation_ctx, prepare_ms, cache_hit = self.base_engine._get_or_build_validation_context()  # noqa: SLF001
        session = GenerationSessionStats(
            prepare_validation_context_ms=prepare_ms,
            validation_context_cache_hit=cache_hit,
        )
        diversity_tracker = GenerationDiversityTracker()
        generation_started = time.monotonic()
        max_attempts = self.base_engine._resolve_max_attempts(count, legacy_multiplier=20, minimum=40)  # noqa: SLF001
        max_generation_seconds = float(getattr(self.adaptive_config, "max_generation_seconds", 0.0) or 0.0)
        consecutive_failures = 0
        while len(candidates) < count and session.attempt_count < max_attempts:
            if self._phase_should_stop(
                session=session,
                phase="exploit",
                generation_started=generation_started,
                phase_started=generation_started,
                phase_attempt_count=session.exploit_attempt_count,
                phase_attempt_budget=max_attempts,
                phase_time_budget=max_generation_seconds,
                consecutive_failures=consecutive_failures,
                candidate_count=len(candidates),
                target_count=count,
            ):
                break
            session.record_attempt()
            parent = self.mutation_policy._select_parent(parent_pool, diversity_tracker=diversity_tracker)  # noqa: SLF001
            variants = self.mutation_policy.generate(
                parent=parent,
                snapshot=snapshot,
                target_count=1,
                force_novelty=False,
                case_snapshot=case_snapshot,
                diversity_tracker=diversity_tracker,
            )
            if not variants:
                session.record_failure("mutation_payload_empty")
                consecutive_failures += 1
                continue
            expression, metadata = variants[0]
            metadata.setdefault(
                "memory_context",
                self.base_engine._memory_context_metadata(  # noqa: SLF001
                    pattern_snapshot=snapshot,
                    case_snapshot=case_snapshot,
                ),
            )
            parent_ids = tuple(ref["alpha_id"] for ref in metadata.get("parent_refs", []) if ref.get("alpha_id"))
            if self.base_engine._reject_pre_dedup_candidate(  # noqa: SLF001
                session=session,
                diversity_tracker=diversity_tracker,
                existing_normalized=existing,
                expression=expression,
                generation_metadata=metadata,
                operator_path=tuple(metadata.get("operator_path") or ()),
                genome_hash=str(metadata.get("genome_hash") or ""),
                structural_key=str(metadata.get("pre_normalized_expression") or ""),
                normalized_expression=str(metadata.get("pre_normalized_expression") or ""),
                pre_dedup=True,
            ):
                consecutive_failures += 1
                continue
            result = self.base_engine._build_candidate_result(  # noqa: SLF001
                expression=expression,
                mode=str(metadata.get("mutation_mode") or "exploit_local"),
                parent_ids=parent_ids,
                generation_metadata=metadata,
                validation_ctx=validation_ctx,
            )
            candidate = result.candidate
            if candidate is None:
                session.record_failure(
                    result.failure_reason,
                    expression=expression,
                    fields=result.failure_fields,
                )
                consecutive_failures += 1
                continue
            if candidate.normalized_expression in existing:
                self.base_engine._record_duplicate_reject(  # noqa: SLF001
                    session=session,
                    diversity_tracker=diversity_tracker,
                    reason="duplicate_normalized_expression",
                    expression=candidate.normalized_expression,
                    generation_metadata=candidate.generation_metadata,
                    operator_path=tuple(candidate.generation_metadata.get("operator_path") or candidate.operators_used),
                    pre_dedup=False,
                )
                consecutive_failures += 1
                continue
            existing.add(candidate.normalized_expression)
            candidates.append(candidate)
            session.record_success()
            self.base_engine._record_diversity_success(diversity_tracker, candidate)  # noqa: SLF001
            consecutive_failures = 0

        session.generation_total_ms = (time.monotonic() - generation_started) * 1000.0
        session.exploit_phase_ms = session.generation_total_ms
        self.last_generation_stats = session
        self.base_engine.last_generation_stats = session
        return candidates[:count]

    def _allocate_counts(self, count: int) -> dict[str, int]:
        mix = self.adaptive_config.strategy_mix
        fractions = {
            "guided_mutation": count * mix.guided_mutation,
            "memory_templates": count * mix.memory_templates,
            "random_exploration": count * mix.random_exploration,
            "novelty_search": count * mix.novelty_behavior,
        }
        quotas = {name: int(value) for name, value in fractions.items()}
        remainder = count - sum(quotas.values())
        if remainder > 0:
            for name, _ in sorted(
                fractions.items(),
                key=lambda item: (item[1] - int(item[1]), item[0]),
                reverse=True,
            ):
                if remainder <= 0:
                    break
                quotas[name] += 1
                remainder -= 1
        return quotas

    def _resolve_phase_attempt_budgets(self, total_attempt_budget: int) -> tuple[int, int]:
        if total_attempt_budget <= 0:
            return 0, 0
        _, explore_ratio = self._normalized_phase_budget_ratios()
        min_explore_attempts = max(0, int(getattr(self.adaptive_config, "min_explore_attempts", 50) or 0))
        reserved_explore_attempts = max(
            int(math.ceil(total_attempt_budget * explore_ratio)),
            min_explore_attempts,
        )
        reserved_explore_attempts = min(total_attempt_budget, reserved_explore_attempts)
        exploit_attempt_budget = max(0, total_attempt_budget - reserved_explore_attempts)
        return exploit_attempt_budget, reserved_explore_attempts

    def _resolve_exploit_time_budget(self, max_generation_seconds: float) -> float:
        if max_generation_seconds <= 0.0:
            return 0.0
        _, explore_ratio = self._normalized_phase_budget_ratios()
        min_explore_seconds = max(0.0, float(getattr(self.adaptive_config, "min_explore_seconds", 2.0) or 0.0))
        reserved_explore_seconds = max(max_generation_seconds * explore_ratio, min_explore_seconds)
        reserved_explore_seconds = min(max_generation_seconds, reserved_explore_seconds)
        return max(0.0, max_generation_seconds - reserved_explore_seconds)

    def _normalized_phase_budget_ratios(self) -> tuple[float, float]:
        exploit_ratio = max(0.0, float(getattr(self.adaptive_config, "exploit_budget_ratio", 0.60) or 0.0))
        explore_ratio = max(0.0, float(getattr(self.adaptive_config, "explore_budget_ratio", 0.40) or 0.0))
        total = exploit_ratio + explore_ratio
        if total <= 0.0:
            return 0.60, 0.40
        return exploit_ratio / total, explore_ratio / total

    def _remaining_generation_seconds(
        self,
        *,
        max_generation_seconds: float,
        generation_started: float,
    ) -> float:
        if max_generation_seconds <= 0.0:
            return 0.0
        return max(0.0, max_generation_seconds - (time.monotonic() - generation_started))

    def _phase_should_stop(
        self,
        *,
        session: GenerationSessionStats,
        phase: str,
        generation_started: float,
        phase_started: float,
        phase_attempt_count: int,
        phase_attempt_budget: int,
        phase_time_budget: float,
        consecutive_failures: int,
        candidate_count: int,
        target_count: int,
    ) -> bool:
        max_generation_seconds = float(getattr(self.adaptive_config, "max_generation_seconds", 0.0) or 0.0)
        now = time.monotonic()
        if (
            max_generation_seconds > 0.0
            and (now - generation_started) >= max_generation_seconds
            and not session.timeout_stop
        ):
            session.timeout_stop = True
            session.record_failure("budget_timeout")
            return True
        if phase_time_budget > 0.0 and (now - phase_started) >= phase_time_budget:
            return True
        if phase_attempt_budget > 0 and phase_attempt_count >= phase_attempt_budget:
            return True

        early_exit_limit = self.base_engine._resolve_consecutive_failure_limit(  # noqa: SLF001
            phase=phase,
            candidate_count=candidate_count,
            target_count=target_count,
        )
        if early_exit_limit <= 0 or consecutive_failures < early_exit_limit:
            return False

        if phase == "explore":
            if session.explore_consecutive_failure_stop:
                return True
        else:
            if session.exploit_consecutive_failure_stop:
                return True
        self.base_engine._mark_phase_failure_stop(session, phase)  # noqa: SLF001
        session.record_failure("consecutive_failure_break")
        return True
