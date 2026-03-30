from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from alpha.ast_nodes import to_expression
from alpha.parser import parse_expression
from alpha.validator import validate_expression
from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import OperatorRegistry
from generator.genome import GenomeRenderResult
from generator.genome_builder import GenomeBuilder
from generator.grammar import MotifGrammar
from generator.mutation_policy import MutationPolicy
from generator.repair_policy import RepairPolicy
from memory.case_memory import CaseMemorySnapshot
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot, RegionLearningContext


@dataclass(frozen=True, slots=True)
class AlphaCandidate:
    alpha_id: str
    expression: str
    normalized_expression: str
    generation_mode: str
    parent_ids: tuple[str, ...]
    complexity: int
    created_at: str
    template_name: str = ""
    fields_used: tuple[str, ...] = ()
    operators_used: tuple[str, ...] = ()
    depth: int = 0
    generation_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationValidationContext:
    allowed_generation_fields: set[str]
    group_fields: set[str]
    field_types: dict[str, str]
    field_categories: dict[str, str]


@dataclass(frozen=True, slots=True)
class CandidateBuildResult:
    candidate: AlphaCandidate | None
    failure_reason: str | None = None


@dataclass(slots=True)
class GenerationSessionStats:
    prepare_validation_context_ms: float = 0.0
    generation_total_ms: float = 0.0
    exploit_phase_ms: float = 0.0
    explore_phase_ms: float = 0.0
    attempt_count: int = 0
    attempt_success_count: int = 0
    timeout_stop: bool = False
    consecutive_failure_stop: bool = False
    validation_context_cache_hit: bool = False
    failure_counts: Counter[str] = field(default_factory=Counter)

    def record_failure(self, reason: str | None) -> None:
        if reason:
            self.failure_counts[reason] += 1

    def record_success(self) -> None:
        self.attempt_success_count += 1

    def top_fail_reasons(self, limit: int = 5) -> dict[str, int]:
        return dict(self.failure_counts.most_common(limit))

    def to_metrics(
        self,
        *,
        generated_count: int,
        selected_for_simulation: int = 0,
    ) -> dict[str, Any]:
        validate_reason_codes = (
            "disallowed_field",
            "invalid_group_field",
            "field_type_resolution_failed",
            "expression_validation_failed",
            "empty_render",
        )
        return {
            "generated": generated_count,
            "selected_for_simulation": selected_for_simulation,
            "prepare_validation_context_ms": round(self.prepare_validation_context_ms, 3),
            "generation_total_ms": round(self.generation_total_ms, 3),
            "exploit_phase_ms": round(self.exploit_phase_ms, 3),
            "explore_phase_ms": round(self.explore_phase_ms, 3),
            "attempt_count": self.attempt_count,
            "attempt_success_count": self.attempt_success_count,
            "parse_fail_count": int(self.failure_counts.get("parse_failed", 0)),
            "validate_fail_count": int(sum(self.failure_counts.get(code, 0) for code in validate_reason_codes)),
            "complexity_fail_count": int(self.failure_counts.get("complexity_exceeded", 0)),
            "redundancy_fail_count": int(self.failure_counts.get("redundant_expression", 0)),
            "duplicate_fail_count": int(self.failure_counts.get("duplicate_normalized_expression", 0)),
            "timeout_stop": self.timeout_stop,
            "consecutive_failure_stop": self.consecutive_failure_stop,
            "validation_context_cache_hit": self.validation_context_cache_hit,
            "failure_reason_counts": dict(self.failure_counts),
            "top_fail_reasons": self.top_fail_reasons(),
        }


class AlphaGenerationEngine:
    def __init__(
        self,
        config: GenerationConfig,
        registry: OperatorRegistry,
        field_registry: FieldRegistry | None = None,
        adaptive_config: AdaptiveGenerationConfig | None = None,
        region_learning_context: RegionLearningContext | None = None,
        mutation_learning_records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.adaptive_config = adaptive_config or AdaptiveGenerationConfig()
        self.field_registry = field_registry or self._fallback_field_registry(config.allowed_fields)
        self.region_learning_context = region_learning_context
        self.memory_service = PatternMemoryService()
        self.grammar = MotifGrammar()
        self.genome_builder = GenomeBuilder(
            generation_config=config,
            adaptive_config=self.adaptive_config,
            registry=registry,
            field_registry=self.field_registry,
            seed=config.random_seed,
        )
        self.repair_policy = RepairPolicy(
            generation_config=config,
            repair_config=self.adaptive_config.repair_policy,
            field_registry=self.field_registry,
            registry=self.registry,
        )
        self.mutation_policy = MutationPolicy(
            config=config,
            adaptive_config=self.adaptive_config,
            memory_service=self.memory_service,
            mutation_learning_records=mutation_learning_records,
            randomizer_seed=config.random_seed + 17,
            field_registry=self.field_registry,
            registry=self.registry,
        )
        self._validation_context: GenerationValidationContext | None = None
        self._validation_context_key: tuple[Any, ...] | None = None
        self.last_generation_stats: GenerationSessionStats | None = None

    def generate(
        self,
        count: int | None = None,
        existing_normalized: set[str] | None = None,
        case_snapshot: CaseMemorySnapshot | None = None,
    ) -> list[AlphaCandidate]:
        target = count or max(self.config.template_count + self.config.grammar_count, self.config.template_pool_size)
        existing = set(existing_normalized or set())
        candidates: list[AlphaCandidate] = []
        validation_ctx, prepare_ms, cache_hit = self._get_or_build_validation_context()
        session = GenerationSessionStats(
            prepare_validation_context_ms=prepare_ms,
            validation_context_cache_hit=cache_hit,
        )
        generation_started = time.monotonic()
        max_attempts = self._resolve_max_attempts(target, legacy_multiplier=25, minimum=80)
        consecutive_failures = 0
        while len(candidates) < target and session.attempt_count < max_attempts:
            if self._should_stop_generation(
                session=session,
                generation_started=generation_started,
                consecutive_failures=consecutive_failures,
                candidate_count=len(candidates),
                target_count=target,
            ):
                break
            session.attempt_count += 1
            novelty_bias = (len(candidates) / max(1, target)) >= (1.0 - self.adaptive_config.exploration_ratio)
            genome = self.genome_builder.build_random_genome(
                source_mode="genome_novelty" if novelty_bias else "genome_random",
                novelty_bias=novelty_bias,
                case_snapshot=case_snapshot,
            )
            repaired_genome, repair_actions = self.repair_policy.repair(genome)
            render = self.grammar.render(repaired_genome)
            result = self._build_candidate_result(
                expression=render.expression,
                mode="novelty" if novelty_bias else "genome",
                parent_ids=(),
                generation_metadata=self._render_metadata(
                    render,
                    mutation_mode="novelty" if novelty_bias else "exploit_local",
                    repair_actions=repair_actions,
                    memory_context=self._memory_context_metadata(case_snapshot=case_snapshot),
                ),
                validation_ctx=validation_ctx,
            )
            candidate = result.candidate
            if candidate is None:
                session.record_failure(result.failure_reason)
                consecutive_failures += 1
                continue
            if candidate.normalized_expression in existing:
                session.record_failure("duplicate_normalized_expression")
                consecutive_failures += 1
                continue
            existing.add(candidate.normalized_expression)
            candidates.append(candidate)
            session.record_success()
            consecutive_failures = 0
        session.generation_total_ms = (time.monotonic() - generation_started) * 1000.0
        session.exploit_phase_ms = session.generation_total_ms
        self.last_generation_stats = session
        return candidates

    def generate_mutations(
        self,
        parents: Sequence[AlphaCandidate],
        count: int | None = None,
        existing_normalized: set[str] | None = None,
        case_snapshot: CaseMemorySnapshot | None = None,
    ) -> list[AlphaCandidate]:
        target = count or self.config.mutation_count
        existing = set(existing_normalized or set())
        candidates: list[AlphaCandidate] = []
        validation_ctx, prepare_ms, cache_hit = self._get_or_build_validation_context()
        session = GenerationSessionStats(
            prepare_validation_context_ms=prepare_ms,
            validation_context_cache_hit=cache_hit,
        )
        generation_started = time.monotonic()
        max_attempts = self._resolve_max_attempts(target, legacy_multiplier=12, minimum=24)
        consecutive_failures = 0
        while len(candidates) < target and session.attempt_count < max_attempts:
            if self._should_stop_generation(
                session=session,
                generation_started=generation_started,
                consecutive_failures=consecutive_failures,
                candidate_count=len(candidates),
                target_count=target,
            ):
                break
            session.attempt_count += 1
            payload = self.mutation_policy.generate_from_candidates(
                parents=list(parents),
                target_count=1,
                case_snapshot=case_snapshot,
            )
            if not payload:
                session.record_failure("expression_validation_failed")
                consecutive_failures += 1
                continue
            expression, mode, parent_ids, metadata = payload[0]
            metadata.setdefault("memory_context", self._memory_context_metadata(case_snapshot=case_snapshot))
            result = self._build_candidate_result(
                expression=expression,
                mode=mode,
                parent_ids=parent_ids,
                generation_metadata=metadata,
                validation_ctx=validation_ctx,
            )
            candidate = result.candidate
            if candidate is None:
                session.record_failure(result.failure_reason)
                consecutive_failures += 1
                continue
            if candidate.normalized_expression in existing:
                session.record_failure("duplicate_normalized_expression")
                consecutive_failures += 1
                continue
            existing.add(candidate.normalized_expression)
            candidates.append(candidate)
            session.record_success()
            consecutive_failures = 0
        session.generation_total_ms = (time.monotonic() - generation_started) * 1000.0
        session.exploit_phase_ms = session.generation_total_ms
        self.last_generation_stats = session
        return candidates

    def _build_candidates(
        self,
        payload: Iterable[tuple[str, str, tuple[str, ...], dict[str, Any]]],
        existing_normalized: set[str] | None = None,
        session: GenerationSessionStats | None = None,
    ) -> list[AlphaCandidate]:
        existing = set(existing_normalized or set())
        candidates: list[AlphaCandidate] = []
        validation_ctx, prepare_ms, cache_hit = self._get_or_build_validation_context()
        if session is not None and session.prepare_validation_context_ms == 0.0:
            session.prepare_validation_context_ms = prepare_ms
            session.validation_context_cache_hit = cache_hit
        for expression, mode, parent_ids, generation_metadata in payload:
            if session is not None:
                session.attempt_count += 1
            result = self._build_candidate_result(
                expression=expression,
                mode=mode,
                parent_ids=parent_ids,
                generation_metadata=generation_metadata,
                validation_ctx=validation_ctx,
            )
            candidate = result.candidate
            if candidate is None:
                if session is not None:
                    session.record_failure(result.failure_reason)
                continue
            if candidate.normalized_expression in existing:
                if session is not None:
                    session.record_failure("duplicate_normalized_expression")
                continue
            existing.add(candidate.normalized_expression)
            candidates.append(candidate)
            if session is not None:
                session.record_success()
        return candidates

    def build_candidate(
        self,
        expression: str,
        mode: str,
        parent_ids: tuple[str, ...],
        generation_metadata: dict[str, Any] | None = None,
    ) -> AlphaCandidate | None:
        return self._build_candidate_result(
            expression=expression,
            mode=mode,
            parent_ids=parent_ids,
            generation_metadata=generation_metadata,
        ).candidate

    def prepare_validation_context(self) -> GenerationValidationContext:
        validation_ctx, _, _ = self._get_or_build_validation_context()
        return validation_ctx

    def invalidate_validation_context_cache(self) -> None:
        self._validation_context = None
        self._validation_context_key = None

    def _build_candidate_result(
        self,
        expression: str,
        mode: str,
        parent_ids: tuple[str, ...],
        generation_metadata: dict[str, Any] | None = None,
        validation_ctx: GenerationValidationContext | None = None,
    ) -> CandidateBuildResult:
        metadata = dict(generation_metadata or {})
        if self.region_learning_context is not None:
            metadata.setdefault("region", self.region_learning_context.region)
            metadata.setdefault("regime_key", self.region_learning_context.regime_key)
            metadata.setdefault("global_regime_key", self.region_learning_context.global_regime_key)
            memory_context = metadata.get("memory_context")
            if not isinstance(memory_context, dict):
                metadata["memory_context"] = self._memory_context_metadata()
        if not expression or not expression.strip():
            return CandidateBuildResult(candidate=None, failure_reason="empty_render")
        try:
            node = parse_expression(expression)
        except ValueError:
            return CandidateBuildResult(candidate=None, failure_reason="parse_failed")

        prepared_validation_ctx = validation_ctx or self.prepare_validation_context()
        validation = validate_expression(
            node=node,
            registry=self.registry,
            allowed_fields=prepared_validation_ctx.allowed_generation_fields,
            max_depth=self.config.max_depth,
            group_fields=prepared_validation_ctx.group_fields,
            field_types=prepared_validation_ctx.field_types,
            complexity_limit=self.config.complexity_limit,
        )
        if not validation.is_valid:
            return CandidateBuildResult(
                candidate=None,
                failure_reason=self._classify_validation_failure(validation.errors),
            )

        normalized_expression = to_expression(node)
        if not normalized_expression.strip():
            return CandidateBuildResult(candidate=None, failure_reason="empty_render")
        signature = self.memory_service.extract_signature(
            normalized_expression,
            generation_metadata=metadata,
            field_categories=prepared_validation_ctx.field_categories,
        )
        complexity = signature.complexity
        if complexity > self.config.complexity_limit:
            return CandidateBuildResult(candidate=None, failure_reason="complexity_exceeded")
        if self._is_redundant(signature.fields):
            return CandidateBuildResult(candidate=None, failure_reason="redundant_expression")

        alpha_id = hashlib.sha1(normalized_expression.encode("utf-8")).hexdigest()[:16]
        parent_refs = metadata.get("parent_refs") if isinstance(metadata.get("parent_refs"), list) else []
        primary_parent = parent_refs[0] if parent_refs else {}
        metadata["family_signature"] = signature.family_signature
        metadata["canonical_structural_signature"] = signature.to_dict()
        metadata["fields_used"] = list(metadata.get("fields_used") or signature.fields)
        metadata["field_families"] = list(metadata.get("field_families") or signature.field_families)
        metadata["operators_used"] = list(metadata.get("operators_used") or signature.operators)
        metadata["operator_path"] = list(metadata.get("operator_path") or signature.operator_path)
        metadata["horizon_bucket"] = str(metadata.get("horizon_bucket") or signature.horizon_bucket)
        metadata["turnover_bucket"] = str(metadata.get("turnover_bucket") or signature.turnover_bucket)
        metadata["complexity_bucket"] = str(metadata.get("complexity_bucket") or signature.complexity_bucket)
        metadata["primary_parent_alpha_id"] = str(primary_parent.get("alpha_id") or "")
        metadata["primary_parent_family_signature"] = str(primary_parent.get("family_signature") or "")
        metadata["lineage_branch_key"] = (
            str(primary_parent.get("alpha_id") or primary_parent.get("family_signature") or "")
        )
        fields_used = tuple(metadata.get("fields_used") or signature.fields)
        operators_used = tuple(metadata.get("operators_used") or signature.operators)
        template_name = str(metadata.get("template_name") or metadata.get("motif") or "")
        return CandidateBuildResult(
            candidate=AlphaCandidate(
                alpha_id=alpha_id,
                expression=expression.strip(),
                normalized_expression=normalized_expression,
                generation_mode=mode,
                parent_ids=parent_ids,
                complexity=complexity,
                created_at=datetime.now(timezone.utc).isoformat(),
                template_name=template_name,
                fields_used=fields_used,
                operators_used=operators_used,
                depth=signature.depth,
                generation_metadata=metadata,
            )
        )

    def _is_redundant(self, fields: tuple[str, ...]) -> bool:
        return len(fields) == 0

    def _render_metadata(
        self,
        render: GenomeRenderResult,
        *,
        mutation_mode: str,
        repair_actions: tuple[str, ...] = (),
        parent_refs: list[dict[str, str]] | None = None,
        selection_objectives: dict[str, float] | None = None,
        memory_context: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "template_name": render.genome.motif,
            "motif": render.genome.motif,
            "genome": render.genome.to_dict(),
            "genome_hash": render.genome.stable_hash,
            "family_signature": render.family_signature,
            "fields_used": list(render.field_names),
            "field_families": list(render.field_families),
            "operators_used": list(dict.fromkeys(render.operator_path)),
            "operator_path": list(render.operator_path),
            "operator_semantic_tags": self._operator_semantic_tags(render.operator_path),
            "turnover_bucket": render.turnover_bucket,
            "horizon_bucket": render.horizon_bucket,
            "complexity_bucket": render.complexity_bucket,
            "mutation_mode": mutation_mode,
            "repair_actions": list(repair_actions),
            "parent_refs": list(parent_refs or []),
            "selection_objectives": dict(selection_objectives or {}),
        }
        if self.region_learning_context is not None:
            payload.update(self.region_learning_context.to_dict())
        if memory_context:
            payload["memory_context"] = dict(memory_context)
        return payload

    def _memory_context_metadata(
        self,
        *,
        pattern_snapshot: PatternMemorySnapshot | None = None,
        case_snapshot: CaseMemorySnapshot | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.region_learning_context is not None:
            payload.update(self.region_learning_context.to_dict())
        if pattern_snapshot is not None and pattern_snapshot.blend is not None:
            payload["pattern_blend"] = pattern_snapshot.blend.to_dict()
        if case_snapshot is not None and case_snapshot.blend is not None:
            payload["case_blend"] = case_snapshot.blend.to_dict()
        return payload

    def _operator_semantic_tags(self, operator_path: tuple[str, ...]) -> list[str]:
        tags: set[str] = set()
        for name in operator_path:
            if not self.registry.contains(name):
                continue
            tags.update(self.registry.get(name).semantic_tags)
        return sorted(tags)

    def _get_or_build_validation_context(self) -> tuple[GenerationValidationContext, float, bool]:
        if not getattr(self.config, "engine_validation_cache_enabled", True):
            self.invalidate_validation_context_cache()
            started = time.perf_counter()
            validation_ctx = self._prepare_validation_context()
            return validation_ctx, (time.perf_counter() - started) * 1000.0, False

        cache_key = (
            id(self.field_registry),
            tuple(self.config.allowed_fields),
            bool(self.config.allow_catalog_fields_without_runtime),
        )
        if self._validation_context is not None and self._validation_context_key == cache_key:
            return self._validation_context, 0.0, True

        started = time.perf_counter()
        validation_ctx = self._prepare_validation_context()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._validation_context = validation_ctx
        self._validation_context_key = cache_key
        return validation_ctx, elapsed_ms, False

    def _prepare_validation_context(self) -> GenerationValidationContext:
        numeric_fields = {
            spec.name
            for spec in self.field_registry.generation_numeric_fields(
                self.config.allowed_fields,
                include_catalog_fields=self.config.allow_catalog_fields_without_runtime,
            )
        }
        group_fields = {
            spec.name
            for spec in self.field_registry.generation_group_fields(
                include_catalog_fields=self.config.allow_catalog_fields_without_runtime,
            )
        }
        allowed_generation_fields = set(numeric_fields)
        allowed_generation_fields.update(group_fields)
        validation_ctx = GenerationValidationContext(
            allowed_generation_fields=allowed_generation_fields,
            group_fields=group_fields,
            field_types=self.field_registry.field_types(allowed=allowed_generation_fields),
            field_categories={name: spec.category for name, spec in self.field_registry.fields.items()},
        )
        return validation_ctx

    def _resolve_max_attempts(self, target: int, *, legacy_multiplier: int, minimum: int) -> int:
        multiplier = int(getattr(self.adaptive_config, "max_attempt_multiplier", legacy_multiplier))
        return max(target * max(1, multiplier), minimum)

    def _should_stop_generation(
        self,
        *,
        session: GenerationSessionStats,
        generation_started: float,
        consecutive_failures: int,
        candidate_count: int,
        target_count: int,
    ) -> bool:
        max_generation_seconds = float(getattr(self.adaptive_config, "max_generation_seconds", 0.0) or 0.0)
        if (
            max_generation_seconds > 0
            and (time.monotonic() - generation_started) >= max_generation_seconds
            and not session.timeout_stop
        ):
            session.timeout_stop = True
            session.record_failure("budget_timeout")
            return True

        failure_limit = int(getattr(self.adaptive_config, "max_consecutive_failures", 0) or 0)
        partial_success_floor = min(
            max(1, int(getattr(self.adaptive_config, "min_candidates_before_early_exit", 1) or 1)),
            max(1, target_count),
        )
        early_exit_limit = max(1, failure_limit // 2) if candidate_count >= partial_success_floor else failure_limit
        if early_exit_limit > 0 and consecutive_failures >= early_exit_limit and not session.consecutive_failure_stop:
            session.consecutive_failure_stop = True
            session.record_failure("consecutive_failure_break")
            return True
        return False

    def _classify_validation_failure(self, errors: Sequence[str]) -> str:
        lowered = [error.lower() for error in errors]
        if any("complexity exceeds limit" in error for error in lowered):
            return "complexity_exceeded"
        if any("redundant" in error for error in lowered):
            return "redundant_expression"
        if any("group field can only" in error for error in lowered):
            return "invalid_group_field"
        if any("unknown field" in error for error in lowered):
            return "disallowed_field"
        if any("input types" in error for error in lowered):
            return "field_type_resolution_failed"
        return "expression_validation_failed"

    def _fallback_field_registry(self, allowed_fields: list[str]) -> FieldRegistry:
        fields = {
            name: FieldSpec(
                name=name,
                dataset="fallback",
                field_type="vector" if name in {"sector", "industry", "country", "subindustry"} else "matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="group" if name in {"sector", "industry", "country", "subindustry"} else "other",
                runtime_available=True,
                field_score=1.0,
                category_weight=0.5,
            )
            for name in allowed_fields
        }
        return FieldRegistry(fields=fields)
