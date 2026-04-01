from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from alpha.ast_nodes import to_expression
from alpha.parser import parse_expression
from alpha.validator import ValidationResult, validate_expression
from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import OperatorRegistry
from generator.diversity_tracker import GenerationDiversityTracker, operator_path_key
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
    exploit_attempt_count: int = 0
    explore_attempt_count: int = 0
    exploit_success_count: int = 0
    explore_success_count: int = 0
    timeout_stop: bool = False
    consecutive_failure_stop: bool = False
    exploit_consecutive_failure_stop: bool = False
    explore_consecutive_failure_stop: bool = False
    explore_phase_entered: bool = False
    validation_context_cache_hit: bool = False
    pre_dedup_reject_count: int = 0
    failure_counts: Counter[str] = field(default_factory=Counter)
    failure_samples: dict[str, list[str]] = field(default_factory=dict)
    duplicate_by_mutation_mode: Counter[str] = field(default_factory=Counter)
    duplicate_by_motif: Counter[str] = field(default_factory=Counter)
    duplicate_by_operator_path: Counter[str] = field(default_factory=Counter)

    def record_attempt(self, phase: str = "exploit") -> None:
        self.attempt_count += 1
        if phase == "explore":
            self.explore_attempt_count += 1
            return
        self.exploit_attempt_count += 1

    def record_failure(self, reason: str | None, *, expression: str | None = None) -> None:
        normalized_reason = self._normalize_failure_reason(reason)
        if normalized_reason is None:
            return
        self.failure_counts[normalized_reason] += 1
        if expression:
            bucket = self.failure_samples.setdefault(normalized_reason, [])
            if expression not in bucket and len(bucket) < 3:
                bucket.append(expression)

    def record_success(self, phase: str = "exploit") -> None:
        self.attempt_success_count += 1
        if phase == "explore":
            self.explore_success_count += 1
            return
        self.exploit_success_count += 1

    def record_duplicate(
        self,
        reason: str,
        *,
        expression: str | None = None,
        mutation_mode: str = "",
        motif: str = "",
        operator_path: tuple[str, ...] = (),
        pre_dedup: bool = False,
    ) -> None:
        self.record_failure(reason, expression=expression)
        if pre_dedup:
            self.pre_dedup_reject_count += 1
        if mutation_mode:
            self.duplicate_by_mutation_mode[mutation_mode] += 1
        if motif:
            self.duplicate_by_motif[motif] += 1
        self.duplicate_by_operator_path[operator_path_key(operator_path)] += 1

    def top_fail_reasons(self, limit: int = 5) -> dict[str, int]:
        return dict(self._materialize_failure_counts().most_common(limit))

    def to_metrics(
        self,
        *,
        generated_count: int,
        selected_for_simulation: int = 0,
        include_debug_samples: bool = False,
    ) -> dict[str, Any]:
        failure_reason_counts = self._materialize_failure_counts()
        metrics = {
            "generated": generated_count,
            "selected_for_simulation": selected_for_simulation,
            "prepare_validation_context_ms": round(self.prepare_validation_context_ms, 3),
            "generation_total_ms": round(self.generation_total_ms, 3),
            "exploit_phase_ms": round(self.exploit_phase_ms, 3),
            "explore_phase_ms": round(self.explore_phase_ms, 3),
            "attempt_count": self.attempt_count,
            "attempt_success_count": self.attempt_success_count,
            "exploit_attempt_count": self.exploit_attempt_count,
            "explore_attempt_count": self.explore_attempt_count,
            "exploit_success_count": self.exploit_success_count,
            "explore_success_count": self.explore_success_count,
            "pre_dedup_reject_count": int(self.pre_dedup_reject_count),
            "parse_fail_count": int(failure_reason_counts.get("parse_failed", 0)),
            "validate_fail_count": int(
                sum(count for reason, count in failure_reason_counts.items() if reason.startswith("validation_"))
            ),
            "complexity_fail_count": int(failure_reason_counts.get("complexity_exceeded", 0)),
            "redundancy_fail_count": int(failure_reason_counts.get("redundant_expression", 0)),
            "duplicate_fail_count": int(
                failure_reason_counts.get("duplicate_normalized_expression", 0)
                + failure_reason_counts.get("structural_duplicate_expression", 0)
            ),
            "structural_duplicate_count": int(failure_reason_counts.get("structural_duplicate_expression", 0)),
            "normalized_duplicate_count": int(failure_reason_counts.get("duplicate_normalized_expression", 0)),
            "timeout_stop": self.timeout_stop,
            "consecutive_failure_stop": self.consecutive_failure_stop,
            "exploit_consecutive_failure_stop": self.exploit_consecutive_failure_stop,
            "explore_consecutive_failure_stop": self.explore_consecutive_failure_stop,
            "explore_phase_entered": self.explore_phase_entered,
            "validation_context_cache_hit": self.validation_context_cache_hit,
            "duplicate_by_mutation_mode": dict(self.duplicate_by_mutation_mode),
            "duplicate_by_motif": dict(self.duplicate_by_motif),
            "duplicate_by_operator_path": dict(self.duplicate_by_operator_path),
            "failure_reason_counts": dict(failure_reason_counts),
            "top_fail_reasons": self.top_fail_reasons(),
        }
        if include_debug_samples:
            metrics["failure_samples"] = {
                reason: list(samples)
                for reason, samples in self.failure_samples.items()
                if samples
            }
        return metrics

    @staticmethod
    def _normalize_failure_reason(reason: str | None) -> str | None:
        if reason == "expression_validation_failed":
            return "validation_unknown_error"
        return reason

    def _materialize_failure_counts(self) -> Counter[str]:
        counts = Counter(self.failure_counts)
        legacy = int(counts.pop("expression_validation_failed", 0))
        if legacy:
            counts["validation_unknown_error"] += legacy
        return counts


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
        diversity_tracker = GenerationDiversityTracker()
        generation_started = time.monotonic()
        max_attempts = self._resolve_max_attempts(target, legacy_multiplier=25, minimum=80)
        consecutive_failures = 0
        while len(candidates) < target and session.attempt_count < max_attempts:
            novelty_bias = (len(candidates) / max(1, target)) >= (1.0 - self.adaptive_config.exploration_ratio)
            phase = "explore" if novelty_bias else "exploit"
            if phase == "explore":
                session.explore_phase_entered = True
            if self._should_stop_generation(
                session=session,
                generation_started=generation_started,
                consecutive_failures=consecutive_failures,
                candidate_count=len(candidates),
                target_count=target,
                phase=phase,
            ):
                break
            session.record_attempt(phase)
            genome = self.genome_builder.build_random_genome(
                source_mode="genome_novelty" if novelty_bias else "genome_random",
                novelty_bias=novelty_bias,
                case_snapshot=case_snapshot,
                diversity_tracker=diversity_tracker,
            )
            repaired_genome, repair_actions = self.repair_policy.repair(genome)
            render = self.grammar.render(repaired_genome)
            if not self._is_parseable_expression(render.expression):
                session.record_failure("parse_failed", expression=render.expression)
                consecutive_failures += 1
                continue
            metadata = self._render_metadata(
                render,
                mutation_mode="novelty" if novelty_bias else "exploit_local",
                repair_actions=repair_actions,
                memory_context=self._memory_context_metadata(case_snapshot=case_snapshot),
            )
            if self._reject_pre_dedup_candidate(
                session=session,
                diversity_tracker=diversity_tracker,
                existing_normalized=existing,
                expression=render.expression,
                generation_metadata=metadata,
                operator_path=render.operator_path,
                genome_hash=repaired_genome.stable_hash,
                structural_key=render.normalized_expression,
                normalized_expression=render.normalized_expression,
                pre_dedup=True,
            ):
                consecutive_failures += 1
                continue
            result = self._build_candidate_result(
                expression=render.expression,
                mode="novelty" if novelty_bias else "genome",
                parent_ids=(),
                generation_metadata=metadata,
                validation_ctx=validation_ctx,
            )
            candidate = result.candidate
            if candidate is None:
                session.record_failure(result.failure_reason, expression=render.expression)
                consecutive_failures += 1
                continue
            if candidate.normalized_expression in existing:
                self._record_duplicate_reject(
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
            session.record_success(phase)
            self._record_diversity_success(diversity_tracker, candidate)
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
        diversity_tracker = GenerationDiversityTracker()
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
                phase="exploit",
            ):
                break
            session.record_attempt("exploit")
            payload = self.mutation_policy.generate_from_candidates(
                parents=list(parents),
                target_count=1,
                case_snapshot=case_snapshot,
                diversity_tracker=diversity_tracker,
            )
            if not payload:
                session.record_failure("mutation_payload_empty")
                consecutive_failures += 1
                continue
            expression, mode, parent_ids, metadata = payload[0]
            metadata.setdefault("memory_context", self._memory_context_metadata(case_snapshot=case_snapshot))
            if self._reject_pre_dedup_candidate(
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
            result = self._build_candidate_result(
                expression=expression,
                mode=mode,
                parent_ids=parent_ids,
                generation_metadata=metadata,
                validation_ctx=validation_ctx,
            )
            candidate = result.candidate
            if candidate is None:
                session.record_failure(result.failure_reason, expression=expression)
                consecutive_failures += 1
                continue
            if candidate.normalized_expression in existing:
                self._record_duplicate_reject(
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
            session.record_success("exploit")
            self._record_diversity_success(diversity_tracker, candidate)
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
        diversity_tracker = GenerationDiversityTracker()
        for expression, mode, parent_ids, generation_metadata in payload:
            if session is not None:
                session.record_attempt()
            if self._reject_pre_dedup_candidate(
                session=session,
                diversity_tracker=diversity_tracker,
                existing_normalized=existing,
                expression=expression,
                generation_metadata=generation_metadata,
                operator_path=tuple((generation_metadata or {}).get("operator_path") or ()),
                genome_hash=str((generation_metadata or {}).get("genome_hash") or ""),
                structural_key=str((generation_metadata or {}).get("pre_normalized_expression") or ""),
                normalized_expression=str((generation_metadata or {}).get("pre_normalized_expression") or ""),
                pre_dedup=True,
            ):
                continue
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
                    session.record_failure(result.failure_reason, expression=expression)
                continue
            if candidate.normalized_expression in existing:
                if session is not None:
                    self._record_duplicate_reject(
                        session=session,
                        diversity_tracker=diversity_tracker,
                        reason="duplicate_normalized_expression",
                        expression=candidate.normalized_expression,
                        generation_metadata=candidate.generation_metadata,
                        operator_path=tuple(candidate.generation_metadata.get("operator_path") or candidate.operators_used),
                        pre_dedup=False,
                    )
                continue
            existing.add(candidate.normalized_expression)
            candidates.append(candidate)
            if session is not None:
                session.record_success()
            self._record_diversity_success(diversity_tracker, candidate)
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
                failure_reason=self._classify_validation_failure(validation),
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
            "pre_normalized_expression": render.normalized_expression,
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

    def _reject_pre_dedup_candidate(
        self,
        *,
        session: GenerationSessionStats | None,
        diversity_tracker: GenerationDiversityTracker,
        existing_normalized: set[str],
        expression: str,
        generation_metadata: dict[str, Any] | None,
        operator_path: tuple[str, ...],
        genome_hash: str,
        structural_key: str,
        normalized_expression: str,
        pre_dedup: bool,
    ) -> bool:
        reason = diversity_tracker.check_pre_dedup(
            genome_hash=genome_hash.strip(),
            structural_key=structural_key.strip(),
            normalized_expression=normalized_expression.strip(),
            existing_normalized=existing_normalized,
        )
        if reason is None:
            return False
        self._record_duplicate_reject(
            session=session,
            diversity_tracker=diversity_tracker,
            reason=reason,
            expression=normalized_expression.strip() or expression,
            generation_metadata=generation_metadata,
            operator_path=operator_path,
            pre_dedup=pre_dedup,
        )
        return True

    def _record_duplicate_reject(
        self,
        *,
        session: GenerationSessionStats | None,
        diversity_tracker: GenerationDiversityTracker,
        reason: str,
        expression: str,
        generation_metadata: dict[str, Any] | None,
        operator_path: tuple[str, ...],
        pre_dedup: bool,
    ) -> None:
        mutation_mode = self._metadata_string(generation_metadata, "mutation_mode")
        motif = self._metadata_string(generation_metadata, "motif", fallback_key="template_name")
        lineage_key = self._lineage_key(generation_metadata)
        diversity_tracker.record_duplicate(
            mutation_mode=mutation_mode,
            motif=motif,
            operator_path=operator_path,
            lineage_key=lineage_key,
        )
        if session is not None:
            session.record_duplicate(
                reason,
                expression=expression,
                mutation_mode=mutation_mode,
                motif=motif,
                operator_path=operator_path,
                pre_dedup=pre_dedup,
            )

    def _record_diversity_success(
        self,
        diversity_tracker: GenerationDiversityTracker,
        candidate: AlphaCandidate,
    ) -> None:
        metadata = candidate.generation_metadata
        diversity_tracker.record_candidate(
            motif=self._metadata_string(metadata, "motif", fallback_key="template_name"),
            operator_path=tuple(metadata.get("operator_path") or candidate.operators_used),
            field_families=tuple(metadata.get("field_families") or ()),
        )

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

    @staticmethod
    def _metadata_string(
        generation_metadata: dict[str, Any] | None,
        key: str,
        *,
        fallback_key: str | None = None,
    ) -> str:
        metadata = generation_metadata or {}
        value = metadata.get(key)
        if not value and fallback_key is not None:
            value = metadata.get(fallback_key)
        return str(value or "")

    def _lineage_key(self, generation_metadata: dict[str, Any] | None) -> str:
        metadata = generation_metadata or {}
        explicit = str(metadata.get("lineage_branch_key") or "")
        if explicit:
            return explicit
        parent_refs = metadata.get("parent_refs")
        if isinstance(parent_refs, list) and parent_refs:
            primary = parent_refs[0]
            if isinstance(primary, dict):
                return str(primary.get("alpha_id") or primary.get("family_signature") or "")
        return ""

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
        phase: str = "exploit",
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

        early_exit_limit = self._resolve_consecutive_failure_limit(
            phase=phase,
            candidate_count=candidate_count,
            target_count=target_count,
        )
        if early_exit_limit <= 0 or consecutive_failures < early_exit_limit:
            return False
        if phase == "explore":
            if session.explore_consecutive_failure_stop:
                return True
        elif session.exploit_consecutive_failure_stop:
            return True
        self._mark_phase_failure_stop(session, phase)
        session.record_failure("consecutive_failure_break")
        return True

    def _resolve_consecutive_failure_limit(
        self,
        *,
        phase: str,
        candidate_count: int,
        target_count: int,
    ) -> int:
        failure_limit = max(0, int(getattr(self.adaptive_config, "max_consecutive_failures", 0) or 0))
        phase_limit = failure_limit
        if phase == "explore":
            explore_limit = getattr(self.adaptive_config, "explore_max_consecutive_failures", None)
            phase_limit = max(0, int(explore_limit if explore_limit is not None else failure_limit * 2))
        partial_success_floor = min(
            max(1, int(getattr(self.adaptive_config, "min_candidates_before_early_exit", 1) or 1)),
            max(1, target_count),
        )
        if phase_limit <= 0:
            return 0
        return max(1, phase_limit // 2) if candidate_count >= partial_success_floor else phase_limit

    @staticmethod
    def _mark_phase_failure_stop(session: GenerationSessionStats, phase: str) -> None:
        session.consecutive_failure_stop = True
        if phase == "explore":
            session.explore_consecutive_failure_stop = True
            return
        session.exploit_consecutive_failure_stop = True

    @staticmethod
    def _is_parseable_expression(expression: str) -> bool:
        try:
            parse_expression(expression)
        except ValueError:
            return False
        return True

    def _classify_validation_failure(self, validation: ValidationResult) -> str:
        if validation.primary_reason_code:
            return validation.primary_reason_code
        return "validation_unknown_error"

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
