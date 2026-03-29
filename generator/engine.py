from __future__ import annotations

import hashlib
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


class AlphaGenerationEngine:
    def __init__(
        self,
        config: GenerationConfig,
        registry: OperatorRegistry,
        field_registry: FieldRegistry | None = None,
        adaptive_config: AdaptiveGenerationConfig | None = None,
        region_learning_context: RegionLearningContext | None = None,
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
            randomizer_seed=config.random_seed + 17,
            field_registry=self.field_registry,
            registry=self.registry,
        )

    def generate(
        self,
        count: int | None = None,
        existing_normalized: set[str] | None = None,
        case_snapshot: CaseMemorySnapshot | None = None,
    ) -> list[AlphaCandidate]:
        target = count or max(self.config.template_count + self.config.grammar_count, self.config.template_pool_size)
        existing = set(existing_normalized or set())
        candidates: list[AlphaCandidate] = []
        attempts = 0
        max_attempts = max(target * 25, 80)
        while len(candidates) < target and attempts < max_attempts:
            attempts += 1
            novelty_bias = (len(candidates) / max(1, target)) >= (1.0 - self.adaptive_config.exploration_ratio)
            genome = self.genome_builder.build_random_genome(
                source_mode="genome_novelty" if novelty_bias else "genome_random",
                novelty_bias=novelty_bias,
                case_snapshot=case_snapshot,
            )
            repaired_genome, repair_actions = self.repair_policy.repair(genome)
            render = self.grammar.render(repaired_genome)
            candidate = self.build_candidate(
                expression=render.expression,
                mode="novelty" if novelty_bias else "genome",
                parent_ids=(),
                generation_metadata=self._render_metadata(
                    render,
                    mutation_mode="novelty" if novelty_bias else "exploit_local",
                    repair_actions=repair_actions,
                    memory_context=self._memory_context_metadata(case_snapshot=case_snapshot),
                ),
            )
            if candidate is None or candidate.normalized_expression in existing:
                continue
            existing.add(candidate.normalized_expression)
            candidates.append(candidate)
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
        attempts = 0
        max_attempts = max(target * 12, 24)
        while len(candidates) < target and attempts < max_attempts:
            attempts += 1
            payload = self.mutation_policy.generate_from_candidates(
                parents=list(parents),
                target_count=1,
                case_snapshot=case_snapshot,
            )
            if not payload:
                continue
            expression, mode, parent_ids, metadata = payload[0]
            metadata.setdefault("memory_context", self._memory_context_metadata(case_snapshot=case_snapshot))
            candidate = self.build_candidate(
                expression=expression,
                mode=mode,
                parent_ids=parent_ids,
                generation_metadata=metadata,
            )
            if candidate is None or candidate.normalized_expression in existing:
                continue
            existing.add(candidate.normalized_expression)
            candidates.append(candidate)
        return candidates

    def _build_candidates(
        self,
        payload: Iterable[tuple[str, str, tuple[str, ...], dict[str, Any]]],
        existing_normalized: set[str] | None = None,
    ) -> list[AlphaCandidate]:
        existing = set(existing_normalized or set())
        candidates: list[AlphaCandidate] = []
        for expression, mode, parent_ids, generation_metadata in payload:
            candidate = self.build_candidate(
                expression=expression,
                mode=mode,
                parent_ids=parent_ids,
                generation_metadata=generation_metadata,
            )
            if candidate is None or candidate.normalized_expression in existing:
                continue
            existing.add(candidate.normalized_expression)
            candidates.append(candidate)
        return candidates

    def build_candidate(
        self,
        expression: str,
        mode: str,
        parent_ids: tuple[str, ...],
        generation_metadata: dict[str, Any] | None = None,
    ) -> AlphaCandidate | None:
        metadata = dict(generation_metadata or {})
        if self.region_learning_context is not None:
            metadata.setdefault("region", self.region_learning_context.region)
            metadata.setdefault("regime_key", self.region_learning_context.regime_key)
            metadata.setdefault("global_regime_key", self.region_learning_context.global_regime_key)
            memory_context = metadata.get("memory_context")
            if not isinstance(memory_context, dict):
                metadata["memory_context"] = self._memory_context_metadata()
        try:
            node = parse_expression(expression)
        except ValueError:
            return None

        allowed_runtime_fields = self.field_registry.allowed_runtime_fields(self.config.allowed_fields) | {
            spec.name for spec in self.field_registry.runtime_group_fields()
        }
        validation = validate_expression(
            node=node,
            registry=self.registry,
            allowed_fields=allowed_runtime_fields,
            max_depth=self.config.max_depth,
            group_fields={spec.name for spec in self.field_registry.runtime_group_fields()},
            field_types=self.field_registry.field_types(allowed=allowed_runtime_fields),
            complexity_limit=self.config.complexity_limit,
        )
        if not validation.is_valid:
            return None

        normalized_expression = to_expression(node)
        field_categories = {
            name: self.field_registry.get(name).category
            for name in self.field_registry.fields
        }
        signature = self.memory_service.extract_signature(
            normalized_expression,
            generation_metadata=metadata,
            field_categories=field_categories,
        )
        complexity = signature.complexity
        if complexity > self.config.complexity_limit or self._is_redundant(signature.fields):
            return None

        alpha_id = hashlib.sha1(normalized_expression.encode("utf-8")).hexdigest()[:16]
        fields_used = tuple(metadata.get("fields_used") or signature.fields)
        operators_used = tuple(metadata.get("operators_used") or signature.operators)
        template_name = str(metadata.get("template_name") or metadata.get("motif") or "")
        return AlphaCandidate(
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
