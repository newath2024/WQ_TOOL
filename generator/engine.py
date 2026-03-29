from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from alpha.ast_nodes import to_expression
from alpha.parser import parse_expression
from alpha.validator import validate_expression
from core.config import GenerationConfig
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import OperatorRegistry
from generator.mutator import mutate_expressions
from generator.templates import TemplateInstance, generate_template_expressions
from memory.pattern_memory import PatternMemoryService


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
    ) -> None:
        self.config = config
        self.registry = registry
        self.field_registry = field_registry or self._fallback_field_registry(config.allowed_fields)
        self.memory_service = PatternMemoryService()

    def generate(
        self,
        count: int | None = None,
        existing_normalized: set[str] | None = None,
    ) -> list[AlphaCandidate]:
        target = count or max(self.config.template_count, self.config.template_pool_size)
        allowed_runtime_fields = self.field_registry.allowed_runtime_fields(self.config.allowed_fields) | {
            spec.name for spec in self.field_registry.runtime_group_fields()
        }
        instances = generate_template_expressions(
            field_registry=self.field_registry,
            allowed_fields=allowed_runtime_fields,
            lookbacks=self.config.lookbacks,
            template_weights=self.config.template_weights,
            template_pool_size=max(target * 3, self.config.template_pool_size),
            max_turnover_bias=self.config.max_turnover_bias,
            seed=self.config.random_seed,
            registry=self.registry,
        )
        payload = [
            (
                instance.expression,
                "template",
                (),
                {
                    "template_name": instance.template_name,
                    "fields_used": list(instance.fields_used),
                    "template_params": instance.parameters,
                },
            )
            for instance in instances[:target]
        ]
        return self._build_candidates(payload, existing_normalized=existing_normalized)

    def generate_mutations(
        self,
        parents: Sequence[AlphaCandidate],
        count: int | None = None,
        existing_normalized: set[str] | None = None,
    ) -> list[AlphaCandidate]:
        target = count or self.config.mutation_count
        payload = [
            (
                expression,
                "mutation",
                parent_ids,
                metadata,
            )
            for expression, parent_ids, metadata in mutate_expressions(
                parents=parents,
                count=target,
                field_registry=self.field_registry,
                lookbacks=self.config.lookbacks,
                normalization_wrappers=self.config.normalization_wrappers,
                seed=self.config.random_seed + len(parents),
            )
        ]
        return self._build_candidates(payload, existing_normalized=existing_normalized)

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

        signature = self.memory_service.extract_signature(expression)
        normalized_expression = to_expression(node)
        complexity = signature.complexity
        if complexity > self.config.complexity_limit or self._is_redundant(signature.fields):
            return None

        alpha_id = hashlib.sha1(normalized_expression.encode("utf-8")).hexdigest()[:16]
        fields_used = tuple(metadata.get("fields_used") or signature.fields)
        operators_used = tuple(metadata.get("operators_used") or signature.operators)
        template_name = str(metadata.get("template_name") or "")
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

    def build_candidate_from_template(
        self,
        instance: TemplateInstance,
        mode: str,
        parent_ids: tuple[str, ...] = (),
        generation_metadata: dict[str, Any] | None = None,
    ) -> AlphaCandidate | None:
        metadata = dict(generation_metadata or {})
        metadata.setdefault("template_name", instance.template_name)
        metadata.setdefault("fields_used", list(instance.fields_used))
        metadata.setdefault("template_params", instance.parameters)
        return self.build_candidate(
            expression=instance.expression,
            mode=mode,
            parent_ids=parent_ids,
            generation_metadata=metadata,
        )

    def _is_redundant(self, fields: tuple[str, ...]) -> bool:
        return len(fields) == 0

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
