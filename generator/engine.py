from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from alpha.ast_nodes import ExprNode, IdentifierNode, NumberNode, UnaryOpNode, node_complexity, to_expression
from alpha.parser import parse_expression
from alpha.validator import validate_expression
from core.config import GenerationConfig
from features.registry import OperatorRegistry
from generator.grammar import GrammarExpressionGenerator
from generator.mutator import mutate_expressions
from generator.templates import generate_template_expressions


@dataclass(frozen=True, slots=True)
class AlphaCandidate:
    alpha_id: str
    expression: str
    normalized_expression: str
    generation_mode: str
    parent_ids: tuple[str, ...]
    complexity: int
    created_at: str
    generation_metadata: dict[str, Any] = field(default_factory=dict)


class AlphaGenerationEngine:
    def __init__(self, config: GenerationConfig, registry: OperatorRegistry) -> None:
        self.config = config
        self.registry = registry

    def generate(
        self,
        count: int | None = None,
        existing_normalized: set[str] | None = None,
    ) -> list[AlphaCandidate]:
        target = count or (self.config.template_count + self.config.grammar_count)
        template_target = min(self.config.template_count, target)
        grammar_target = max(0, target - template_target)

        payload: list[tuple[str, str, tuple[str, ...], dict[str, Any]]] = []
        for expression in generate_template_expressions(
            fields=self.config.allowed_fields,
            lookbacks=self.config.lookbacks,
            normalization_wrappers=self.config.normalization_wrappers,
        )[:template_target]:
            payload.append((expression, "template", (), {}))

        grammar_generator = GrammarExpressionGenerator(
            fields=self.config.allowed_fields,
            lookbacks=self.config.lookbacks,
            max_depth=self.config.max_depth,
            seed=self.config.random_seed,
        )
        for expression in grammar_generator.generate(grammar_target):
            payload.append((expression, "grammar", (), {}))

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
                {"parent_refs": [{"alpha_id": parent_id} for parent_id in parent_ids]},
            )
            for expression, parent_ids in mutate_expressions(
                parents=[(parent.alpha_id, parent.expression) for parent in parents],
                count=target,
                fields=self.config.allowed_fields,
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
        try:
            node = parse_expression(expression)
        except ValueError:
            return None

        validation = validate_expression(
            node=node,
            registry=self.registry,
            allowed_fields=set(self.config.allowed_fields),
            max_depth=self.config.max_depth,
        )
        if not validation.is_valid:
            return None

        normalized_expression = to_expression(node)
        complexity = node_complexity(node)
        if complexity > self.config.complexity_limit or self._is_redundant(node):
            return None

        alpha_id = hashlib.sha1(normalized_expression.encode("utf-8")).hexdigest()[:16]
        return AlphaCandidate(
            alpha_id=alpha_id,
            expression=expression.strip(),
            normalized_expression=normalized_expression,
            generation_mode=mode,
            parent_ids=parent_ids,
            complexity=complexity,
            created_at=datetime.now(timezone.utc).isoformat(),
            generation_metadata=generation_metadata or {},
        )

    def _is_redundant(self, node: ExprNode) -> bool:
        if isinstance(node, (IdentifierNode, NumberNode)):
            return True
        if isinstance(node, UnaryOpNode) and isinstance(node.operand, NumberNode):
            return True
        return False
