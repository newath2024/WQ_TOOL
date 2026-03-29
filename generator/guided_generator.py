from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry
from features.registry import OperatorRegistry
from generator.engine import AlphaCandidate, AlphaGenerationEngine
from generator.mutation_policy import MutationPolicy
from generator.templates import generate_template_expressions
from memory.pattern_memory import MemoryParent, PatternMemoryService, PatternMemorySnapshot


class GuidedGenerator:
    def __init__(
        self,
        generation_config: GenerationConfig,
        adaptive_config: AdaptiveGenerationConfig,
        registry: OperatorRegistry,
        memory_service: PatternMemoryService,
        field_registry: FieldRegistry,
    ) -> None:
        self.generation_config = generation_config
        self.adaptive_config = adaptive_config
        self.registry = registry
        self.memory_service = memory_service
        self.field_registry = field_registry
        self.base_engine = AlphaGenerationEngine(
            config=generation_config,
            registry=registry,
            field_registry=field_registry,
        )
        self.random = random.Random(generation_config.random_seed)
        self.mutation_policy = MutationPolicy(
            config=generation_config,
            memory_service=memory_service,
            randomizer=self.random,
            field_registry=field_registry,
        )

    def generate(
        self,
        count: int,
        snapshot: PatternMemorySnapshot,
        existing_normalized: set[str] | None = None,
    ) -> list[AlphaCandidate]:
        if not self.adaptive_config.enabled or not snapshot.patterns:
            return self.base_engine.generate(count=count, existing_normalized=existing_normalized)

        quotas = self._allocate_counts(count)
        existing = set(existing_normalized or set())
        selected: list[AlphaCandidate] = []
        family_counts: dict[str, int] = defaultdict(int)
        family_cap = max(1, int(math.ceil(count * self.adaptive_config.family_cap_fraction)))
        allowed_fields = self.field_registry.allowed_runtime_fields(self.generation_config.allowed_fields) | {
            spec.name for spec in self.field_registry.runtime_group_fields()
        }

        parents = list(snapshot.top_parents[: self.adaptive_config.parent_pool_size])
        selected.extend(
            self._fill_mode(
                target_count=quotas["guided_mutation"],
                existing_normalized=existing,
                family_counts=family_counts,
                family_cap=family_cap,
                payload_builder=lambda shortfall, attempt: self._guided_mutation_payload(
                    parents=parents,
                    snapshot=snapshot,
                    target_count=max(shortfall * (attempt + 1), shortfall),
                ),
            )
        )
        selected.extend(
            self._fill_mode(
                target_count=quotas["memory_templates"],
                existing_normalized=existing,
                family_counts=family_counts,
                family_cap=family_cap,
                payload_builder=lambda shortfall, attempt: self._template_payload(
                    snapshot=snapshot,
                    allowed_fields=allowed_fields,
                    target_count=max(shortfall * (attempt + 1), shortfall),
                    mode="memory_template",
                    memory_boost=True,
                    novelty_boost=False,
                ),
            )
        )
        selected.extend(
            self._fill_mode(
                target_count=quotas["random_exploration"],
                existing_normalized=existing,
                family_counts=family_counts,
                family_cap=family_cap,
                payload_builder=lambda shortfall, attempt: self._template_payload(
                    snapshot=snapshot,
                    allowed_fields=allowed_fields,
                    target_count=max(shortfall * (attempt + 2), shortfall),
                    mode="structured_exploration",
                    memory_boost=False,
                    novelty_boost=False,
                ),
            )
        )
        selected.extend(
            self._fill_mode(
                target_count=quotas["novelty_search"],
                existing_normalized=existing,
                family_counts=family_counts,
                family_cap=family_cap,
                payload_builder=lambda shortfall, attempt: self._novelty_payload(
                    snapshot=snapshot,
                    allowed_fields=allowed_fields,
                    target_count=max(shortfall * (attempt + 2), shortfall),
                ),
            )
        )

        if len(selected) < count:
            selected.extend(
                self._fill_mode(
                    target_count=count - len(selected),
                    existing_normalized=existing,
                    family_counts=family_counts,
                    family_cap=family_cap,
                    payload_builder=lambda shortfall, attempt: self._template_payload(
                        snapshot=snapshot,
                        allowed_fields=allowed_fields,
                        target_count=max(shortfall * (attempt + 2), shortfall),
                        mode="structured_exploration",
                        memory_boost=False,
                        novelty_boost=False,
                    ),
                )
            )
        return selected[:count]

    def generate_mutations(
        self,
        count: int,
        snapshot: PatternMemorySnapshot,
        parent_pool: list[MemoryParent],
        existing_normalized: set[str] | None = None,
    ) -> list[AlphaCandidate]:
        existing = set(existing_normalized or set())
        family_counts: dict[str, int] = defaultdict(int)
        family_cap = max(1, int(math.ceil(count * self.adaptive_config.family_cap_fraction)))
        return self._fill_mode(
            target_count=count,
            existing_normalized=existing,
            family_counts=family_counts,
            family_cap=family_cap,
            payload_builder=lambda shortfall, attempt: self._guided_mutation_payload(
                parents=parent_pool,
                snapshot=snapshot,
                target_count=max(shortfall * (attempt + 1), shortfall),
            ),
        )[:count]

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

    def _guided_mutation_payload(
        self,
        parents: list[MemoryParent],
        snapshot: PatternMemorySnapshot,
        target_count: int,
    ) -> list[tuple[str, str, tuple[str, ...], dict]]:
        if not parents or target_count <= 0:
            return []
        parent_weights = [self._parent_weight(parent) for parent in parents]
        payload: list[tuple[str, str, tuple[str, ...], dict]] = []
        attempts = 0
        max_attempts = max(target_count * 12, 24)
        while len(payload) < target_count and attempts < max_attempts:
            attempts += 1
            parent = self.random.choices(parents, weights=parent_weights, k=1)[0]
            expressions = self.mutation_policy.generate(
                parent,
                snapshot,
                target_count=max(2, min(4, target_count)),
                force_novelty=False,
            )
            for expression, metadata in expressions:
                payload.append((expression, "guided_mutation", (parent.alpha_id,), metadata))
                if len(payload) >= target_count:
                    break
        return payload[:target_count]

    def _template_payload(
        self,
        *,
        snapshot: PatternMemorySnapshot,
        allowed_fields: set[str],
        target_count: int,
        mode: str,
        memory_boost: bool,
        novelty_boost: bool,
    ) -> list[tuple[str, str, tuple[str, ...], dict]]:
        if target_count <= 0:
            return []
        instances = generate_template_expressions(
            field_registry=self.field_registry,
            allowed_fields=allowed_fields,
            lookbacks=self.generation_config.lookbacks,
            template_weights=self.generation_config.template_weights,
            template_pool_size=max(target_count * 4, self.generation_config.template_pool_size),
            max_turnover_bias=self.generation_config.max_turnover_bias,
            seed=self.generation_config.random_seed + self.random.randint(0, 100_000),
            registry=self.registry,
            field_memory=self._field_memory(snapshot) if memory_boost or novelty_boost else None,
            template_memory=self._template_memory(snapshot) if memory_boost or novelty_boost else None,
        )
        scored: list[tuple[TemplateCandidateScore, dict]] = []
        for instance in instances:
            score, novelty, _, observations = self.memory_service.score_expression(
                instance.expression,
                snapshot,
                min_pattern_support=1 if novelty_boost else self.adaptive_config.min_pattern_support,
            )
            composite = novelty if novelty_boost else score + (0.15 * novelty if memory_boost else 0.0)
            scored.append(
                (
                    TemplateCandidateScore(
                        expression=instance.expression,
                        score=composite,
                        novelty=novelty,
                    ),
                    {
                        "template_name": instance.template_name,
                        "fields_used": list(instance.fields_used),
                        "template_params": instance.parameters,
                        "source_pattern_ids": [
                            item.pattern_id for item in observations if item.pattern_kind != "subexpression"
                        ],
                        "source_gene_ids": [
                            item.pattern_id for item in observations if item.pattern_kind == "subexpression"
                        ],
                        "mutation_hint_tags": ["diversify_feature_family"] if novelty_boost else [],
                        "target_novelty": novelty_boost,
                    },
                )
            )
        ordered = sorted(scored, key=lambda item: (item[0].score, item[0].novelty, item[0].expression), reverse=True)
        payload: list[tuple[str, str, tuple[str, ...], dict]] = []
        for candidate_score, metadata in ordered[:target_count]:
            payload.append((candidate_score.expression, mode, (), metadata))
        return payload

    def _novelty_payload(
        self,
        *,
        snapshot: PatternMemorySnapshot,
        allowed_fields: set[str],
        target_count: int,
    ) -> list[tuple[str, str, tuple[str, ...], dict]]:
        payload = self._template_payload(
            snapshot=snapshot,
            allowed_fields=allowed_fields,
            target_count=target_count,
            mode="novelty_search",
            memory_boost=True,
            novelty_boost=True,
        )
        if payload:
            return payload
        return []

    def _fill_mode(
        self,
        target_count: int,
        existing_normalized: set[str],
        family_counts: dict[str, int],
        family_cap: int,
        payload_builder: Callable[[int, int], list[tuple[str, str, tuple[str, ...], dict]]],
    ) -> list[AlphaCandidate]:
        if target_count <= 0:
            return []
        built: list[AlphaCandidate] = []
        attempts = 0
        max_attempts = 6
        while len(built) < target_count and attempts < max_attempts:
            attempts += 1
            shortfall = target_count - len(built)
            payload = payload_builder(shortfall, attempts)
            if not payload:
                break
            new_candidates = self._build_from_payload(
                payload=payload,
                existing_normalized=existing_normalized,
                family_counts=family_counts,
                family_cap=family_cap,
                limit=shortfall,
            )
            if not new_candidates:
                continue
            built.extend(new_candidates)
        return built

    def _build_from_payload(
        self,
        payload: list[tuple[str, str, tuple[str, ...], dict]],
        existing_normalized: set[str],
        family_counts: dict[str, int],
        family_cap: int,
        limit: int | None = None,
    ) -> list[AlphaCandidate]:
        built: list[AlphaCandidate] = []
        for expression, mode, parent_ids, metadata in payload:
            if limit is not None and len(built) >= limit:
                break
            try:
                signature = self.memory_service.extract_signature(expression)
            except ValueError:
                continue
            family_signature = signature.family_signature
            if family_counts[family_signature] >= family_cap:
                continue
            candidate = self.base_engine.build_candidate(
                expression=expression,
                mode=mode,
                parent_ids=parent_ids,
                generation_metadata=metadata,
            )
            if candidate is None or candidate.normalized_expression in existing_normalized:
                continue
            existing_normalized.add(candidate.normalized_expression)
            family_counts[family_signature] += 1
            built.append(candidate)
        return built

    def _field_memory(self, snapshot: PatternMemorySnapshot) -> dict[str, float]:
        return {
            item.pattern_value: item.pattern_score
            for item in snapshot.by_kind("field")
        }

    def _template_memory(self, snapshot: PatternMemorySnapshot) -> dict[str, float]:
        return {
            item.pattern_value: item.pattern_score
            for item in snapshot.by_kind("template")
        }

    def _parent_weight(self, parent: MemoryParent) -> float:
        diversity_penalty = 0.25 * len(parent.fail_tags)
        novelty_bonus = 0.15 * parent.behavioral_novelty_score
        return max(
            self.adaptive_config.exploration_epsilon,
            math.exp(
                (parent.outcome_score + novelty_bonus - diversity_penalty)
                / max(self.adaptive_config.sampling_temperature, 1e-6)
            ),
        )


@dataclass(frozen=True, slots=True)
class TemplateCandidateScore:
    expression: str
    score: float
    novelty: float
