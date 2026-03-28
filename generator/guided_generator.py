from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable

from core.config import AdaptiveGenerationConfig, GenerationConfig
from features.registry import OperatorRegistry
from generator.engine import AlphaCandidate, AlphaGenerationEngine
from generator.grammar import GrammarExpressionGenerator
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
    ) -> None:
        self.generation_config = generation_config
        self.adaptive_config = adaptive_config
        self.registry = registry
        self.memory_service = memory_service
        self.base_engine = AlphaGenerationEngine(config=generation_config, registry=registry)
        self.random = random.Random(generation_config.random_seed)
        self.mutation_policy = MutationPolicy(
            config=generation_config,
            memory_service=memory_service,
            randomizer=self.random,
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
                    attempt=attempt,
                ),
            )
        )
        selected.extend(
            self._fill_mode(
                target_count=quotas["memory_templates"],
                existing_normalized=existing,
                family_counts=family_counts,
                family_cap=family_cap,
                payload_builder=lambda shortfall, attempt: self._memory_weighted_template_payload(
                    snapshot=snapshot,
                    target_count=max(shortfall * (attempt + 2), shortfall),
                    attempt=attempt,
                ),
            )
        )
        selected.extend(
            self._fill_mode(
                target_count=quotas["random_exploration"],
                existing_normalized=existing,
                family_counts=family_counts,
                family_cap=family_cap,
                payload_builder=lambda shortfall, attempt: self._random_payload(
                    target_count=max(shortfall * (attempt + 2), shortfall),
                    attempt=attempt,
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
                    target_count=max(shortfall * (attempt + 2), shortfall),
                    attempt=attempt,
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
                    payload_builder=lambda shortfall, attempt: self._random_payload(
                        target_count=max(shortfall * (attempt + 2), shortfall),
                        attempt=attempt + 100,
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
                attempt=attempt,
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
        attempt: int = 0,
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
                target_count=max(2, min(4 + attempt, target_count)),
                force_novelty=False,
            )
            for expression, metadata in expressions:
                payload.append(
                    (
                        expression,
                        "guided_mutation",
                        (parent.alpha_id,),
                        metadata,
                    )
                )
                if len(payload) >= target_count:
                    break
            if all(weight <= 0 for weight in parent_weights):
                break
        return payload[:target_count]

    def _memory_weighted_template_payload(
        self,
        snapshot: PatternMemorySnapshot,
        target_count: int,
        attempt: int = 0,
    ) -> list[tuple[str, str, tuple[str, ...], dict]]:
        if target_count <= 0:
            return []
        template_pool = generate_template_expressions(
            fields=self.generation_config.allowed_fields,
            lookbacks=self.generation_config.lookbacks,
            normalization_wrappers=self.generation_config.normalization_wrappers,
        )
        scored_pool = []
        for expression in template_pool:
            score, novelty, _, observations = self.memory_service.score_expression(
                expression,
                snapshot,
                min_pattern_support=self.adaptive_config.min_pattern_support,
            )
            scored_pool.append(
                (
                    expression,
                    score,
                    novelty,
                    {
                        "source_pattern_ids": [item.pattern_id for item in observations if item.pattern_kind != "subexpression"],
                        "source_gene_ids": [item.pattern_id for item in observations if item.pattern_kind == "subexpression"],
                        "mutation_hint_tags": [],
                        "target_novelty": False,
                    },
                )
            )
        return self._weighted_payload(
            scored_pool,
            target_count=target_count,
            mode="memory_template",
            attempt=attempt,
        )

    def _random_payload(
        self,
        target_count: int,
        attempt: int = 0,
    ) -> list[tuple[str, str, tuple[str, ...], dict]]:
        if target_count <= 0:
            return []
        grammar_generator = GrammarExpressionGenerator(
            fields=self.generation_config.allowed_fields,
            lookbacks=self.generation_config.lookbacks,
            max_depth=self.generation_config.max_depth,
            seed=self.generation_config.random_seed + target_count + (attempt * 9973) + self.random.randint(0, 10_000),
        )
        template_pool = generate_template_expressions(
            fields=self.generation_config.allowed_fields,
            lookbacks=self.generation_config.lookbacks,
            normalization_wrappers=self.generation_config.normalization_wrappers,
        )
        grammar_pool = grammar_generator.generate(max(target_count * 3, self.generation_config.grammar_count))
        pool = template_pool + grammar_pool
        self.random.shuffle(pool)
        payload: list[tuple[str, str, tuple[str, ...], dict]] = []
        for expression in pool[: target_count * 3]:
            payload.append((expression, "random_exploration", (), {"target_novelty": False, "mutation_hint_tags": []}))
            if len(payload) >= target_count:
                break
        return payload

    def _novelty_payload(
        self,
        snapshot: PatternMemorySnapshot,
        target_count: int,
        attempt: int = 0,
    ) -> list[tuple[str, str, tuple[str, ...], dict]]:
        if target_count <= 0:
            return []
        novelty_parents = sorted(
            snapshot.top_parents,
            key=lambda item: (item.behavioral_novelty_score, item.outcome_score),
            reverse=True,
        )
        payload: list[tuple[str, str, tuple[str, ...], dict]] = []
        if novelty_parents:
            for parent in novelty_parents[: max(1, self.adaptive_config.parent_pool_size // 2)]:
                expressions = self.mutation_policy.generate(
                    parent,
                    snapshot,
                    target_count=max(2, min(4 + attempt, target_count)),
                    force_novelty=True,
                )
                for expression, metadata in expressions:
                    payload.append((expression, "novelty_search", (parent.alpha_id,), metadata))
                    if len(payload) >= target_count:
                        return payload[:target_count]

        template_pool = generate_template_expressions(
            fields=self.generation_config.allowed_fields,
            lookbacks=self.generation_config.lookbacks,
            normalization_wrappers=self.generation_config.normalization_wrappers,
        )
        scored_pool = []
        for expression in template_pool:
            score, novelty, _, observations = self.memory_service.score_expression(
                expression,
                snapshot,
                min_pattern_support=1,
            )
            scored_pool.append(
                (
                    expression,
                    novelty - 0.25 * max(score, 0.0),
                    novelty,
                    {
                        "source_pattern_ids": [item.pattern_id for item in observations if item.pattern_kind != "subexpression"],
                        "source_gene_ids": [item.pattern_id for item in observations if item.pattern_kind == "subexpression"],
                        "mutation_hint_tags": ["diversify_feature_family"],
                        "target_novelty": True,
                    },
                )
            )
        payload.extend(
            self._weighted_payload(
                scored_pool,
                target_count=target_count - len(payload),
                mode="novelty_search",
                attempt=attempt,
            )
        )
        return payload[:target_count]

    def _weighted_payload(
        self,
        scored_pool: list[tuple[str, float, float, dict]],
        target_count: int,
        mode: str,
        attempt: int = 0,
    ) -> list[tuple[str, str, tuple[str, ...], dict]]:
        if target_count <= 0 or not scored_pool:
            return []
        ordered = sorted(
            scored_pool,
            key=lambda item: (item[1], item[2], item[0]),
            reverse=True,
        )
        remaining = list(ordered)
        used_expressions: set[str] = set()
        payload: list[tuple[str, str, tuple[str, ...], dict]] = []
        attempts = 0
        max_attempts = max(target_count * 12, 24)
        while len(payload) < target_count and remaining and attempts < max_attempts:
            attempts += 1
            weights = [
                self._sampling_weight(
                    score=item[1] + (0.10 * item[2]) - (0.03 * attempt),
                )
                for item in remaining
            ]
            index = self.random.choices(range(len(remaining)), weights=weights, k=1)[0]
            expression, _, _, metadata = remaining.pop(index)
            if expression in used_expressions:
                continue
            used_expressions.add(expression)
            payload.append((expression, mode, (), metadata))
        return payload

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
            signature = self.memory_service.extract_signature(expression)
            observations = self.memory_service.build_observations(signature)
            family_signature = signature.family_signature
            if family_counts[family_signature] >= family_cap:
                continue
            generation_metadata = dict(metadata)
            generation_metadata.setdefault(
                "source_pattern_ids",
                [item.pattern_id for item in observations if item.pattern_kind != "subexpression"],
            )
            generation_metadata.setdefault(
                "source_gene_ids",
                [item.pattern_id for item in observations if item.pattern_kind == "subexpression"],
            )
            generation_metadata.setdefault("mutation_hint_tags", [])
            generation_metadata.setdefault("target_novelty", False)
            candidate = self.base_engine.build_candidate(
                expression=expression,
                mode=mode,
                parent_ids=parent_ids,
                generation_metadata=generation_metadata,
            )
            if candidate is None or candidate.normalized_expression in existing_normalized:
                continue
            existing_normalized.add(candidate.normalized_expression)
            family_counts[family_signature] += 1
            built.append(candidate)
        return built

    def _parent_weight(self, parent: MemoryParent) -> float:
        diversity_penalty = 0.25 * len(parent.fail_tags)
        novelty_bonus = 0.15 * parent.behavioral_novelty_score
        return max(
            self.adaptive_config.exploration_epsilon,
            math.exp((parent.outcome_score + novelty_bonus - diversity_penalty) / max(self.adaptive_config.sampling_temperature, 1e-6)),
        )

    def _sampling_weight(self, score: float) -> float:
        return self.adaptive_config.exploration_epsilon + math.exp(
            score / max(self.adaptive_config.sampling_temperature, 1e-6)
        )
