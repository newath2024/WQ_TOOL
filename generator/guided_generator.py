from __future__ import annotations

import math
import random

from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry
from features.registry import OperatorRegistry
from generator.engine import AlphaCandidate, AlphaGenerationEngine
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
        )

    def generate(
        self,
        count: int,
        snapshot: PatternMemorySnapshot,
        existing_normalized: set[str] | None = None,
        case_snapshot: CaseMemorySnapshot | None = None,
    ) -> list[AlphaCandidate]:
        if not self.adaptive_config.enabled:
            return self.base_engine.generate(
                count=count,
                existing_normalized=existing_normalized,
                case_snapshot=case_snapshot,
            )

        existing = set(existing_normalized or set())
        candidates: list[AlphaCandidate] = []
        exploration_count = max(1, int(math.ceil(count * self.adaptive_config.exploration_ratio)))
        exploitation_count = max(0, count - exploration_count)
        attempts = 0
        max_attempts = max(count * 25, 80)
        while len(candidates) < exploitation_count and attempts < max_attempts:
            attempts += 1
            genome = self.genome_builder.build_guided_genome(case_snapshot=case_snapshot, explore=False)
            repaired, repair_actions = self.repair_policy.repair(genome)
            render = self.grammar.render(repaired)
            candidate = self.base_engine.build_candidate(
                expression=render.expression,
                mode="guided_exploit",
                parent_ids=(),
                generation_metadata=self.base_engine._render_metadata(  # noqa: SLF001
                    render,
                    mutation_mode="exploit_local",
                    repair_actions=repair_actions,
                    memory_context=self.base_engine._memory_context_metadata(  # noqa: SLF001
                        pattern_snapshot=snapshot,
                        case_snapshot=case_snapshot,
                    ),
                ),
            )
            if candidate is None or candidate.normalized_expression in existing:
                continue
            existing.add(candidate.normalized_expression)
            candidates.append(candidate)
        attempts = 0
        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            genome = self.genome_builder.build_guided_genome(case_snapshot=case_snapshot, explore=True)
            repaired, repair_actions = self.repair_policy.repair(genome)
            render = self.grammar.render(repaired)
            candidate = self.base_engine.build_candidate(
                expression=render.expression,
                mode="guided_explore",
                parent_ids=(),
                generation_metadata=self.base_engine._render_metadata(  # noqa: SLF001
                    render,
                    mutation_mode="novelty",
                    repair_actions=repair_actions,
                    memory_context=self.base_engine._memory_context_metadata(  # noqa: SLF001
                        pattern_snapshot=snapshot,
                        case_snapshot=case_snapshot,
                    ),
                ),
            )
            if candidate is None or candidate.normalized_expression in existing:
                continue
            existing.add(candidate.normalized_expression)
            candidates.append(candidate)
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
            return []
        attempts = 0
        max_attempts = max(count * 20, 40)
        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            parent = self.random.choice(parent_pool)
            variants = self.mutation_policy.generate(
                parent=parent,
                snapshot=snapshot,
                target_count=1,
                force_novelty=False,
                case_snapshot=case_snapshot,
            )
            if not variants:
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
            candidate = self.base_engine.build_candidate(
                expression=expression,
                mode=str(metadata.get("mutation_mode") or "exploit_local"),
                parent_ids=parent_ids,
                generation_metadata=metadata,
            )
            if candidate is None or candidate.normalized_expression in existing:
                continue
            existing.add(candidate.normalized_expression)
            candidates.append(candidate)
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
