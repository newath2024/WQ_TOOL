from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Iterable, Sequence

from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import OperatorRegistry, build_default_registry
from generator.crossover import GenomeCrossover
from generator.diversity_tracker import GenerationDiversityTracker
from generator.genome import Genome
from generator.genome_builder import GenomeBuilder
from generator.grammar import MOTIF_LIBRARY, MotifGrammar
from generator.novelty import NoveltySearch
from generator.repair_policy import RepairPolicy
from memory.case_memory import CaseMemoryService, CaseMemorySnapshot
from memory.pattern_memory import MemoryParent, PatternMemoryService, PatternMemorySnapshot


class MutationPolicy:
    MODES = ("exploit_local", "structural", "crossover", "novelty", "repair")

    def __init__(
        self,
        *,
        config: GenerationConfig,
        adaptive_config: AdaptiveGenerationConfig | None = None,
        memory_service: PatternMemoryService,
        mutation_learning_records: list[dict[str, Any]] | None = None,
        randomizer_seed: int | None = None,
        randomizer: random.Random | None = None,
        field_registry: FieldRegistry | None = None,
        registry: OperatorRegistry | None = None,
    ) -> None:
        self.config = config
        self.adaptive_config = adaptive_config or AdaptiveGenerationConfig()
        self.memory_service = memory_service
        self.case_memory_service = CaseMemoryService()
        self.mutation_learning_records = list(mutation_learning_records or [])
        self.randomizer = randomizer or random.Random(randomizer_seed if randomizer_seed is not None else config.random_seed)
        self.field_registry = field_registry or self._fallback_field_registry(config.allowed_fields)
        self.registry = registry or build_default_registry()
        self.genome_builder = GenomeBuilder(
            generation_config=config,
            adaptive_config=self.adaptive_config,
            registry=self.registry,
            field_registry=self.field_registry,
            seed=(randomizer_seed if randomizer_seed is not None else config.random_seed) + 13,
        )
        self.grammar = MotifGrammar()
        self.repair_policy = RepairPolicy(
            generation_config=config,
            repair_config=self.adaptive_config.repair_policy,
            field_registry=self.field_registry,
            registry=self.registry,
        )
        self.crossover = GenomeCrossover(self.randomizer)
        self.novelty_search = NoveltySearch()

    def generate(
        self,
        parent: MemoryParent,
        snapshot: PatternMemorySnapshot,
        target_count: int,
        force_novelty: bool = False,
        case_snapshot: CaseMemorySnapshot | None = None,
        diversity_tracker: GenerationDiversityTracker | None = None,
    ) -> list[tuple[str, dict]]:
        parent_candidate = self._memory_parent_to_candidate(parent)
        payload = self.generate_from_candidates(
            parents=[parent_candidate],
            target_count=target_count,
            case_snapshot=case_snapshot,
            pattern_snapshot=snapshot,
            force_novelty=force_novelty,
            diversity_tracker=diversity_tracker,
        )
        return [(expression, metadata) for expression, _, _, metadata in payload]

    def generate_from_candidates(
        self,
        *,
        parents: Sequence,
        target_count: int,
        case_snapshot: CaseMemorySnapshot | None,
        pattern_snapshot: PatternMemorySnapshot | None = None,
        force_novelty: bool = False,
        diversity_tracker: GenerationDiversityTracker | None = None,
    ) -> list[tuple[str, str, tuple[str, ...], dict[str, Any]]]:
        payload: list[tuple[str, str, tuple[str, ...], dict[str, Any]]] = []
        if not parents or target_count <= 0:
            return payload
        attempts = 0
        max_attempts = max(target_count * 12, 24)
        parent_refs = {getattr(parent, "alpha_id"): parent for parent in parents if getattr(parent, "alpha_id", "")}
        while len(payload) < target_count and attempts < max_attempts:
            attempts += 1
            primary_parent = self._select_parent(parents, diversity_tracker=diversity_tracker)
            parent_genome = self._extract_parent_genome(primary_parent)
            mutation_mode = self._choose_mode(
                primary_parent,
                case_snapshot=case_snapshot,
                force_novelty=force_novelty,
                diversity_tracker=diversity_tracker,
            )
            mutated = self._mutate_genome(
                parent_genome,
                mutation_mode=mutation_mode,
                parents=list(parents),
                case_snapshot=case_snapshot,
                diversity_tracker=diversity_tracker,
            )
            repaired, repair_actions = self.repair_policy.repair(
                mutated,
                fail_tags=tuple(getattr(primary_parent, "fail_tags", ()) or ()),
            )
            render = self.grammar.render(repaired)
            metadata = {
                "template_name": repaired.motif,
                "motif": repaired.motif,
                "genome": repaired.to_dict(),
                "genome_hash": repaired.stable_hash,
                "fields_used": list(render.field_names),
                "field_families": list(render.field_families),
                "operators_used": list(dict.fromkeys(render.operator_path)),
                "operator_path": list(render.operator_path),
                "operator_path_key": ">".join(render.operator_path[:4]) if render.operator_path else "none",
                "operator_semantic_tags": self._operator_semantic_tags(render.operator_path),
                "turnover_bucket": render.turnover_bucket,
                "horizon_bucket": render.horizon_bucket,
                "complexity_bucket": render.complexity_bucket,
                "mutation_mode": mutation_mode,
                "pre_normalized_expression": render.normalized_expression,
                "repair_actions": list(repair_actions),
                "parent_refs": self._build_parent_refs(primary_parent, parent_refs, mutation_mode),
                "mutation_hint_tags": list(getattr(primary_parent, "mutation_hints", ()) or ()),
            }
            payload.append((render.expression, mutation_mode, tuple(ref["alpha_id"] for ref in metadata["parent_refs"]), metadata))
        return payload

    def _choose_mode(
        self,
        parent,
        *,
        case_snapshot: CaseMemorySnapshot | None,
        force_novelty: bool,
        diversity_tracker: GenerationDiversityTracker | None = None,
    ) -> str:
        if force_novelty:
            return "novelty"
        family_signature = str(getattr(parent, "family_signature", "") or "")
        fail_tags = tuple(getattr(parent, "fail_tags", ()) or ())
        weights = self.case_memory_service.mutation_mode_preferences(
            family_signature=family_signature,
            fail_tags=fail_tags,
            snapshot=case_snapshot,
        )
        config_weights = self.adaptive_config.mutation_mode_weights
        weights = {
            "exploit_local": weights["exploit_local"] * config_weights.exploit_local,
            "structural": weights["structural"] * config_weights.structural,
            "crossover": weights["crossover"] * (config_weights.crossover if self.randomizer.random() < self.adaptive_config.crossover_rate else 0.01),
            "novelty": weights["novelty"] * config_weights.novelty,
            "repair": weights["repair"] * config_weights.repair,
        }
        outcome_multipliers = self._mutation_outcome_multipliers(family_signature=family_signature)
        for mode, multiplier in outcome_multipliers.items():
            weights[mode] = weights.get(mode, 0.0) * multiplier
        if diversity_tracker is not None:
            for mode in list(weights):
                weights[mode] = weights.get(mode, 0.0) * diversity_tracker.mutation_mode_weight(mode)
        return self.randomizer.choices(list(weights.keys()), weights=[max(value, 1e-6) for value in weights.values()], k=1)[0]

    def _mutate_genome(
        self,
        parent_genome: Genome,
        *,
        mutation_mode: str,
        parents: list,
        case_snapshot: CaseMemorySnapshot | None,
        diversity_tracker: GenerationDiversityTracker | None,
    ) -> Genome:
        if mutation_mode == "exploit_local":
            return self._exploit_local(parent_genome)
        if mutation_mode == "structural":
            return self._structural(parent_genome, case_snapshot=case_snapshot, diversity_tracker=diversity_tracker)
        if mutation_mode == "crossover" and len(parents) > 1:
            partner = self._extract_parent_genome(self.randomizer.choice([item for item in parents if item is not parents[0]] or parents))
            return self.crossover.crossover(parent_genome, partner)
        if mutation_mode == "novelty":
            return self._novelty(parent_genome, case_snapshot=case_snapshot, diversity_tracker=diversity_tracker)
        return self._repair_seed(parent_genome)

    def _exploit_local(self, genome: Genome) -> Genome:
        ordered = sorted(set(self.config.lookbacks))
        slower_choices = [value for value in ordered if value >= genome.horizon_gene.fast_window] or ordered
        next_window = self.randomizer.choice(slower_choices)
        wrappers = genome.wrapper_gene.post_wrappers or tuple(self.config.normalization_wrappers[:1])
        return replace(
            genome,
            horizon_gene=replace(genome.horizon_gene, slow_window=next_window, context_window=max(next_window, genome.horizon_gene.context_window)),
            wrapper_gene=replace(genome.wrapper_gene, post_wrappers=wrappers[:1]),
            source_mode="exploit_local",
        )

    def _structural(
        self,
        genome: Genome,
        *,
        case_snapshot: CaseMemorySnapshot | None,
        diversity_tracker: GenerationDiversityTracker | None,
    ) -> Genome:
        motif_choices = [motif for motif in MOTIF_LIBRARY if motif != genome.motif] or [genome.motif]
        seeded = self.genome_builder.build_parent_seeded_genome(
            motif=self.randomizer.choice(motif_choices),
            primary_family=genome.feature_gene.primary_family,
            source_mode="structural",
            case_snapshot=case_snapshot,
            diversity_tracker=diversity_tracker,
        )
        return replace(seeded, feature_gene=replace(seeded.feature_gene, primary_field=genome.feature_gene.primary_field, primary_family=genome.feature_gene.primary_family))

    def _novelty(
        self,
        genome: Genome,
        *,
        case_snapshot: CaseMemorySnapshot | None,
        diversity_tracker: GenerationDiversityTracker | None,
    ) -> Genome:
        novel = self.genome_builder.build_guided_genome(
            case_snapshot=case_snapshot,
            explore=True,
            diversity_tracker=diversity_tracker,
        )
        return replace(
            novel,
            feature_gene=replace(
                novel.feature_gene,
                primary_field=novel.feature_gene.primary_field,
                primary_family=novel.feature_gene.primary_family,
            ),
            source_mode="novelty",
        )

    def _repair_seed(self, genome: Genome) -> Genome:
        repaired, _ = self.repair_policy.repair(genome, fail_tags=("high_turnover", "excessive_complexity"))
        return replace(repaired, source_mode="repair")

    def _extract_parent_genome(self, parent) -> Genome:
        metadata = dict(getattr(parent, "generation_metadata", {}) or {})
        genome = self.case_memory_service.genome_from_metadata(metadata)
        if genome is not None:
            return genome
        motif = str(metadata.get("motif") or metadata.get("template_name") or "momentum")
        fields_used = tuple(metadata.get("fields_used") or ())
        primary_field = fields_used[0] if fields_used else getattr(parent, "fields_used", ("close",))[0] if getattr(parent, "fields_used", ()) else "close"
        primary_family = self.field_registry.get(primary_field).category if self.field_registry.contains(primary_field) else "other"
        return self.genome_builder.build_parent_seeded_genome(
            motif=motif,
            primary_family=primary_family,
            source_mode="inferred_parent",
        )

    def _build_parent_refs(self, primary_parent, parent_refs: dict[str, Any], mutation_mode: str) -> list[dict[str, str]]:
        refs = [
            {
                "run_id": str(getattr(primary_parent, "run_id", "") or ""),
                "alpha_id": str(getattr(primary_parent, "alpha_id", "") or ""),
                "family_signature": str(getattr(primary_parent, "family_signature", "") or ""),
            }
        ]
        if mutation_mode == "crossover":
            other_ids = [alpha_id for alpha_id in parent_refs if alpha_id != getattr(primary_parent, "alpha_id", "")]
            if other_ids:
                partner = parent_refs[self.randomizer.choice(other_ids)]
                refs.append(
                    {
                        "run_id": str(getattr(partner, "run_id", "") or ""),
                        "alpha_id": str(getattr(partner, "alpha_id", "") or ""),
                        "family_signature": str(getattr(partner, "family_signature", "") or ""),
                    }
                )
        return [ref for ref in refs if ref["alpha_id"]]

    def _memory_parent_to_candidate(self, parent: MemoryParent):
        class CandidateLike:
            def __init__(self) -> None:
                self.alpha_id = parent.alpha_id
                self.run_id = parent.run_id
                self.generation_metadata = parent.generation_metadata
                self.fail_tags = parent.fail_tags
                self.family_signature = parent.family_signature
                self.fields_used = tuple(parent.generation_metadata.get("fields_used", ()))
                self.expression = parent.expression

        return CandidateLike()

    def _fallback_field_registry(self, allowed_fields: list[str]) -> FieldRegistry:
        return FieldRegistry(
            fields={
                name: FieldSpec(
                    name=name,
                    dataset="fallback",
                    field_type="vector" if name in {"sector", "industry", "country", "subindustry"} else "matrix",
                    coverage=1.0,
                    alpha_usage_count=0,
                    category="group" if name in {"sector", "industry", "country", "subindustry"} else "other",
                    runtime_available=True,
                    category_weight=0.5,
                    field_score=1.0,
                )
                for name in allowed_fields
            }
        )

    def _operator_semantic_tags(self, operator_path: tuple[str, ...]) -> list[str]:
        tags: set[str] = set()
        for name in operator_path:
            if not self.registry.contains(name):
                continue
            tags.update(self.registry.get(name).semantic_tags)
        return sorted(tags)

    def _select_parent(self, parents: Sequence, *, diversity_tracker: GenerationDiversityTracker | None) -> Any:
        parent_list = list(parents)
        if diversity_tracker is None or len(parent_list) <= 1:
            return self.randomizer.choice(parent_list)
        weights: list[float] = []
        for parent in parent_list:
            parent_id = str(getattr(parent, "alpha_id", "") or "")
            family_signature = str(getattr(parent, "family_signature", "") or "")
            lineage_key = parent_id or family_signature
            weights.append(max(1e-6, diversity_tracker.lineage_weight(lineage_key)))
        return self.randomizer.choices(parent_list, weights=weights, k=1)[0]

    def _mutation_outcome_multipliers(self, *, family_signature: str) -> dict[str, float]:
        learning_config = self.adaptive_config.mutation_learning
        if not learning_config.enabled or not self.mutation_learning_records:
            return {}
        by_mode: dict[str, list[float]] = {}
        for row in self.mutation_learning_records:
            row_family = str(row.get("family_signature") or "")
            if family_signature and row_family and row_family != family_signature:
                continue
            mode = str(row.get("mutation_mode") or "")
            if not mode:
                continue
            by_mode.setdefault(mode, []).append(float(row.get("outcome_delta") or 0.0))
        multipliers: dict[str, float] = {}
        for mode, deltas in by_mode.items():
            support = len(deltas)
            if support < learning_config.min_support:
                continue
            success_rate = sum(1 for value in deltas if value > 0.0) / support
            avg_uplift = sum(deltas) / support
            negative_penalty = max(0.0, -avg_uplift) * learning_config.negative_lift_penalty
            multiplier = (
                1.0
                + learning_config.success_rate_weight * success_rate
                + learning_config.uplift_weight * avg_uplift
                - negative_penalty
            )
            multipliers[mode] = max(0.10, float(multiplier))
        return multipliers
