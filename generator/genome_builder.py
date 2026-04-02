from __future__ import annotations

import logging
import random
from dataclasses import replace
from collections import defaultdict

from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import NORMALIZATION_OPERATORS, OperatorRegistry
from generator.diversity_tracker import GenerationDiversityTracker
from generator.genome import (
    ComplexityGene,
    FeatureGene,
    Genome,
    HorizonGene,
    RegimeGene,
    TransformGene,
    TurnoverGene,
    WrapperGene,
)
from generator.genome import bucket_horizon
from generator.grammar import MOTIF_LIBRARY, MotifGrammar
from memory.case_memory import CaseMemorySnapshot


logger = logging.getLogger(__name__)


COMPATIBLE_CATEGORIES: dict[str, set[str]] = {
    "price": {"price", "volume", "fundamental", "analyst", "model", "risk"},
    "volume": {"price", "volume", "liquidity"},
    "fundamental": {"fundamental", "price", "analyst", "model"},
    "analyst": {"fundamental", "analyst", "price", "model"},
    "model": {"fundamental", "analyst", "model", "price"},
    "risk": {"price", "risk", "volume"},
    "liquidity": {"volume", "liquidity", "price"},
}
_UNIT_SAFE_ARITHMETIC_MOTIFS = frozenset({"spread", "residualized_signal"})
_UNIT_SAFE_CATEGORY_PAIRS = frozenset({frozenset({"fundamental", "analyst"})})


class GenomeBuilder:
    _NORMALIZE_AFTER_SMOOTHING_MOTIFS = frozenset(
        {
            "quality_score",
            "price_volume_divergence",
            "conditional_momentum",
            "regime_conditioned_signal",
            "liquidity_conditioned_signal",
        }
    )

    def __init__(
        self,
        *,
        generation_config: GenerationConfig,
        adaptive_config: AdaptiveGenerationConfig,
        registry: OperatorRegistry,
        field_registry: FieldRegistry,
        seed: int,
    ) -> None:
        self.generation_config = generation_config
        self.adaptive_config = adaptive_config
        self.registry = registry
        self.field_registry = field_registry
        self.sim_neutralization = str(getattr(generation_config, "sim_neutralization", "none") or "none").strip().lower()
        self.sim_decay = max(0, int(getattr(generation_config, "sim_decay", 0) or 0))
        self.sim_neutralization_active = self.sim_neutralization != "none"
        self.sim_decay_active = self.sim_decay > 0
        self.random = random.Random(seed)
        self._grammar = MotifGrammar()
        self.allowed_numeric_fields = self.field_registry.generation_numeric_fields(
            generation_config.allowed_fields,
            include_catalog_fields=generation_config.allow_catalog_fields_without_runtime,
        )
        self.allowed_group_fields = self.field_registry.generation_group_fields(
            include_catalog_fields=generation_config.allow_catalog_fields_without_runtime,
        )
        self.fields_by_category = self._fields_by_category(self.allowed_numeric_fields)
        self.wrapper_choices = [
            name
            for name in generation_config.normalization_wrappers
            if (
                name in NORMALIZATION_OPERATORS
                or (self.registry.contains(name) and not self.registry.get(name).has_tag("requires_positive_input"))
            )
        ] or ["rank"]
        self.primitive_ops = [
            name for name in ("ts_delta", "ts_mean", "ts_std_dev", "ts_decay_linear", "ts_rank", "ts_sum")
            if self.registry.contains(name)
        ]
        self.pair_ops = [name for name in ("ts_corr", "ts_covariance") if self.registry.contains(name)]
        self.smoothing_ops = [
            name
            for name in self.primitive_ops
            if self.registry.get(name).has_tag("smoothing")
            or self.registry.get(name).has_tag("reduces_turnover")
            or name in {"ts_mean", "ts_decay_linear", "ts_sum"}
        ]
        if not self.smoothing_ops:
            self.smoothing_ops = [name for name in ("ts_mean", "ts_decay_linear") if self.registry.contains(name)]

    def build_random_genome(
        self,
        *,
        source_mode: str = "random",
        novelty_bias: bool = False,
        case_snapshot: CaseMemorySnapshot | None = None,
        diversity_tracker: GenerationDiversityTracker | None = None,
    ) -> Genome:
        motif = self._pick_motif(
            novelty_bias=novelty_bias,
            case_snapshot=case_snapshot,
            diversity_tracker=diversity_tracker,
        )
        return self._build_genome(
            motif=motif,
            source_mode=source_mode,
            novelty_bias=novelty_bias,
            case_snapshot=case_snapshot,
            diversity_tracker=diversity_tracker,
        )

    def build_guided_genome(
        self,
        *,
        case_snapshot: CaseMemorySnapshot | None,
        explore: bool,
        diversity_tracker: GenerationDiversityTracker | None = None,
    ) -> Genome:
        novelty_bias = explore
        source_mode = "guided_explore" if explore else "guided_exploit"
        motif = self._pick_motif(
            novelty_bias=novelty_bias,
            case_snapshot=case_snapshot,
            diversity_tracker=diversity_tracker,
            support_boost=True,
        )
        return self._build_genome(
            motif=motif,
            source_mode=source_mode,
            novelty_bias=novelty_bias,
            case_snapshot=case_snapshot,
            diversity_tracker=diversity_tracker,
        )

    def build_parent_seeded_genome(
        self,
        *,
        motif: str,
        primary_family: str,
        source_mode: str,
        case_snapshot: CaseMemorySnapshot | None = None,
        diversity_tracker: GenerationDiversityTracker | None = None,
    ) -> Genome:
        return self._build_genome(
            motif=motif,
            source_mode=source_mode,
            novelty_bias=False,
            preferred_primary_family=primary_family,
            case_snapshot=case_snapshot,
            diversity_tracker=diversity_tracker,
        )

    def _build_genome(
        self,
        *,
        motif: str,
        source_mode: str,
        novelty_bias: bool,
        preferred_primary_family: str = "",
        case_snapshot: CaseMemorySnapshot | None = None,
        diversity_tracker: GenerationDiversityTracker | None = None,
    ) -> Genome:
        primary = self._pick_numeric_field(
            motif=motif,
            preferred_family=preferred_primary_family,
            novelty_bias=novelty_bias,
            case_snapshot=case_snapshot,
            diversity_tracker=diversity_tracker,
        )
        secondary = self._pick_secondary_field(
            primary,
            motif=motif,
            novelty_bias=novelty_bias,
            case_snapshot=case_snapshot,
            diversity_tracker=diversity_tracker,
        )
        auxiliary = self._pick_auxiliary_field(primary, secondary, novelty_bias=novelty_bias)
        group_field = self._pick_group_field() if motif == "group_relative_signal" or self.random.random() < 0.20 else ""
        liquidity_field = self._pick_liquidity_field(auxiliary, secondary, primary) if motif == "liquidity_conditioned_signal" else ""
        fast_window, slow_window, context_window = self._pick_windows(
            novelty_bias=novelty_bias,
            case_snapshot=case_snapshot,
        )
        primitive = self._pick_primitive(
            motif,
            primary=primary,
            novelty_bias=novelty_bias,
            case_snapshot=case_snapshot,
            diversity_tracker=diversity_tracker,
        )
        secondary_transform = self._pick_secondary_transform(
            motif,
            primitive,
            diversity_tracker=diversity_tracker,
        )
        pair_operator = self.random.choice(self.pair_ops or ["ts_corr"]) if motif not in MOTIF_LIBRARY else ""
        conditioning_mode = self._pick_conditioning_mode(motif)
        smoothing_op = self._pick_smoothing_operator(motif=motif, case_snapshot=case_snapshot) if self.smoothing_ops else ""
        # Calculate depth headroom BEFORE picking wrappers so we stay within max_depth.
        # Keep one conservative layer of headroom for deferred post-smoothing ops.
        depth_used = self._base_render_depth(motif) + 1
        if conditioning_mode != "none":
            # group_relative_signal already includes group_neutralize in base depth
            if not (motif == "group_relative_signal" and conditioning_mode == "group_neutralize"):
                depth_used += 1
        if smoothing_op:
            depth_used += 1
        wrapper_headroom = max(0, self.generation_config.max_depth - depth_used - 1)
        wrappers = self._pick_wrappers(
            motif=motif,
            case_snapshot=case_snapshot,
            novelty_bias=novelty_bias,
            diversity_tracker=diversity_tracker,
            depth_headroom=wrapper_headroom,
        )
        turnover_hint = self._estimate_turnover_hint(primitive, secondary_transform, wrappers)
        use_smoothing = bool(smoothing_op) and turnover_hint > self._smoothing_activation_threshold()
        turnover_gene = TurnoverGene(
            smoothing_operator=smoothing_op if use_smoothing else "",
            smoothing_window=slow_window if use_smoothing else 0,
            turnover_hint=turnover_hint,
        )
        genome = Genome(
            feature_gene=FeatureGene(
                primary_field=primary.name,
                primary_family=primary.category,
                secondary_field=secondary.name if secondary else "",
                secondary_family=secondary.category if secondary else "",
                auxiliary_field=auxiliary.name if auxiliary else "",
                auxiliary_family=auxiliary.category if auxiliary else "",
                group_field=group_field,
                liquidity_field=liquidity_field,
            ),
            transform_gene=TransformGene(
                motif=motif,
                primitive_transform=primitive,
                secondary_transform=secondary_transform,
                pair_operator=pair_operator,
                arithmetic_operator=self.random.choice(["-", "+", "*"]),
            ),
            horizon_gene=HorizonGene(fast_window=fast_window, slow_window=slow_window, context_window=context_window),
            wrapper_gene=WrapperGene(post_wrappers=wrappers[:1], pre_wrappers=wrappers[1:2]),
            regime_gene=RegimeGene(
                conditioning_mode=conditioning_mode,
                conditioning_field=auxiliary.name if auxiliary and conditioning_mode == "liquidity_gate" else "",
            ),
            turnover_gene=turnover_gene,
            complexity_gene=ComplexityGene(
                target_depth=3 if not wrappers else 4,
                binary_branching=2
                if motif
                in {
                    "spread",
                    "ratio",
                    "quality_score",
                    "conditional_momentum",
                    "residualized_signal",
                    "regime_conditioned_signal",
                }
                else 1,
                wrapper_budget=len(wrappers),
            ),
            source_mode=source_mode,
            sim_neutralization_active=self.sim_neutralization_active,
            sim_decay_active=self.sim_decay_active,
        )
        return self._constrain_by_actual_depth(genome)

    def _pick_motif(
        self,
        *,
        novelty_bias: bool,
        case_snapshot: CaseMemorySnapshot | None,
        diversity_tracker: GenerationDiversityTracker | None,
        support_boost: bool = False,
    ) -> str:
        motifs = list(MOTIF_LIBRARY)
        weights: list[float] = []
        motif_support = case_snapshot.stats_for_scope("motif", scope="blended") if case_snapshot else {}
        motif_neutralization_support = (
            case_snapshot.stats_for_scope("motif_neutralization", scope="blended") if case_snapshot else {}
        )
        motif_decay_support = case_snapshot.stats_for_scope("motif_decay", scope="blended") if case_snapshot else {}
        neutralization_key = self._current_neutralization_key()
        decay_key = self._current_decay_key()
        for motif in motifs:
            support = motif_support.get(motif)
            if novelty_bias:
                weight = 1.0 / max(1, support.support if support else 0)
            else:
                weight = self._motif_prior_weight(support)
            weight *= self._aggregate_preference_multiplier(
                motif_neutralization_support.get(f"{motif}|{neutralization_key}"),
                novelty_bias=novelty_bias,
            )
            weight *= self._aggregate_preference_multiplier(
                motif_decay_support.get(f"{motif}|{decay_key}"),
                novelty_bias=novelty_bias,
            )
            if motif == "mean_reversion" and self.sim_neutralization_active:
                weight *= 0.25
            if case_snapshot and not novelty_bias:
                weight *= self._motif_failure_penalty(case_snapshot, motif)
            if diversity_tracker is not None:
                weight *= diversity_tracker.motif_weight(motif)
            weights.append(max(weight, 1e-6))
        max_weight = max(weights) if weights else 1.0
        min_weight = max_weight * 0.05
        floored_weights = [max(weight, min_weight) for weight in weights]
        if support_boost and case_snapshot and not novelty_bias:
            boosted_weights: list[float] = []
            for motif, weight in zip(motifs, floored_weights, strict=True):
                support = motif_support.get(motif)
                if support is not None and float(getattr(support, "support", 0) or 0) > 0:
                    positive_outcome = max(0.0, float(getattr(support, "avg_outcome", 0.0) or 0.0))
                    weight *= 1.25 + min(0.30, positive_outcome * 0.10)
                boosted_weights.append(weight)
            floored_weights = boosted_weights
        selected = self.random.choices(motifs, weights=floored_weights, k=1)[0]
        if diversity_tracker is not None:
            domination_threshold = len(motifs) * 3
            if diversity_tracker.sampled_motif_count(selected) > domination_threshold:
                penalized_weights = list(floored_weights)
                selected_index = motifs.index(selected)
                penalized_weights[selected_index] = max(1e-6, penalized_weights[selected_index] * 0.3)
                selected = self.random.choices(motifs, weights=penalized_weights, k=1)[0]
            diversity_tracker.record_motif_pick(selected)
        return selected

    def _pick_numeric_field(
        self,
        *,
        motif: str,
        preferred_family: str,
        novelty_bias: bool,
        case_snapshot: CaseMemorySnapshot | None,
        diversity_tracker: GenerationDiversityTracker | None,
    ) -> FieldSpec:
        candidates = self.allowed_numeric_fields or self.field_registry.runtime_numeric_fields()
        weights: list[float] = []
        family_support = case_snapshot.stats_for_scope("family", scope="blended") if case_snapshot else {}
        for spec in candidates:
            weight = max(spec.field_score, 1e-6)
            if preferred_family and spec.category == preferred_family:
                weight *= 1.35
            if motif == "quality_score" and spec.category in {"fundamental", "analyst", "model"}:
                weight *= 1.50
            if motif == "price_volume_divergence" and spec.category == "price":
                weight *= 1.75
            if novelty_bias:
                weight *= 0.75 if spec.alpha_usage_count else 1.15
            else:
                weight *= 1.0 + min(0.5, family_support.get(spec.category, None).avg_outcome if spec.category in family_support else 0.0)
            if diversity_tracker is not None:
                weight *= diversity_tracker.field_family_weight(spec.category)
            weights.append(max(weight, 1e-6))
        return self.random.choices(candidates, weights=weights, k=1)[0]

    def _pick_secondary_field(
        self,
        primary: FieldSpec,
        *,
        motif: str,
        novelty_bias: bool,
        case_snapshot: CaseMemorySnapshot | None,
        diversity_tracker: GenerationDiversityTracker | None,
    ) -> FieldSpec | None:
        candidates = [spec for spec in self.allowed_numeric_fields if spec.name != primary.name]
        if not candidates:
            return None
        candidates = self._filter_unit_safe_secondary_candidates(primary, candidates, motif=motif)
        preferred_category = primary.category if not novelty_bias else ""
        weights = []
        for spec in candidates:
            weight = max(spec.field_score, 1e-6)
            if preferred_category and spec.category == preferred_category:
                weight *= 1.25
            weight *= self._compatibility_multiplier(primary.category, spec.category)
            if motif == "quality_score" and spec.category in {"fundamental", "analyst", "model"}:
                weight *= 1.50
            if motif == "price_volume_divergence" and spec.category == "volume":
                weight *= 1.90
            if novelty_bias and spec.category == primary.category:
                weight *= 0.80
            if diversity_tracker is not None:
                weight *= diversity_tracker.field_family_weight(spec.category)
            weights.append(max(weight, 1e-6))
        return self.random.choices(candidates, weights=weights, k=1)[0]

    def _pick_auxiliary_field(self, primary: FieldSpec, secondary: FieldSpec | None, *, novelty_bias: bool) -> FieldSpec | None:
        excluded = {primary.name}
        if secondary is not None:
            excluded.add(secondary.name)
        candidates = [spec for spec in self.allowed_numeric_fields if spec.name not in excluded]
        if not candidates:
            return secondary or primary
        weighted_candidates = candidates if novelty_bias else candidates[: max(1, min(len(candidates), 6))]
        weights: list[float] = []
        for spec in weighted_candidates:
            weight = max(spec.field_score, 1e-6)
            weight *= self._compatibility_multiplier(primary.category, spec.category)
            if novelty_bias:
                weight *= 1.15 if spec.alpha_usage_count <= 0 else 0.75 / max(1, spec.alpha_usage_count)
            weights.append(max(weight, 1e-6))
        return self.random.choices(weighted_candidates, weights=weights, k=1)[0]

    def _pick_group_field(self) -> str:
        if not self.allowed_group_fields:
            return "sector"
        return self.random.choice([spec.name for spec in self.allowed_group_fields])

    def _pick_liquidity_field(self, auxiliary: FieldSpec | None, secondary: FieldSpec | None, primary: FieldSpec) -> str:
        for candidate in (auxiliary, secondary, primary):
            if candidate and candidate.category in {"volume", "liquidity"}:
                return candidate.name
        return auxiliary.name if auxiliary else primary.name

    def _pick_windows(
        self,
        *,
        novelty_bias: bool,
        case_snapshot: CaseMemorySnapshot | None,
    ) -> tuple[int, int, int]:
        ordered = sorted(set(self.generation_config.lookbacks))
        fast = self.random.choice(ordered[: max(1, len(ordered) // 2)] or ordered)
        slow_choices = [value for value in ordered if value >= fast] or ordered
        if novelty_bias or case_snapshot is None:
            slow = self.random.choice(ordered if novelty_bias else slow_choices)
        else:
            horizon_stats = case_snapshot.stats_for_scope("horizon_bucket", scope="blended")
            weights = [
                max(
                    1e-6,
                    1.0 + max(0.0, horizon_stats.get(bucket_horizon(value), None).avg_outcome if bucket_horizon(value) in horizon_stats else 0.0),
                )
                for value in slow_choices
            ]
            slow = self.random.choices(slow_choices, weights=weights, k=1)[0]
        context_choices = [value for value in ordered if value >= slow] or ordered
        context = self.random.choice(context_choices)
        return int(fast), int(slow), int(context)

    def _pick_primitive(
        self,
        motif: str,
        *,
        primary: FieldSpec | None = None,
        novelty_bias: bool = False,
        case_snapshot: CaseMemorySnapshot | None = None,
        diversity_tracker: GenerationDiversityTracker | None,
    ) -> str:
        if motif == "quality_score":
            return ""
        if motif == "price_volume_divergence" and self.registry.contains("ts_corr"):
            return "ts_corr"
        if motif == "conditional_momentum" and self.registry.contains("ts_delta"):
            return "ts_delta"
        candidates = self.primitive_ops or ["ts_mean"]
        field_operator_support = case_snapshot.stats_for_scope("field_operator", scope="blended") if case_snapshot else {}
        weights: list[float] = []
        for name in candidates:
            spec = self.registry.get(name)
            weight = 1.0
            if spec.prefers_motif(motif):
                weight *= 2.5
            if motif in {"momentum", "volatility_adjusted_momentum", "liquidity_conditioned_signal", "conditional_momentum"} and spec.has_tag("change_sensitive"):
                weight *= 1.75
            if motif in {"mean_reversion", "spread", "ratio", "residualized_signal", "group_relative_signal"} and spec.has_tag("smoothing"):
                weight *= 1.75
            if motif == "group_relative_signal" and spec.has_tag("group_aware"):
                weight *= 1.25
            if primary is not None:
                weight *= self._aggregate_preference_multiplier(
                    field_operator_support.get(f"{primary.name}|{name}"),
                    novelty_bias=novelty_bias,
                )
            if diversity_tracker is not None:
                weight *= diversity_tracker.operator_weight(name)
            weights.append(max(weight, 1e-6))
        return self.random.choices(candidates, weights=weights, k=1)[0]

    def _pick_secondary_transform(
        self,
        motif: str,
        primitive: str,
        *,
        diversity_tracker: GenerationDiversityTracker | None,
    ) -> str:
        if motif in {"quality_score", "price_volume_divergence"}:
            return ""
        if motif == "conditional_momentum" and self.registry.contains("ts_mean"):
            return "ts_mean"
        if motif == "volatility_adjusted_momentum" and self.registry.contains("ts_std_dev"):
            return "ts_std_dev"
        options = [name for name in self.primitive_ops if name != primitive]
        if not options:
            return primitive
        weights: list[float] = []
        for name in options:
            spec = self.registry.get(name)
            weight = 1.0
            if spec.prefers_motif(motif):
                weight *= 2.0
            if motif in {"regime_conditioned_signal", "liquidity_conditioned_signal"} and (
                spec.has_tag("smoothing") or spec.has_tag("coverage_improving")
            ):
                weight *= 1.5
            if motif in {"spread", "ratio", "residualized_signal"} and (
                spec.has_tag("smoothing") or spec.has_tag("cross_field")
            ):
                weight *= 1.35
            if motif == "group_relative_signal" and spec.has_tag("group_aware"):
                weight *= 1.25
            if diversity_tracker is not None:
                weight *= diversity_tracker.operator_weight(name)
            weights.append(max(weight, 1e-6))
        return self.random.choices(options, weights=weights, k=1)[0]

    def _pick_wrappers(
        self,
        *,
        motif: str = "",
        case_snapshot: CaseMemorySnapshot | None = None,
        novelty_bias: bool,
        diversity_tracker: GenerationDiversityTracker | None,
        depth_headroom: int = 2,
    ) -> tuple[str, ...]:
        if not self.wrapper_choices or depth_headroom <= 0:
            return ()
        max_wrappers = min(2 if novelty_bias else 1, depth_headroom)
        count = self.random.randint(0, max_wrappers)
        if count <= 0:
            return ()
        neutralization_support = (
            case_snapshot.stats_for_scope("motif_neutralization", scope="blended")
            if motif and case_snapshot is not None and not novelty_bias
            else {}
        )
        weighted: list[tuple[str, float]] = []
        for wrapper in self.wrapper_choices:
            weight = 1.0
            if motif and neutralization_support:
                weight *= self._aggregate_preference_multiplier(
                    neutralization_support.get(f"{motif}|{wrapper}"),
                    novelty_bias=novelty_bias,
                )
            if diversity_tracker is not None:
                weight *= diversity_tracker.operator_weight(wrapper)
            weighted.append((wrapper, max(1e-6, weight)))
        selected: list[str] = []
        choices = list(weighted)
        for _ in range(min(count, len(choices))):
            labels = [item[0] for item in choices if item[0] not in selected]
            weights = [item[1] for item in choices if item[0] not in selected]
            if not labels:
                break
            selected.append(self.random.choices(labels, weights=weights, k=1)[0])
        return tuple(selected)

    def _pick_conditioning_mode(self, motif: str) -> str:
        if motif == "group_relative_signal":
            if not self.sim_neutralization_active:
                return "group_neutralize"
            labels, weights = self._conditioning_mode_options(motif)
            return self.random.choices(labels, weights=weights, k=1)[0]
        if motif == "liquidity_conditioned_signal":
            return "liquidity_gate"
        labels, weights = self._conditioning_mode_options(motif)
        return self.random.choices(labels, weights=weights, k=1)[0]

    def _conditioning_mode_options(self, motif: str) -> tuple[list[str], list[float]]:
        if motif == "group_relative_signal":
            return (
                ["none", "volatility_gate", "liquidity_gate", "group_neutralize"],
                [0.60, 0.20, 0.15, 0.05],
            )
        return (
            ["none", "volatility_gate", "liquidity_gate"],
            [0.65, 0.20, 0.15],
        )

    def _smoothing_activation_threshold(self) -> float:
        return 0.30 if self.sim_decay_active else 0.15

    def _estimate_turnover_hint(self, primitive: str, secondary: str, wrappers: tuple[str, ...]) -> float:
        hint = 0.0
        for operator in (primitive, secondary):
            if operator and self.registry.contains(operator):
                hint += self.registry.get(operator).turnover_hint
        for wrapper in wrappers:
            if wrapper and self.registry.contains(wrapper):
                hint += self.registry.get(wrapper).turnover_hint
        return hint / max(1, len([item for item in (primitive, secondary, *wrappers) if item]))

    def _pick_smoothing_operator(
        self,
        motif: str = "",
        case_snapshot: CaseMemorySnapshot | None = None,
    ) -> str:
        if not self.smoothing_ops:
            return "ts_mean"
        weights: list[float] = []
        for name in self.smoothing_ops:
            spec = self.registry.get(name)
            weight = 1.0
            if spec.has_tag("reduces_turnover"):
                weight *= 2.0
            if spec.has_tag("smoothing"):
                weight *= 1.5
            if spec.turnover_hint < 0:
                weight *= 1.0 + abs(spec.turnover_hint)
            weights.append(max(weight, 1e-6))
        return self.random.choices(self.smoothing_ops, weights=weights, k=1)[0]

    def _fields_by_category(self, fields: list[FieldSpec]) -> dict[str, list[FieldSpec]]:
        grouped: dict[str, list[FieldSpec]] = defaultdict(list)
        for spec in fields:
            grouped[spec.category].append(spec)
        return grouped

    def _compatibility_multiplier(self, primary_category: str, candidate_category: str) -> float:
        compatible = COMPATIBLE_CATEGORIES.get(str(primary_category or "").strip().lower())
        normalized_candidate = str(candidate_category or "").strip().lower()
        if not compatible or not normalized_candidate:
            return 1.0
        return 1.5 if normalized_candidate in compatible else 0.3

    def _filter_unit_safe_secondary_candidates(
        self,
        primary: FieldSpec,
        candidates: list[FieldSpec],
        *,
        motif: str,
    ) -> list[FieldSpec]:
        if not self.sim_neutralization_active or motif not in _UNIT_SAFE_ARITHMETIC_MOTIFS:
            return candidates
        safe_candidates = [
            spec
            for spec in candidates
            if self._is_unit_safe_arithmetic_pair(primary.category, spec.category)
        ]
        return safe_candidates or candidates

    def _is_unit_safe_arithmetic_pair(self, primary_category: str, candidate_category: str) -> bool:
        left = str(primary_category or "").strip().lower()
        right = str(candidate_category or "").strip().lower()
        if not left or not right:
            return False
        if left == right:
            return True
        return frozenset({left, right}) in _UNIT_SAFE_CATEGORY_PAIRS

    def _motif_prior_weight(self, support) -> float:
        if support is None:
            return 1.0
        positive_outcome = max(0.0, float(support.avg_outcome))
        confidence = 1.0 + min(5.0, float(support.support) / 2.0)
        success_multiplier = 0.5 + max(0.0, float(support.success_rate))
        return max(1.0, 1.0 + (positive_outcome * confidence * success_multiplier))

    def _current_neutralization_key(self) -> str:
        normalized = str(self.sim_neutralization or "none").strip().lower()
        return normalized or "none"

    def _current_decay_key(self) -> str:
        return str(max(0, int(self.sim_decay or 0)))

    def _aggregate_preference_multiplier(self, aggregate, *, novelty_bias: bool) -> float:
        if aggregate is None:
            return 1.0
        support_scale = min(1.0, float(getattr(aggregate, "support", 0) or 0.0) / 5.0)
        outcome_bias = max(-0.50, min(0.75, float(getattr(aggregate, "avg_outcome", 0.0) or 0.0) * 0.25))
        multiplier = 1.0 + (outcome_bias * support_scale)
        failure_rate = float(getattr(aggregate, "failure_rate", 0.0) or 0.0)
        if failure_rate > 0.70:
            multiplier *= max(0.15, 1.0 - ((failure_rate - 0.70) / 0.30))
        multiplier = max(0.10, multiplier)
        if novelty_bias:
            return 1.0 + ((multiplier - 1.0) * 0.35)
        return multiplier

    def _motif_failure_penalty(self, case_snapshot: CaseMemorySnapshot, motif: str) -> float:
        local_failures = sum(
            count
            for key, count in case_snapshot.failure_combo_counts.items()
            if motif in key.split("|")
        )
        global_failures = sum(
            count
            for key, count in case_snapshot.global_failure_combo_counts.items()
            if motif in key.split("|")
        )
        penalty = 1.0
        if local_failures >= 5:
            penalty *= max(0.15, 1.0 - (min(local_failures, 20) / 25.0))
        if global_failures >= 8:
            penalty *= max(0.20, 1.0 - (min(global_failures, 30) / 35.0))
        return penalty

    # ------------------------------------------------------------------
    # Depth constraint helpers
    # ------------------------------------------------------------------

    def _base_render_depth(self, motif: str) -> int:
        """Base AST depth without post-smoothing normalization or deferred group ops."""
        # Motifs that divide by a stabilized time-series branch still render deeply
        # even before any post-smoothing normalization is applied.
        stabilized_ts_motifs = {"volatility_adjusted_momentum", "ratio"}
        if motif in stabilized_ts_motifs:
            return 5
        # quality_score: field / (abs(field) + 1.0)
        if motif == "quality_score":
            return 4
        # Binary motifs: BinaryOp(op(x, w), op(y, w)) = depth 3
        binary_motifs = {
            "conditional_momentum",
            "regime_conditioned_signal",
            "liquidity_conditioned_signal",
            "spread",
            "residualized_signal",
        }
        if motif in binary_motifs:
            return 3
        # Pair and single-branch motifs render as one time-series call before deferred ops.
        if motif == "price_volume_divergence":
            return 2
        if motif == "group_relative_signal":
            return 2
        if motif == "momentum":
            return 2
        # mean_reversion: field - op(field, w) = depth 3
        if motif == "mean_reversion":
            return 3
        return 3

    def _estimate_render_depth(self, genome) -> int:
        """Estimate the maximum AST depth grammar.render_ast() will produce for this genome."""
        depth = self._base_render_depth(genome.motif)

        if genome.regime_gene.conditioning_mode in {"volatility_gate", "liquidity_gate"}:
            depth += 1
        if genome.turnover_gene.smoothing_operator:
            depth += 1

        if genome.motif in self._NORMALIZE_AFTER_SMOOTHING_MOTIFS:
            depth += 1
        elif genome.regime_gene.conditioning_mode in {"volatility_gate", "liquidity_gate"}:
            depth += 1

        if genome.motif == "group_relative_signal":
            depth += 1
        if genome.regime_gene.conditioning_mode == "group_neutralize" and genome.motif != "group_relative_signal":
            depth += 1

        depth += len(genome.wrapper_gene.pre_wrappers)
        depth += len(genome.wrapper_gene.post_wrappers)
        return depth

    def _constrain_by_actual_depth(self, genome: Genome) -> Genome:
        """Strip optional layers until ACTUAL rendered depth <= max_depth.

        Uses a two-phase approach:
        1. Quick estimate-based pass to strip obviously oversized genomes (cheap).
        2. Actual render+parse verification to catch estimate drift (accurate).
        """
        max_depth = self.generation_config.max_depth
        current = genome

        # Phase 1: Cheap estimate-based pre-strip (avoids unnecessary renders)
        if self._estimate_render_depth(current) > max_depth:
            current = self._strip_by_estimate(current, max_depth)

        # Phase 2: Verify with actual rendered depth and strip further if needed
        for _ in range(4):  # max 4 strip rounds
            actual_depth = self._actual_render_depth(current)
            if actual_depth is None or actual_depth <= max_depth:
                return current
            stripped = self._strip_next_optional_layer(current)
            if stripped is current:  # nothing left to strip
                logger.debug(
                    "Genome actual depth %s exceeds max %s for motif=%s after stripping all optional layers.",
                    actual_depth,
                    max_depth,
                    current.motif,
                )
                return current
            current = stripped
        return current

    def _strip_by_estimate(self, genome: Genome, max_depth: int) -> Genome:
        """Fast estimate-based stripping (no render/parse overhead)."""
        current = genome
        # Layer 1: drop pre_wrappers (lowest signal value)
        if self._estimate_render_depth(current) > max_depth and current.wrapper_gene.pre_wrappers:
            current = replace(current, wrapper_gene=replace(current.wrapper_gene, pre_wrappers=()))
        # Layer 2: drop post_wrappers
        if self._estimate_render_depth(current) > max_depth and current.wrapper_gene.post_wrappers:
            current = replace(current, wrapper_gene=replace(current.wrapper_gene, post_wrappers=()))
        # Layer 3: drop smoothing operator
        if self._estimate_render_depth(current) > max_depth and current.turnover_gene.smoothing_operator:
            current = replace(
                current,
                turnover_gene=replace(current.turnover_gene, smoothing_operator="", smoothing_window=0),
            )
        # Layer 4: drop conditioning (most costly to remove, do last)
        if self._estimate_render_depth(current) > max_depth and current.regime_gene.conditioning_mode != "none":
            current = replace(
                current,
                regime_gene=replace(current.regime_gene, conditioning_mode="none", conditioning_field=""),
            )
        return current

    def _actual_render_depth(self, genome: Genome) -> int | None:
        """Render genome to expression, parse it, and return REAL AST depth."""
        try:
            from alpha.ast_nodes import node_depth
            from alpha.parser import parse_expression

            rendered = self._grammar.render(genome)
            return node_depth(parse_expression(rendered.expression))
        except Exception:
            return None

    def _strip_next_optional_layer(self, genome: Genome) -> Genome:
        """Strip ONE optional layer in priority order (lowest value first)."""
        if genome.wrapper_gene.pre_wrappers:
            return replace(genome, wrapper_gene=replace(genome.wrapper_gene, pre_wrappers=()))
        if genome.wrapper_gene.post_wrappers:
            return replace(genome, wrapper_gene=replace(genome.wrapper_gene, post_wrappers=()))
        if genome.turnover_gene.smoothing_operator:
            return replace(
                genome,
                turnover_gene=replace(genome.turnover_gene, smoothing_operator="", smoothing_window=0),
            )
        if genome.regime_gene.conditioning_mode != "none":
            return replace(
                genome,
                regime_gene=replace(genome.regime_gene, conditioning_mode="none", conditioning_field=""),
            )
        return genome

    def _enforce_actual_depth_limit(self, genome: Genome) -> Genome:
        """Backward-compatible alias for tests that still target the old helper name."""
        return self._constrain_by_actual_depth(genome)

