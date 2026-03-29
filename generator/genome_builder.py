from __future__ import annotations

import random
from collections import defaultdict

from core.config import AdaptiveGenerationConfig, GenerationConfig
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import NORMALIZATION_OPERATORS, OperatorRegistry
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
from generator.grammar import MOTIF_LIBRARY
from memory.case_memory import CaseMemorySnapshot


class GenomeBuilder:
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
        self.random = random.Random(seed)
        self.allowed_numeric_fields = self.field_registry.runtime_numeric_fields(
            allowed=self.field_registry.allowed_runtime_fields(generation_config.allowed_fields)
        )
        self.allowed_group_fields = self.field_registry.runtime_group_fields(
            allowed=self.field_registry.allowed_runtime_fields(generation_config.allowed_fields)
            | {spec.name for spec in self.field_registry.runtime_group_fields()}
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
    ) -> Genome:
        motif = self._pick_motif(novelty_bias=novelty_bias, case_snapshot=case_snapshot)
        return self._build_genome(motif=motif, source_mode=source_mode, novelty_bias=novelty_bias, case_snapshot=case_snapshot)

    def build_guided_genome(
        self,
        *,
        case_snapshot: CaseMemorySnapshot | None,
        explore: bool,
    ) -> Genome:
        novelty_bias = explore
        source_mode = "guided_explore" if explore else "guided_exploit"
        motif = self._pick_motif(novelty_bias=novelty_bias, case_snapshot=case_snapshot)
        return self._build_genome(motif=motif, source_mode=source_mode, novelty_bias=novelty_bias, case_snapshot=case_snapshot)

    def build_parent_seeded_genome(
        self,
        *,
        motif: str,
        primary_family: str,
        source_mode: str,
        case_snapshot: CaseMemorySnapshot | None = None,
    ) -> Genome:
        return self._build_genome(
            motif=motif,
            source_mode=source_mode,
            novelty_bias=False,
            preferred_primary_family=primary_family,
            case_snapshot=case_snapshot,
        )

    def _build_genome(
        self,
        *,
        motif: str,
        source_mode: str,
        novelty_bias: bool,
        preferred_primary_family: str = "",
        case_snapshot: CaseMemorySnapshot | None = None,
    ) -> Genome:
        primary = self._pick_numeric_field(preferred_family=preferred_primary_family, novelty_bias=novelty_bias, case_snapshot=case_snapshot)
        secondary = self._pick_secondary_field(primary, novelty_bias=novelty_bias, case_snapshot=case_snapshot)
        auxiliary = self._pick_auxiliary_field(primary, secondary, novelty_bias=novelty_bias)
        group_field = self._pick_group_field() if motif == "group_relative_signal" or self.random.random() < 0.20 else ""
        liquidity_field = self._pick_liquidity_field(auxiliary, secondary, primary) if motif == "liquidity_conditioned_signal" else ""
        fast_window, slow_window, context_window = self._pick_windows(novelty_bias=novelty_bias)
        primitive = self._pick_primitive(motif)
        secondary_transform = self._pick_secondary_transform(motif, primitive)
        pair_operator = self.random.choice(self.pair_ops or ["ts_corr"]) if motif not in MOTIF_LIBRARY else ""
        wrappers = self._pick_wrappers(novelty_bias=novelty_bias)
        conditioning_mode = self._pick_conditioning_mode(motif)
        turnover_hint = self._estimate_turnover_hint(primitive, secondary_transform, wrappers)
        turnover_gene = TurnoverGene(
            smoothing_operator=self._pick_smoothing_operator() if turnover_hint > 0.15 and self.smoothing_ops else "",
            smoothing_window=slow_window if turnover_hint > 0.15 else 0,
            turnover_hint=turnover_hint,
        )
        return Genome(
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
                binary_branching=2 if motif in {"spread", "ratio", "residualized_signal", "regime_conditioned_signal"} else 1,
                wrapper_budget=len(wrappers),
            ),
            source_mode=source_mode,
        )

    def _pick_motif(self, *, novelty_bias: bool, case_snapshot: CaseMemorySnapshot | None) -> str:
        motifs = list(MOTIF_LIBRARY)
        weights: list[float] = []
        motif_support = case_snapshot.motif_stats if case_snapshot else {}
        for motif in motifs:
            support = motif_support.get(motif)
            if novelty_bias:
                weight = 1.0 / max(1, support.support if support else 0)
            else:
                weight = max(1.0, 1.0 + (support.avg_outcome if support else 0.0))
            weights.append(max(weight, 1e-6))
        return self.random.choices(motifs, weights=weights, k=1)[0]

    def _pick_numeric_field(
        self,
        *,
        preferred_family: str,
        novelty_bias: bool,
        case_snapshot: CaseMemorySnapshot | None,
    ) -> FieldSpec:
        candidates = self.allowed_numeric_fields or self.field_registry.runtime_numeric_fields()
        weights: list[float] = []
        family_support = case_snapshot.family_stats if case_snapshot else {}
        for spec in candidates:
            weight = max(spec.field_score, 1e-6)
            if preferred_family and spec.category == preferred_family:
                weight *= 1.35
            if novelty_bias:
                weight *= 0.75 if spec.alpha_usage_count else 1.15
            else:
                weight *= 1.0 + min(0.5, family_support.get(spec.category, None).avg_outcome if spec.category in family_support else 0.0)
            weights.append(max(weight, 1e-6))
        return self.random.choices(candidates, weights=weights, k=1)[0]

    def _pick_secondary_field(
        self,
        primary: FieldSpec,
        *,
        novelty_bias: bool,
        case_snapshot: CaseMemorySnapshot | None,
    ) -> FieldSpec | None:
        candidates = [spec for spec in self.allowed_numeric_fields if spec.name != primary.name]
        if not candidates:
            return None
        preferred_category = primary.category if not novelty_bias else ""
        weights = []
        for spec in candidates:
            weight = max(spec.field_score, 1e-6)
            if preferred_category and spec.category == preferred_category:
                weight *= 1.25
            if novelty_bias and spec.category == primary.category:
                weight *= 0.80
            weights.append(max(weight, 1e-6))
        return self.random.choices(candidates, weights=weights, k=1)[0]

    def _pick_auxiliary_field(self, primary: FieldSpec, secondary: FieldSpec | None, *, novelty_bias: bool) -> FieldSpec | None:
        excluded = {primary.name}
        if secondary is not None:
            excluded.add(secondary.name)
        candidates = [spec for spec in self.allowed_numeric_fields if spec.name not in excluded]
        if not candidates:
            return secondary or primary
        if novelty_bias:
            candidates = sorted(candidates, key=lambda item: (item.alpha_usage_count, item.field_score, item.name))
            return candidates[0]
        return self.random.choice(candidates[: max(1, min(len(candidates), 6))])

    def _pick_group_field(self) -> str:
        if not self.allowed_group_fields:
            return "sector"
        return self.random.choice([spec.name for spec in self.allowed_group_fields])

    def _pick_liquidity_field(self, auxiliary: FieldSpec | None, secondary: FieldSpec | None, primary: FieldSpec) -> str:
        for candidate in (auxiliary, secondary, primary):
            if candidate and candidate.category in {"volume", "liquidity"}:
                return candidate.name
        return auxiliary.name if auxiliary else primary.name

    def _pick_windows(self, *, novelty_bias: bool) -> tuple[int, int, int]:
        ordered = sorted(set(self.generation_config.lookbacks))
        fast = self.random.choice(ordered[: max(1, len(ordered) // 2)] or ordered)
        if novelty_bias:
            slow = self.random.choice(ordered)
        else:
            slow_choices = [value for value in ordered if value >= fast] or ordered
            slow = self.random.choice(slow_choices)
        context_choices = [value for value in ordered if value >= slow] or ordered
        context = self.random.choice(context_choices)
        return int(fast), int(slow), int(context)

    def _pick_primitive(self, motif: str) -> str:
        candidates = self.primitive_ops or ["ts_mean"]
        weights: list[float] = []
        for name in candidates:
            spec = self.registry.get(name)
            weight = 1.0
            if spec.prefers_motif(motif):
                weight *= 2.5
            if motif in {"momentum", "volatility_adjusted_momentum", "liquidity_conditioned_signal"} and spec.has_tag("change_sensitive"):
                weight *= 1.75
            if motif in {"mean_reversion", "spread", "ratio", "residualized_signal", "group_relative_signal"} and spec.has_tag("smoothing"):
                weight *= 1.75
            if motif == "group_relative_signal" and spec.has_tag("group_aware"):
                weight *= 1.25
            weights.append(max(weight, 1e-6))
        return self.random.choices(candidates, weights=weights, k=1)[0]

    def _pick_secondary_transform(self, motif: str, primitive: str) -> str:
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
            weights.append(max(weight, 1e-6))
        return self.random.choices(options, weights=weights, k=1)[0]

    def _pick_wrappers(self, *, novelty_bias: bool) -> tuple[str, ...]:
        if not self.wrapper_choices:
            return ()
        max_wrappers = 2 if novelty_bias else 1
        count = self.random.randint(0, max_wrappers)
        if count <= 0:
            return ()
        return tuple(self.random.sample(self.wrapper_choices, k=min(count, len(self.wrapper_choices))))

    def _pick_conditioning_mode(self, motif: str) -> str:
        if motif == "group_relative_signal":
            return "group_neutralize"
        if motif == "liquidity_conditioned_signal":
            return "liquidity_gate"
        return self.random.choices(
            ["none", "volatility_gate", "liquidity_gate"],
            weights=[0.65, 0.20, 0.15],
            k=1,
        )[0]

    def _estimate_turnover_hint(self, primitive: str, secondary: str, wrappers: tuple[str, ...]) -> float:
        hint = 0.0
        for operator in (primitive, secondary):
            if operator and self.registry.contains(operator):
                hint += self.registry.get(operator).turnover_hint
        for wrapper in wrappers:
            if wrapper and self.registry.contains(wrapper):
                hint += self.registry.get(wrapper).turnover_hint
        return hint / max(1, len([item for item in (primitive, secondary, *wrappers) if item]))

    def _pick_smoothing_operator(self) -> str:
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
