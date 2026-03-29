from __future__ import annotations

import random
from dataclasses import replace

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


class GenomeCrossover:
    def __init__(self, randomizer: random.Random) -> None:
        self.randomizer = randomizer

    def crossover(self, left: Genome, right: Genome) -> Genome:
        feature_gene = FeatureGene(
            primary_field=self._pick(left.feature_gene.primary_field, right.feature_gene.primary_field),
            primary_family=self._pick(left.feature_gene.primary_family, right.feature_gene.primary_family),
            secondary_field=self._pick(left.feature_gene.secondary_field, right.feature_gene.secondary_field),
            secondary_family=self._pick(left.feature_gene.secondary_family, right.feature_gene.secondary_family),
            auxiliary_field=self._pick(left.feature_gene.auxiliary_field, right.feature_gene.auxiliary_field),
            auxiliary_family=self._pick(left.feature_gene.auxiliary_family, right.feature_gene.auxiliary_family),
            group_field=self._pick(left.feature_gene.group_field, right.feature_gene.group_field),
            liquidity_field=self._pick(left.feature_gene.liquidity_field, right.feature_gene.liquidity_field),
        )
        transform_gene = TransformGene(
            motif=self._pick(left.transform_gene.motif, right.transform_gene.motif),
            primitive_transform=self._pick(left.transform_gene.primitive_transform, right.transform_gene.primitive_transform),
            secondary_transform=self._pick(left.transform_gene.secondary_transform, right.transform_gene.secondary_transform),
            pair_operator=self._pick(left.transform_gene.pair_operator, right.transform_gene.pair_operator),
            arithmetic_operator=self._pick(left.transform_gene.arithmetic_operator, right.transform_gene.arithmetic_operator),
            residualization_mode=self._pick(
                left.transform_gene.residualization_mode,
                right.transform_gene.residualization_mode,
            ),
        )
        horizon_gene = HorizonGene(
            fast_window=int(self._pick(left.horizon_gene.fast_window, right.horizon_gene.fast_window) or left.horizon_gene.fast_window),
            slow_window=int(self._pick(left.horizon_gene.slow_window, right.horizon_gene.slow_window) or left.horizon_gene.slow_window),
            context_window=int(
                self._pick(left.horizon_gene.context_window, right.horizon_gene.context_window) or left.horizon_gene.context_window
            ),
        )
        wrapper_gene = WrapperGene(
            pre_wrappers=self._merge_wrappers(left.wrapper_gene.pre_wrappers, right.wrapper_gene.pre_wrappers),
            post_wrappers=self._merge_wrappers(left.wrapper_gene.post_wrappers, right.wrapper_gene.post_wrappers),
        )
        regime_gene = RegimeGene(
            conditioning_mode=self._pick(left.regime_gene.conditioning_mode, right.regime_gene.conditioning_mode),
            conditioning_field=self._pick(left.regime_gene.conditioning_field, right.regime_gene.conditioning_field),
            invert_condition=bool(self._pick(left.regime_gene.invert_condition, right.regime_gene.invert_condition)),
        )
        turnover_gene = TurnoverGene(
            smoothing_operator=self._pick(left.turnover_gene.smoothing_operator, right.turnover_gene.smoothing_operator),
            smoothing_window=int(self._pick(left.turnover_gene.smoothing_window, right.turnover_gene.smoothing_window) or 0),
            turnover_hint=float(self._pick(left.turnover_gene.turnover_hint, right.turnover_gene.turnover_hint) or 0.0),
        )
        complexity_gene = ComplexityGene(
            target_depth=int(self._pick(left.complexity_gene.target_depth, right.complexity_gene.target_depth) or 3),
            binary_branching=int(self._pick(left.complexity_gene.binary_branching, right.complexity_gene.binary_branching) or 1),
            wrapper_budget=int(self._pick(left.complexity_gene.wrapper_budget, right.complexity_gene.wrapper_budget) or 1),
        )
        child = Genome(
            feature_gene=feature_gene,
            transform_gene=transform_gene,
            horizon_gene=horizon_gene,
            wrapper_gene=wrapper_gene,
            regime_gene=regime_gene,
            turnover_gene=turnover_gene,
            complexity_gene=complexity_gene,
            source_mode="crossover",
        )
        return self._repair_incompatible_child(child, left=left, right=right)

    def _repair_incompatible_child(self, child: Genome, *, left: Genome, right: Genome) -> Genome:
        motif = child.motif
        if motif in {"spread", "ratio", "residualized_signal", "regime_conditioned_signal"} and not child.feature_gene.secondary_field:
            child = replace(
                child,
                feature_gene=replace(
                    child.feature_gene,
                    secondary_field=right.feature_gene.secondary_field or right.feature_gene.primary_field,
                    secondary_family=right.feature_gene.secondary_family or right.feature_gene.primary_family,
                ),
            )
        if motif == "group_relative_signal" and not child.feature_gene.group_field:
            child = replace(child, feature_gene=replace(child.feature_gene, group_field=left.feature_gene.group_field or "sector"))
        if motif == "liquidity_conditioned_signal" and not child.feature_gene.liquidity_field:
            liquidity_field = right.feature_gene.liquidity_field or right.feature_gene.auxiliary_field or right.feature_gene.primary_field
            child = replace(child, feature_gene=replace(child.feature_gene, liquidity_field=liquidity_field))
        return child

    def _merge_wrappers(self, left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
        merged: list[str] = []
        for wrapper in left + right:
            if wrapper and wrapper not in merged and self.randomizer.random() < 0.5:
                merged.append(wrapper)
        return tuple(merged[:2])

    def _pick(self, left, right):
        return self.randomizer.choice([value for value in (left, right) if value not in {None, ""}] or [left, right])

