from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from alpha.ast_nodes import BinaryOpNode, ExprNode, FunctionCallNode, IdentifierNode, NumberNode, to_expression
from alpha.validator import has_nesting_violation
from generator.genome import Genome, GenomeRenderResult


_CROSS_SECTIONAL_WRAPPERS = frozenset(
    {"rank", "zscore", "group_rank", "group_zscore", "group_neutralize", "normalize"}
)


MOTIF_LIBRARY: tuple[str, ...] = (
    "momentum",
    "mean_reversion",
    "volatility_adjusted_momentum",
    "spread",
    "ratio",
    "quality_score",
    "price_volume_divergence",
    "conditional_momentum",
    "residualized_signal",
    "regime_conditioned_signal",
    "group_relative_signal",
    "liquidity_conditioned_signal",
)


@dataclass(frozen=True, slots=True)
class MotifSpec:
    name: str
    required_fields: int
    uses_group: bool = False
    uses_liquidity: bool = False


class MotifGrammar:
    def __init__(self) -> None:
        self.specs = {
            "momentum": MotifSpec("momentum", 1),
            "mean_reversion": MotifSpec("mean_reversion", 1),
            "volatility_adjusted_momentum": MotifSpec("volatility_adjusted_momentum", 1),
            "spread": MotifSpec("spread", 2),
            "ratio": MotifSpec("ratio", 2),
            "quality_score": MotifSpec("quality_score", 2),
            "price_volume_divergence": MotifSpec("price_volume_divergence", 2),
            "conditional_momentum": MotifSpec("conditional_momentum", 2),
            "residualized_signal": MotifSpec("residualized_signal", 2),
            "regime_conditioned_signal": MotifSpec("regime_conditioned_signal", 2),
            "group_relative_signal": MotifSpec("group_relative_signal", 1, uses_group=True),
            "liquidity_conditioned_signal": MotifSpec("liquidity_conditioned_signal", 1, uses_liquidity=True),
        }

    def render(self, genome: Genome) -> GenomeRenderResult:
        node = self.render_ast(genome)
        normalized_expression = to_expression(node)
        payload = genome.family_signature_payload()
        family_signature = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return GenomeRenderResult(
            genome=genome,
            expression=normalized_expression,
            normalized_expression=normalized_expression,
            family_signature=family_signature,
            operator_path=genome.operator_path,
            field_names=genome.field_names,
            field_families=genome.field_families,
            wrappers=genome.wrapper_gene.all_wrappers(),
            horizon_bucket=genome.horizon_bucket,
            turnover_bucket=genome.turnover_bucket,
            complexity_bucket=genome.complexity_bucket,
        )

    def render_ast(self, genome: Genome) -> ExprNode:
        motif = genome.transform_gene.motif
        primary = IdentifierNode(genome.feature_gene.primary_field)
        secondary = IdentifierNode(genome.feature_gene.secondary_field or genome.feature_gene.primary_field)
        auxiliary = IdentifierNode(genome.feature_gene.auxiliary_field or genome.feature_gene.secondary_field or genome.feature_gene.primary_field)
        group_node = IdentifierNode(genome.feature_gene.group_field) if genome.feature_gene.group_field else None
        liquidity_node = IdentifierNode(genome.feature_gene.liquidity_field) if genome.feature_gene.liquidity_field else None
        fast_window = NumberNode(float(max(1, genome.horizon_gene.fast_window)))
        slow_window = NumberNode(float(max(1, genome.horizon_gene.slow_window or genome.horizon_gene.fast_window)))
        context_window = NumberNode(float(max(1, genome.horizon_gene.context_window or genome.horizon_gene.fast_window)))
        normalize_after_smoothing = False
        deferred_group_neutralizers: list[ExprNode] = []

        if motif == "momentum":
            signal = self._call(genome.transform_gene.primitive_transform or "ts_delta", primary, slow_window)
        elif motif == "mean_reversion":
            signal = BinaryOpNode(
                operator="-",
                left=primary,
                right=self._call(genome.transform_gene.primitive_transform or "ts_mean", primary, slow_window),
            )
        elif motif == "volatility_adjusted_momentum":
            numerator = self._call(genome.transform_gene.primitive_transform or "ts_delta", primary, fast_window)
            denominator = self._stabilize_divisor(
                self._call(genome.transform_gene.secondary_transform or "ts_std_dev", primary, slow_window)
            )
            signal = BinaryOpNode(operator="/", left=numerator, right=denominator)
        elif motif == "spread":
            left = self._call(genome.transform_gene.primitive_transform or "ts_mean", primary, slow_window)
            right = self._call(genome.transform_gene.secondary_transform or "ts_mean", secondary, slow_window)
            signal = BinaryOpNode(operator=genome.transform_gene.arithmetic_operator or "-", left=left, right=right)
        elif motif == "ratio":
            numerator = self._call(genome.transform_gene.primitive_transform or "ts_mean", primary, slow_window)
            denominator = self._stabilize_divisor(
                self._call(genome.transform_gene.secondary_transform or "ts_mean", secondary, slow_window)
            )
            signal = BinaryOpNode(operator="/", left=numerator, right=denominator)
        elif motif == "quality_score":
            signal = BinaryOpNode(
                operator="/",
                left=primary,
                right=self._stabilize_divisor(secondary),
            )
            normalize_after_smoothing = True
        elif motif == "price_volume_divergence":
            signal = self._call(genome.transform_gene.primitive_transform or "ts_corr", primary, secondary, slow_window)
            normalize_after_smoothing = True
        elif motif == "conditional_momentum":
            momentum = self._call(genome.transform_gene.primitive_transform or "ts_delta", primary, fast_window)
            conditioner = self._call(genome.transform_gene.secondary_transform or "ts_mean", secondary, slow_window)
            signal = BinaryOpNode(operator="*", left=momentum, right=conditioner)
            normalize_after_smoothing = True
        elif motif == "residualized_signal":
            base = self._call(genome.transform_gene.primitive_transform or "ts_mean", primary, slow_window)
            residual = self._call(genome.transform_gene.secondary_transform or "ts_mean", secondary, slow_window)
            signal = BinaryOpNode(operator="-", left=base, right=residual)
        elif motif == "regime_conditioned_signal":
            base = self._call(genome.transform_gene.primitive_transform or "ts_delta", primary, slow_window)
            regime = self._call(genome.transform_gene.secondary_transform or "ts_std_dev", secondary, context_window)
            signal = BinaryOpNode(operator="*", left=base, right=regime)
            normalize_after_smoothing = True
        elif motif == "group_relative_signal":
            signal = self._call(genome.transform_gene.primitive_transform or "ts_mean", primary, slow_window)
            deferred_group_neutralizers.append(group_node or IdentifierNode("sector"))
        elif motif == "liquidity_conditioned_signal":
            base = self._call(genome.transform_gene.primitive_transform or "ts_delta", primary, slow_window)
            liquidity = self._call(genome.transform_gene.secondary_transform or "ts_mean", liquidity_node or auxiliary, context_window)
            signal = BinaryOpNode(operator="*", left=base, right=liquidity)
            normalize_after_smoothing = True
        else:
            pair_operator = genome.transform_gene.pair_operator or "ts_corr"
            signal = self._call(pair_operator, primary, secondary, slow_window)

        if genome.regime_gene.conditioning_mode == "volatility_gate":
            conditioner = self._call("ts_std_dev", auxiliary, context_window)
            signal = BinaryOpNode(operator="*", left=signal, right=conditioner)
            normalize_after_smoothing = True
        elif genome.regime_gene.conditioning_mode == "liquidity_gate":
            condition_field = IdentifierNode(genome.regime_gene.conditioning_field or genome.feature_gene.liquidity_field or auxiliary.name)
            conditioner = self._call("ts_mean", condition_field, context_window)
            signal = BinaryOpNode(operator="*", left=signal, right=conditioner)
            normalize_after_smoothing = True
        elif genome.regime_gene.conditioning_mode == "group_neutralize":
            deferred_group_neutralizers.append(group_node or IdentifierNode("sector"))

        if genome.turnover_gene.smoothing_operator and genome.turnover_gene.smoothing_window > 0:
            signal = self._call(
                genome.turnover_gene.smoothing_operator,
                signal,
                NumberNode(float(genome.turnover_gene.smoothing_window)),
            )

        if normalize_after_smoothing:
            signal = self._normalize(signal)
        for group_field in deferred_group_neutralizers:
            signal = self._call("group_neutralize", signal, group_field)
        for wrapper in genome.wrapper_gene.pre_wrappers:
            candidate = self._call(wrapper, signal)
            if wrapper in _CROSS_SECTIONAL_WRAPPERS and has_nesting_violation(candidate):
                continue
            signal = candidate
        for wrapper in genome.wrapper_gene.post_wrappers:
            candidate = self._call(wrapper, signal)
            if wrapper in _CROSS_SECTIONAL_WRAPPERS and has_nesting_violation(candidate):
                continue
            signal = candidate
        return signal

    def _call(self, name: str, *args: ExprNode) -> FunctionCallNode:
        return FunctionCallNode(name=name, args=tuple(arg for arg in args if arg is not None))

    def _normalize(self, node: ExprNode) -> ExprNode:
        return self._call("rank", node)

    def _stabilize_divisor(self, node: ExprNode) -> ExprNode:
        return BinaryOpNode(operator="+", left=self._call("abs", node), right=NumberNode(1.0))
