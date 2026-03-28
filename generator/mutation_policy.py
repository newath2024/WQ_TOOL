from __future__ import annotations

import random
from typing import Iterable

from alpha.ast_nodes import BinaryOpNode, FunctionCallNode, UnaryOpNode, to_expression
from alpha.parser import parse_expression
from core.config import GenerationConfig
from generator.mutator import _swap_field, _swap_lookback, _swap_operator
from memory.pattern_memory import MemoryParent, PatternMemoryService, PatternMemorySnapshot


class MutationPolicy:
    def __init__(
        self,
        config: GenerationConfig,
        memory_service: PatternMemoryService,
        randomizer: random.Random,
    ) -> None:
        self.config = config
        self.memory_service = memory_service
        self.randomizer = randomizer
        self.operator_swaps = {
            "ts_mean": ["ts_std", "rolling_mean", "decay_linear"],
            "ts_std": ["ts_mean", "rolling_std"],
            "delta": ["delay", "ts_mean"],
            "returns": ["delay", "ts_mean"],
            "correlation": ["covariance"],
            "covariance": ["correlation"],
            "rank": ["zscore", "sign"],
            "zscore": ["rank", "sign"],
        }

    def generate(
        self,
        parent: MemoryParent,
        snapshot: PatternMemorySnapshot,
        target_count: int,
        force_novelty: bool = False,
    ) -> list[tuple[str, dict]]:
        hints = list(parent.mutation_hints)
        if force_novelty and "diversify_feature_family" not in hints:
            hints.append("diversify_feature_family")

        candidates: list[tuple[str, dict]] = []
        strategies = self._strategies_from_hints(hints)
        attempts = 0
        max_attempts = max(target_count * 10, 20)
        while len(candidates) < target_count and strategies and attempts < max_attempts:
            attempts += 1
            strategy = self.randomizer.choice(strategies)
            expression = strategy(parent.expression, snapshot)
            if expression and expression != parent.expression:
                candidates.append(
                    (
                        expression,
                        {
                            "parent_refs": [{"run_id": parent.run_id, "alpha_id": parent.alpha_id}],
                            "source_pattern_ids": [],
                            "source_gene_ids": [],
                            "mutation_hint_tags": list(hints),
                            "target_novelty": force_novelty,
                        },
                    )
                )
        return candidates

    def _strategies_from_hints(self, hints: Iterable[str]) -> list:
        strategies = [self._swap_operator_family, self._swap_wrapper]
        if "smoothen_and_slow_down" in hints or "lengthen_lookbacks" in hints:
            strategies.extend([self._lengthen_lookbacks, self._smoothen_expression])
        if "simplify_and_stabilize" in hints or "reduce_complexity" in hints:
            strategies.extend([self._simplify_expression, self._lengthen_lookbacks])
        if "diversify_feature_family" in hints:
            strategies.extend([self._swap_field_family, self._novel_wrapper])
        if "favor_robust_windows" in hints or "prefer_stable_genes" in hints:
            strategies.extend([self._lengthen_lookbacks, self._stable_wrapper])
        return strategies

    def _lengthen_lookbacks(self, expression: str, _: PatternMemorySnapshot) -> str | None:
        larger = sorted(self.config.lookbacks)[-max(1, min(2, len(self.config.lookbacks))):]
        if not larger:
            return None
        return _swap_lookback(expression, larger, self.randomizer)

    def _smoothen_expression(self, expression: str, _: PatternMemorySnapshot) -> str:
        window = max(self.config.lookbacks)
        wrapper = self.randomizer.choice(["ts_mean", "decay_linear"])
        return f"{wrapper}(({expression}), {window})"

    def _simplify_expression(self, expression: str, _: PatternMemorySnapshot) -> str | None:
        try:
            node = parse_expression(expression)
        except ValueError:
            return None
        if isinstance(node, FunctionCallNode) and len(node.args) == 1:
            return to_expression(node.args[0])
        if isinstance(node, UnaryOpNode):
            return to_expression(node.operand)
        if isinstance(node, BinaryOpNode):
            return to_expression(self.randomizer.choice([node.left, node.right]))
        return None

    def _swap_field_family(self, expression: str, snapshot: PatternMemorySnapshot) -> str | None:
        fields = self._preferred_fields(snapshot, mode="novel")
        if not fields:
            fields = [field for field in self.config.allowed_fields if field]
        return _swap_field(expression, fields, self.randomizer)

    def _swap_operator_family(self, expression: str, _: PatternMemorySnapshot) -> str | None:
        return _swap_operator(expression, self.operator_swaps, self.randomizer)

    def _swap_wrapper(self, expression: str, _: PatternMemorySnapshot) -> str:
        wrapper = self.randomizer.choice(self.config.normalization_wrappers)
        return f"{wrapper}({expression})"

    def _stable_wrapper(self, expression: str, snapshot: PatternMemorySnapshot) -> str:
        wrappers = self._preferred_wrappers(snapshot, mode="stable")
        wrapper = self.randomizer.choice(wrappers or self.config.normalization_wrappers)
        return f"{wrapper}({expression})"

    def _novel_wrapper(self, expression: str, snapshot: PatternMemorySnapshot) -> str:
        wrappers = self._preferred_wrappers(snapshot, mode="novel")
        wrapper = self.randomizer.choice(wrappers or self.config.normalization_wrappers)
        return f"{wrapper}({expression})"

    def _preferred_fields(self, snapshot: PatternMemorySnapshot, mode: str) -> list[str]:
        field_patterns = snapshot.by_kind("field")
        if mode == "novel":
            ranked = sorted(
                field_patterns,
                key=lambda item: (item.avg_behavioral_novelty, -item.support, item.pattern_value),
                reverse=True,
            )
        else:
            ranked = sorted(field_patterns, key=lambda item: (item.pattern_score, item.support), reverse=True)
        return [item.pattern_value for item in ranked[: max(3, len(self.config.allowed_fields))]]

    def _preferred_wrappers(self, snapshot: PatternMemorySnapshot, mode: str) -> list[str]:
        wrapper_patterns = snapshot.by_kind("wrapper")
        if mode == "novel":
            ranked = sorted(
                wrapper_patterns,
                key=lambda item: (item.avg_behavioral_novelty, -item.support, item.pattern_value),
                reverse=True,
            )
        else:
            ranked = sorted(wrapper_patterns, key=lambda item: (item.pattern_score, item.support), reverse=True)
        return [item.pattern_value for item in ranked[: max(3, len(self.config.normalization_wrappers))]]
