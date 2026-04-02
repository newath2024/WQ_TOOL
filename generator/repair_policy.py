from __future__ import annotations

from dataclasses import replace

from alpha.parser import parse_expression
from alpha.validator import has_nesting_violation
from core.config import GenerationConfig, RepairPolicyConfig
from data.field_registry import FieldRegistry
from features.registry import OperatorRegistry
from generator.genome import Genome
from generator.grammar import MotifGrammar

_CROSS_SECTIONAL_WRAPPERS = frozenset(
    {"rank", "zscore", "group_rank", "group_zscore", "group_neutralize", "normalize"}
)


class RepairPolicy:
    def __init__(
        self,
        *,
        generation_config: GenerationConfig,
        repair_config: RepairPolicyConfig,
        field_registry: FieldRegistry,
        registry: OperatorRegistry,
    ) -> None:
        self.generation_config = generation_config
        self.repair_config = repair_config
        self.field_registry = field_registry
        self.registry = registry
        self._grammar = MotifGrammar()

    def repair(
        self,
        genome: Genome,
        *,
        fail_tags: tuple[str, ...] = (),
    ) -> tuple[Genome, tuple[str, ...]]:
        if not self.repair_config.enabled:
            return genome, ()
        current = genome
        actions: list[str] = []
        attempts = 0
        while attempts < self.repair_config.max_attempts:
            attempts += 1
            changed = False
            deduped_pre = self._dedupe_wrappers(current.wrapper_gene.pre_wrappers)
            deduped_post = self._dedupe_wrappers(current.wrapper_gene.post_wrappers)
            safe_pre = self._drop_unsafe_wrappers(deduped_pre)
            safe_post = self._drop_unsafe_wrappers(deduped_post)
            if safe_pre != current.wrapper_gene.pre_wrappers or safe_post != current.wrapper_gene.post_wrappers:
                current = replace(current, wrapper_gene=replace(current.wrapper_gene, pre_wrappers=safe_pre, post_wrappers=safe_post))
                actions.append("cleanup_wrappers")
                changed = True

            if current.motif == "group_relative_signal" and not current.feature_gene.group_field and self.repair_config.allow_group_fixups:
                current = replace(current, feature_gene=replace(current.feature_gene, group_field="sector"))
                actions.append("add_group_field")
                changed = True
            if current.motif == "liquidity_conditioned_signal" and not current.feature_gene.liquidity_field:
                fallback = current.feature_gene.auxiliary_field or current.feature_gene.secondary_field or current.feature_gene.primary_field
                current = replace(current, feature_gene=replace(current.feature_gene, liquidity_field=fallback))
                actions.append("add_liquidity_field")
                changed = True

            if ("high_turnover" in fail_tags or current.turnover_bucket in {"active", "very_active"}) and self.repair_config.allow_turnover_reduction:
                larger_window = max(self.generation_config.lookbacks)
                smoothing_operator = self._best_smoothing_operator()
                current = replace(
                    current,
                    horizon_gene=replace(
                        current.horizon_gene,
                        slow_window=max(current.horizon_gene.slow_window or 0, larger_window),
                        context_window=max(current.horizon_gene.context_window or 0, larger_window),
                    ),
                    turnover_gene=replace(
                        current.turnover_gene,
                        smoothing_operator=current.turnover_gene.smoothing_operator or smoothing_operator,
                        smoothing_window=max(current.turnover_gene.smoothing_window, larger_window),
                        turnover_hint=min(current.turnover_gene.turnover_hint, -0.20),
                    ),
                )
                actions.append("reduce_turnover")
                changed = True

            if ("excessive_complexity" in fail_tags or current.complexity_bucket == "complex") and self.repair_config.allow_complexity_reduction:
                current = replace(
                    current,
                    wrapper_gene=replace(current.wrapper_gene, post_wrappers=current.wrapper_gene.post_wrappers[:1], pre_wrappers=current.wrapper_gene.pre_wrappers[:1]),
                    complexity_gene=replace(
                        current.complexity_gene,
                        target_depth=max(2, current.complexity_gene.target_depth - 1),
                        binary_branching=1,
                        wrapper_budget=1,
                    ),
                    transform_gene=replace(current.transform_gene, secondary_transform="", pair_operator=current.transform_gene.pair_operator),
                )
                actions.append("reduce_complexity")
                changed = True

            if "brain_rejected" in fail_tags and self.repair_config.allow_complexity_reduction:
                current = replace(
                    current,
                    transform_gene=replace(current.transform_gene, motif="momentum", secondary_transform=""),
                    regime_gene=replace(current.regime_gene, conditioning_mode="none", conditioning_field=""),
                )
                actions.append("simplify_after_rejection")
                changed = True

            if not changed:
                break

        # Post-repair nesting safety: render and check for violations.
        current, nesting_action = self._repair_nesting_violations(current)
        if nesting_action:
            actions.append(nesting_action)

        return current, tuple(actions)

    def _dedupe_wrappers(self, wrappers: tuple[str, ...]) -> tuple[str, ...]:
        seen: list[str] = []
        for wrapper in wrappers:
            if wrapper and wrapper not in seen:
                seen.append(wrapper)
        return tuple(seen[: max(1, self.repair_config.max_attempts)])

    def _drop_unsafe_wrappers(self, wrappers: tuple[str, ...]) -> tuple[str, ...]:
        safe = tuple(
            wrapper
            for wrapper in wrappers
            if not (self.registry.contains(wrapper) and self.registry.get(wrapper).has_tag("requires_positive_input"))
        )
        return safe

    def _best_smoothing_operator(self) -> str:
        candidates: list[tuple[float, str]] = []
        for name, spec in self.registry.items():
            if spec.min_args > 2:
                continue
            score = 0.0
            if spec.has_tag("smoothing"):
                score += 2.0
            if spec.has_tag("reduces_turnover"):
                score += 2.5
            if spec.turnover_hint < 0:
                score += abs(spec.turnover_hint)
            if spec.prefers_motif("mean_reversion") or spec.prefers_motif("residualized_signal"):
                score += 0.25
            if score > 0:
                candidates.append((score, name))
        if not candidates:
            return "ts_mean"
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _repair_nesting_violations(self, genome: Genome) -> tuple[Genome, str | None]:
        """Render the genome and strip cross-sectional wrappers when needed."""
        try:
            render = self._grammar.render(genome)
            node = parse_expression(render.expression)
        except (ValueError, TypeError):
            return genome, None

        if not has_nesting_violation(node):
            return genome, None

        clean_pre = tuple(w for w in genome.wrapper_gene.pre_wrappers if w not in _CROSS_SECTIONAL_WRAPPERS)
        clean_post = tuple(w for w in genome.wrapper_gene.post_wrappers if w not in _CROSS_SECTIONAL_WRAPPERS)
        repaired = replace(genome, wrapper_gene=replace(genome.wrapper_gene, pre_wrappers=clean_pre, post_wrappers=clean_post))

        try:
            render2 = self._grammar.render(repaired)
            node2 = parse_expression(render2.expression)
        except (ValueError, TypeError):
            return genome, "nesting_repair_failed"

        if not has_nesting_violation(node2):
            return repaired, "nesting_repair"

        return genome, "nesting_repair_failed"
