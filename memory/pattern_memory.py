from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Iterable

from alpha.ast_nodes import (
    BinaryOpNode,
    ExprNode,
    FunctionCallNode,
    IdentifierNode,
    NumberNode,
    UnaryOpNode,
    iter_child_nodes,
    node_complexity,
    node_depth,
    to_expression,
)
from alpha.parser import parse_expression
from core.config import AppConfig
from core.signatures import build_simulation_signature
from features.registry import WINDOWED_OPERATORS
from features.registry import build_default_registry


FAIL_TAG_PENALTIES: dict[str, float] = {
    "high_turnover": 0.20,
    "overfit_train_validation_gap": 0.25,
    "low_stability": 0.20,
    "high_correlation_with_existing": 0.15,
    "excessive_complexity": 0.10,
    "noisy_short_horizon": 0.10,
    "weak_validation": 0.20,
}


@dataclass(frozen=True, slots=True)
class GeneObservation:
    pattern_id: str
    pattern_kind: str
    pattern_value: str


@dataclass(frozen=True, slots=True)
class StructuralSignature:
    operators: tuple[str, ...]
    operator_families: tuple[str, ...]
    operator_path: tuple[str, ...]
    fields: tuple[str, ...]
    field_families: tuple[str, ...]
    lookbacks: tuple[int, ...]
    wrappers: tuple[str, ...]
    depth: int
    complexity: int
    complexity_bucket: str
    horizon_bucket: str
    turnover_bucket: str
    motif: str
    family_signature: str
    subexpressions: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PatternScore:
    pattern_id: str
    pattern_kind: str
    pattern_value: str
    support: int
    success_count: int
    failure_count: int
    avg_outcome: float
    avg_behavioral_novelty: float
    fail_tag_counts: dict[str, int]
    pattern_score: float


@dataclass(frozen=True, slots=True)
class MemoryParent:
    run_id: str
    alpha_id: str
    expression: str
    normalized_expression: str
    generation_mode: str
    generation_metadata: dict
    parent_refs: tuple[dict[str, str], ...]
    family_signature: str
    outcome_score: float
    behavioral_novelty_score: float
    fail_tags: tuple[str, ...]
    success_tags: tuple[str, ...]
    mutation_hints: tuple[str, ...]
    structural_signature: StructuralSignature | None = None


@dataclass(frozen=True, slots=True)
class PatternMemorySnapshot:
    regime_key: str
    patterns: dict[str, PatternScore] = field(default_factory=dict)
    top_parents: tuple[MemoryParent, ...] = ()
    fail_tag_counts: dict[str, int] = field(default_factory=dict)

    def by_kind(self, pattern_kind: str) -> list[PatternScore]:
        return [pattern for pattern in self.patterns.values() if pattern.pattern_kind == pattern_kind]


class PatternMemoryService:
    def __init__(self) -> None:
        self.registry = build_default_registry()

    def build_regime_key(self, dataset_fingerprint: str, config: AppConfig) -> str:
        return build_simulation_signature(
            {
                "dataset_fingerprint": dataset_fingerprint,
                "timeframe": config.backtest.timeframe,
                "simulation": asdict(config.simulation),
                "backtest": asdict(config.backtest),
                "allowed_fields": config.generation.allowed_fields,
                "allowed_operators": config.generation.allowed_operators,
            }
        )

    def extract_signature(
        self,
        expression: str,
        *,
        generation_metadata: dict | None = None,
        field_categories: dict[str, str] | None = None,
    ) -> StructuralSignature:
        node = parse_expression(expression)
        operator_path = tuple(self._collect_operators(node))
        operators = tuple(sorted(set(operator_path)))
        operator_families = tuple(sorted({self._operator_family(operator) for operator in operators}))
        fields = tuple(sorted(set(self._collect_fields(node))))
        field_families = self._resolve_field_families(
            fields,
            generation_metadata=generation_metadata,
            field_categories=field_categories,
        )
        lookbacks = tuple(sorted(set(self._collect_lookbacks(node))))
        wrappers = tuple(self._collect_wrappers(node))
        depth = node_depth(node)
        complexity = node_complexity(node)
        complexity_bucket = self._complexity_bucket(complexity)
        horizon_bucket = self._horizon_bucket(lookbacks)
        motif = str((generation_metadata or {}).get("motif") or (generation_metadata or {}).get("template_name") or "")
        turnover_bucket = str((generation_metadata or {}).get("turnover_bucket") or self._estimate_turnover_bucket(operators))
        subexpressions = tuple(sorted(set(self._collect_subexpressions(node))))
        family_signature = self._build_family_signature(
            operators,
            fields,
            lookbacks,
            wrappers,
            depth,
            field_families=field_families,
            motif=motif,
            horizon_bucket=horizon_bucket,
            turnover_bucket=turnover_bucket,
            complexity_bucket=complexity_bucket,
        )
        return StructuralSignature(
            operators=operators,
            operator_families=operator_families,
            operator_path=operator_path,
            fields=fields,
            field_families=field_families,
            lookbacks=lookbacks,
            wrappers=wrappers,
            depth=depth,
            complexity=complexity,
            complexity_bucket=complexity_bucket,
            horizon_bucket=horizon_bucket,
            turnover_bucket=turnover_bucket,
            motif=motif,
            family_signature=family_signature,
            subexpressions=subexpressions,
        )

    def build_observations(
        self,
        signature: StructuralSignature,
        *,
        template_name: str | None = None,
        rejection_reasons: Iterable[str] | None = None,
        generation_metadata: dict | None = None,
        success_tags: Iterable[str] | None = None,
        fail_tags: Iterable[str] | None = None,
    ) -> list[GeneObservation]:
        observations: list[GeneObservation] = []
        observations.append(self._make_observation("family", signature.family_signature))
        if signature.motif:
            observations.append(self._make_observation("motif", signature.motif))
        for operator in signature.operators:
            observations.append(self._make_observation("operator", operator))
        for operator_family in signature.operator_families:
            observations.append(self._make_observation("operator_family", operator_family))
        if signature.operator_path:
            observations.append(self._make_observation("operator_path", ">".join(signature.operator_path)))
        for field in signature.fields:
            observations.append(self._make_observation("field", field))
        for field_family in signature.field_families:
            observations.append(self._make_observation("field_family", field_family))
        for lookback in signature.lookbacks:
            observations.append(self._make_observation("lookback", str(lookback)))
        for wrapper in signature.wrappers:
            observations.append(self._make_observation("wrapper", wrapper))
        if signature.complexity_bucket:
            observations.append(self._make_observation("complexity_bucket", signature.complexity_bucket))
        if signature.horizon_bucket:
            observations.append(self._make_observation("horizon_bucket", signature.horizon_bucket))
        if signature.turnover_bucket:
            observations.append(self._make_observation("turnover_bucket", signature.turnover_bucket))
        for subexpression in signature.subexpressions:
            observations.append(self._make_observation("subexpression", subexpression))
        resolved_template = str((generation_metadata or {}).get("motif") or template_name or "")
        if resolved_template:
            observations.append(self._make_observation("template", resolved_template))
        mutation_mode = str((generation_metadata or {}).get("mutation_mode") or "")
        if mutation_mode:
            observations.append(self._make_observation("mutation_mode", mutation_mode))
        for tag in (generation_metadata or {}).get("operator_semantic_tags", []) or []:
            observations.append(self._make_observation("operator_semantic_tag", str(tag)))
        for reason in rejection_reasons or []:
            observations.append(self._make_observation("rejection_reason", reason))
        for tag in success_tags or []:
            observations.append(self._make_observation("success_tag", tag))
        for tag in fail_tags or []:
            observations.append(self._make_observation("fail_tag", tag))
        deduped = {(item.pattern_kind, item.pattern_value): item for item in observations}
        return list(deduped.values())

    def compute_outcome_score(
        self,
        validation_fitness: float,
        passed_filters: bool,
        selected_top_alpha: bool,
        behavioral_novelty_score: float,
        fail_tags: Iterable[str],
    ) -> float:
        score = math.tanh(validation_fitness / 3.0)
        if passed_filters:
            score += 0.25
        if selected_top_alpha:
            score += 0.25
        score += 0.10 * behavioral_novelty_score
        for tag in fail_tags:
            score -= FAIL_TAG_PENALTIES.get(tag, 0.0)
        return float(score)

    def compute_brain_outcome_score(
        self,
        *,
        metrics: dict[str, float | None],
        submission_eligible: bool | None,
        rejection_reason: str | None,
        fail_tags: Iterable[str],
    ) -> float:
        score = 0.0
        sharpe = metrics.get("sharpe")
        fitness = metrics.get("fitness")
        turnover = metrics.get("turnover")
        margin = metrics.get("margin")
        returns = metrics.get("returns")

        if sharpe is not None:
            score += 0.35 * math.tanh(sharpe / 3.0)
        if fitness is not None:
            score += 0.35 * math.tanh(fitness / 3.0)
        if returns is not None:
            score += 0.15 * math.tanh(returns / 0.15)
        if margin is not None:
            score += 0.10 * math.tanh(margin / 0.10)
        if turnover is not None:
            if turnover <= 0.7:
                score += 0.10
            elif turnover >= 1.2:
                score -= 0.10
        if submission_eligible is True:
            score += 0.20
        elif submission_eligible is False:
            score -= 0.05
        if rejection_reason:
            score -= 0.25
        for tag in fail_tags:
            score -= FAIL_TAG_PENALTIES.get(tag, 0.0)
        return float(score)

    def score_expression(
        self,
        expression: str,
        snapshot: PatternMemorySnapshot,
        min_pattern_support: int,
        *,
        generation_metadata: dict | None = None,
        field_categories: dict[str, str] | None = None,
    ) -> tuple[float, float, StructuralSignature, list[GeneObservation]]:
        signature = self.extract_signature(
            expression,
            generation_metadata=generation_metadata,
            field_categories=field_categories,
        )
        observations = self.build_observations(signature, generation_metadata=generation_metadata)
        scores: list[float] = []
        novelty_scores: list[float] = []
        for observation in observations:
            pattern = snapshot.patterns.get(observation.pattern_id)
            if pattern is None or pattern.support < min_pattern_support:
                continue
            scores.append(pattern.pattern_score)
            novelty_scores.append(pattern.avg_behavioral_novelty)
        base_score = float(sum(scores) / len(scores)) if scores else 0.0
        novelty_score = float(sum(novelty_scores) / len(novelty_scores)) if novelty_scores else 0.5
        return base_score, novelty_score, signature, observations

    def behavioral_novelty(
        self,
        signal_correlation: float | None,
        returns_correlation: float | None,
    ) -> float:
        max_correlation = max(abs(signal_correlation or 0.0), abs(returns_correlation or 0.0))
        return float(max(0.0, min(1.0, 1.0 - max_correlation)))

    def _make_observation(self, pattern_kind: str, pattern_value: str) -> GeneObservation:
        payload = f"{pattern_kind}:{pattern_value}"
        return GeneObservation(
            pattern_id=hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16],
            pattern_kind=pattern_kind,
            pattern_value=pattern_value,
        )

    def _collect_operators(self, node: ExprNode) -> list[str]:
        operators: list[str] = []
        if isinstance(node, FunctionCallNode):
            operators.append(node.name)
        elif isinstance(node, BinaryOpNode):
            operators.append(f"binary:{node.operator}")
        elif isinstance(node, UnaryOpNode):
            operators.append(f"unary:{node.operator}")
        for child in iter_child_nodes(node):
            operators.extend(self._collect_operators(child))
        return operators

    def _collect_fields(self, node: ExprNode) -> list[str]:
        fields: list[str] = []
        if isinstance(node, IdentifierNode):
            fields.append(node.name)
        for child in iter_child_nodes(node):
            fields.extend(self._collect_fields(child))
        return fields

    def _collect_lookbacks(self, node: ExprNode) -> list[int]:
        lookbacks: list[int] = []
        if isinstance(node, FunctionCallNode) and node.name in WINDOWED_OPERATORS and node.args:
            maybe_window = node.args[-1]
            if isinstance(maybe_window, NumberNode) and float(maybe_window.value).is_integer() and maybe_window.value > 0:
                lookbacks.append(int(maybe_window.value))
        for child in iter_child_nodes(node):
            lookbacks.extend(self._collect_lookbacks(child))
        return lookbacks

    def _collect_wrappers(self, node: ExprNode) -> list[str]:
        wrappers: list[str] = []
        current = node
        while isinstance(current, FunctionCallNode) and len(current.args) == 1:
            wrappers.append(current.name)
            current = current.args[0]
        return wrappers

    def _collect_subexpressions(self, node: ExprNode) -> list[str]:
        subexpressions: list[str] = []
        complexity = node_complexity(node)
        depth = node_depth(node)
        if 2 <= complexity <= 6 and depth <= 3:
            subexpressions.append(to_expression(node))
        for child in iter_child_nodes(node):
            subexpressions.extend(self._collect_subexpressions(child))
        return subexpressions

    def _build_family_signature(
        self,
        operators: tuple[str, ...],
        fields: tuple[str, ...],
        lookbacks: tuple[int, ...],
        wrappers: tuple[str, ...],
        depth: int,
        *,
        field_families: tuple[str, ...],
        motif: str,
        horizon_bucket: str,
        turnover_bucket: str,
        complexity_bucket: str,
    ) -> str:
        payload = {
            "operators": operators,
            "fields": fields,
            "field_families": field_families,
            "lookbacks": lookbacks,
            "wrappers": wrappers[:3],
            "depth_bucket": min(depth, 6),
            "motif": motif,
            "horizon_bucket": horizon_bucket,
            "turnover_bucket": turnover_bucket,
            "complexity_bucket": complexity_bucket,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _operator_family(self, operator_name: str) -> str:
        if operator_name.startswith("binary:") or operator_name.startswith("unary:"):
            return operator_name.split(":", 1)[0]
        if self.registry.contains(operator_name):
            return self.registry.family_for(operator_name)
        return "other"

    def _resolve_field_families(
        self,
        fields: tuple[str, ...],
        *,
        generation_metadata: dict | None,
        field_categories: dict[str, str] | None,
    ) -> tuple[str, ...]:
        metadata = generation_metadata or {}
        payload = metadata.get("field_families")
        if isinstance(payload, list) and payload:
            return tuple(sorted({str(item) for item in payload if str(item)}))
        categories = field_categories or {}
        resolved = {
            str(categories.get(field) or ("group" if field in {"sector", "industry", "country", "subindustry"} else "other"))
            for field in fields
        }
        return tuple(sorted(item for item in resolved if item))

    def _complexity_bucket(self, complexity: int) -> str:
        if complexity <= 5:
            return "simple"
        if complexity <= 10:
            return "moderate"
        if complexity <= 16:
            return "layered"
        return "complex"

    def _horizon_bucket(self, lookbacks: tuple[int, ...]) -> str:
        if not lookbacks:
            return "unknown"
        max_window = max(lookbacks)
        if max_window <= 3:
            return "very_short"
        if max_window <= 10:
            return "short"
        if max_window <= 20:
            return "medium"
        return "long"

    def _estimate_turnover_bucket(self, operators: tuple[str, ...]) -> str:
        hints = [
            self.registry.get(operator).turnover_hint
            for operator in operators
            if self.registry.contains(operator)
        ]
        if not hints:
            return "balanced"
        average_hint = sum(hints) / len(hints)
        if average_hint <= -0.15:
            return "low"
        if average_hint <= 0.15:
            return "balanced"
        if average_hint <= 0.50:
            return "active"
        return "very_active"
