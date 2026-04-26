from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpha.ast_nodes import (
    BinaryOpNode,
    ExprNode,
    FunctionCallNode,
    IdentifierNode,
    NumberNode,
    UnaryOpNode,
    to_expression,
)
from alpha.parser import parse_expression
from core.config import EliteMotifConfig
from domain.candidate import AlphaCandidate
from features.registry import NORMALIZATION_OPERATORS, OperatorRegistry, WINDOWED_OPERATORS


@dataclass(frozen=True, slots=True)
class EliteSeedVariant:
    seed_id: str
    expression: str
    variant: str
    similarity: float
    match_score: float


@dataclass(frozen=True, slots=True)
class _SeedProfile:
    seed_id: str
    expression: str
    normalized_expression: str
    tokens: frozenset[str]
    subexpressions: frozenset[str]


def annotate_elite_motif_candidates(
    candidates: list[AlphaCandidate],
    config: EliteMotifConfig,
) -> dict[str, Any]:
    if not bool(config.enabled) or not candidates:
        return {
            "elite_motif_annotation_enabled": bool(config.enabled),
            "elite_motif_annotated_count": 0,
            "elite_motif_penalty_count": 0,
        }
    profiles = _seed_profiles(config)
    if not profiles:
        return {
            "elite_motif_annotation_enabled": bool(config.enabled),
            "elite_motif_annotated_count": 0,
            "elite_motif_penalty_count": 0,
        }

    annotated = 0
    penalty_count = 0
    for candidate in candidates:
        match_score, motif_ids = elite_motif_match(
            candidate.expression,
            profiles=profiles,
        )
        similarity, seed_id = elite_seed_similarity(
            candidate.expression,
            profiles=profiles,
        )
        penalty = _similarity_penalty(
            similarity,
            threshold=float(config.clone_similarity_threshold),
        )
        metadata = candidate.generation_metadata
        existing_match = _to_float(metadata.get("elite_motif_match_score"))
        if match_score > existing_match:
            metadata["elite_motif_match_score"] = round(float(match_score), 6)
            metadata["elite_motif_ids"] = list(motif_ids)
        else:
            metadata.setdefault("elite_motif_match_score", round(float(existing_match), 6))
            metadata.setdefault("elite_motif_ids", [])
        existing_similarity = _to_float(metadata.get("elite_seed_similarity"))
        if similarity > existing_similarity:
            metadata["elite_seed_similarity"] = round(float(similarity), 6)
            metadata["elite_seed_similarity_source"] = seed_id
        else:
            metadata.setdefault("elite_seed_similarity", round(float(existing_similarity), 6))
        existing_penalty = _to_float(metadata.get("elite_seed_similarity_penalty"))
        metadata["elite_seed_similarity_penalty"] = round(max(existing_penalty, penalty), 6)
        if match_score > 0.0:
            annotated += 1
        if penalty > 0.0:
            penalty_count += 1
    return {
        "elite_motif_annotation_enabled": True,
        "elite_motif_annotated_count": int(annotated),
        "elite_motif_penalty_count": int(penalty_count),
    }


def build_elite_seed_variants(
    config: EliteMotifConfig,
    *,
    registry: OperatorRegistry,
    existing_normalized: set[str] | None = None,
) -> list[EliteSeedVariant]:
    if (
        not bool(config.enabled)
        or int(config.max_quality_polish_seeds_per_round) <= 0
        or int(config.max_seed_variants_per_seed) <= 0
    ):
        return []
    existing = set(existing_normalized or set())
    profiles = _seed_profiles(config)
    variants: list[EliteSeedVariant] = []
    seen: set[str] = set(existing)
    for profile in profiles[: int(config.max_quality_polish_seeds_per_round)]:
        try:
            root = parse_expression(profile.expression)
        except ValueError:
            continue
        per_seed = 0
        for expression, variant_name in _variant_expressions_for_seed(
            root,
            lookbacks=config.lookbacks,
            registry=registry,
        ):
            normalized = normalize_expression(expression)
            if (
                not normalized
                or normalized == profile.normalized_expression
                or normalized in seen
                or _is_redundant_normalization(expression)
            ):
                continue
            seen.add(normalized)
            similarity = expression_similarity(normalized, profile.normalized_expression)
            match_score, _ = elite_motif_match(normalized, profiles=(profile,))
            variants.append(
                EliteSeedVariant(
                    seed_id=profile.seed_id,
                    expression=expression,
                    variant=variant_name,
                    similarity=float(similarity),
                    match_score=float(match_score),
                )
            )
            per_seed += 1
            if per_seed >= int(config.max_seed_variants_per_seed):
                break
    return variants


def elite_motif_match(
    expression: str,
    *,
    profiles: tuple[_SeedProfile, ...],
) -> tuple[float, tuple[str, ...]]:
    candidate_tokens = expression_tokens(expression)
    if not candidate_tokens:
        return 0.0, ()
    scored: list[tuple[float, str]] = []
    for profile in profiles:
        score = _jaccard(candidate_tokens, profile.tokens)
        if score > 0.0:
            scored.append((score, profile.seed_id))
    if not scored:
        return 0.0, ()
    scored.sort(key=lambda item: (-item[0], item[1]))
    best = scored[0][0]
    motif_ids = tuple(seed_id for score, seed_id in scored if score >= max(0.01, best * 0.85))[:3]
    return float(best), motif_ids


def elite_seed_similarity(
    expression: str,
    *,
    profiles: tuple[_SeedProfile, ...],
) -> tuple[float, str]:
    normalized = normalize_expression(expression)
    if not normalized:
        return 0.0, ""
    candidate_tokens = expression_tokens(normalized)
    candidate_subexpressions = expression_subexpressions(normalized)
    best: tuple[float, str] = (0.0, "")
    for profile in profiles:
        if normalized == profile.normalized_expression:
            score = 1.0
        else:
            token_score = _jaccard(candidate_tokens, profile.tokens)
            subtree_score = _jaccard(candidate_subexpressions, profile.subexpressions)
            score = 0.65 * token_score + 0.35 * subtree_score
        if score > best[0]:
            best = (float(score), profile.seed_id)
    return best


def expression_similarity(left: str, right: str) -> float:
    left_normalized = normalize_expression(left)
    right_normalized = normalize_expression(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    token_score = _jaccard(expression_tokens(left_normalized), expression_tokens(right_normalized))
    subtree_score = _jaccard(expression_subexpressions(left_normalized), expression_subexpressions(right_normalized))
    return float(0.65 * token_score + 0.35 * subtree_score)


def expression_tokens(expression: str) -> frozenset[str]:
    try:
        root = parse_expression(expression)
    except ValueError:
        return frozenset()
    tokens: set[str] = set()
    _collect_tokens(root, tokens)
    return frozenset(tokens)


def expression_subexpressions(expression: str) -> frozenset[str]:
    try:
        root = parse_expression(expression)
    except ValueError:
        return frozenset()
    subexpressions: set[str] = set()
    _collect_subexpressions(root, subexpressions)
    return frozenset(subexpressions)


def normalize_expression(expression: str) -> str:
    try:
        return to_expression(parse_expression(expression))
    except ValueError:
        return str(expression or "").strip()


def _seed_profiles(config: EliteMotifConfig) -> tuple[_SeedProfile, ...]:
    profiles: list[_SeedProfile] = []
    for index, expression in enumerate(config.seed_expressions or [], start=1):
        normalized = normalize_expression(expression)
        if not normalized:
            continue
        profiles.append(
            _SeedProfile(
                seed_id=f"elite_seed_{index}",
                expression=expression,
                normalized_expression=normalized,
                tokens=expression_tokens(normalized),
                subexpressions=expression_subexpressions(normalized),
            )
        )
    return tuple(profiles)


def _variant_expressions_for_seed(
    root: ExprNode,
    *,
    lookbacks: list[int],
    registry: OperatorRegistry,
) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    root_expr = to_expression(root)

    for wrapper in ("rank", "zscore", "quantile"):
        if not registry.contains(wrapper):
            continue
        if _is_root_call(root, wrapper):
            continue
        variants.append((f"{wrapper}({root_expr})", f"wrap_{wrapper}"))

    if isinstance(root, FunctionCallNode) and root.name in {"rank", "zscore", "quantile"} and len(root.args) == 1:
        base_expr = to_expression(root.args[0])
        for wrapper in ("rank", "zscore", "quantile"):
            if wrapper != root.name and registry.contains(wrapper):
                variants.append((f"{wrapper}({base_expr})", f"swap_wrapper_{wrapper}"))
        variants.append((base_expr, "unwrap_root"))

    for term_index, term in enumerate(_top_level_terms(root), start=1):
        term_expr = to_expression(term)
        if term_expr != root_expr:
            for wrapper in ("rank", "zscore", "quantile"):
                if registry.contains(wrapper):
                    variants.append((f"{wrapper}({term_expr})", f"drop_complex_terms_{term_index}_{wrapper}"))

    for sub_index, subterm in enumerate(_candidate_subterms(root), start=1):
        sub_expr = to_expression(_unwrap_normalization(subterm))
        for wrapper in ("rank", "zscore", "quantile"):
            if registry.contains(wrapper):
                variants.append((f"{wrapper}({sub_expr})", f"unwrap_subterm_{sub_index}_{wrapper}"))

    for window_index, expression in enumerate(_window_swap_expressions(root, lookbacks=lookbacks), start=1):
        variants.append((expression, f"window_swap_{window_index}"))

    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for expression, variant in variants:
        normalized = normalize_expression(expression)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append((expression, variant))
    return unique


def _window_swap_expressions(root: ExprNode, *, lookbacks: list[int]) -> list[str]:
    expressions: list[str] = []
    for node in _replace_window_nodes(root, lookbacks=lookbacks):
        expressions.append(to_expression(node))
    return expressions


def _replace_window_nodes(node: ExprNode, *, lookbacks: list[int]) -> list[ExprNode]:
    replacements: list[ExprNode] = []
    if isinstance(node, FunctionCallNode):
        args = list(node.args)
        if node.name in WINDOWED_OPERATORS and len(args) >= 2 and isinstance(args[-1], NumberNode):
            current = int(args[-1].value)
            for window in lookbacks:
                if int(window) == current:
                    continue
                updated_args = list(args)
                updated_args[-1] = NumberNode(float(window))
                replacements.append(FunctionCallNode(name=node.name, args=tuple(updated_args)))
        for arg_index, arg in enumerate(args):
            for child_replacement in _replace_window_nodes(arg, lookbacks=lookbacks):
                updated_args = list(args)
                updated_args[arg_index] = child_replacement
                replacements.append(FunctionCallNode(name=node.name, args=tuple(updated_args)))
    if isinstance(node, BinaryOpNode):
        for child_replacement in _replace_window_nodes(node.left, lookbacks=lookbacks):
            replacements.append(BinaryOpNode(operator=node.operator, left=child_replacement, right=node.right))
        for child_replacement in _replace_window_nodes(node.right, lookbacks=lookbacks):
            replacements.append(BinaryOpNode(operator=node.operator, left=node.left, right=child_replacement))
    if isinstance(node, UnaryOpNode):
        for child_replacement in _replace_window_nodes(node.operand, lookbacks=lookbacks):
            replacements.append(UnaryOpNode(operator=node.operator, operand=child_replacement))
    return replacements


def _candidate_subterms(root: ExprNode) -> list[ExprNode]:
    subterms: list[ExprNode] = []

    def visit(node: ExprNode) -> None:
        if node is not root and isinstance(node, (FunctionCallNode, BinaryOpNode)):
            subterms.append(node)
        for child in _children(node):
            visit(child)

    visit(root)
    return subterms


def _top_level_terms(node: ExprNode) -> list[ExprNode]:
    if isinstance(node, BinaryOpNode) and node.operator in {"+", "-"}:
        return [*_top_level_terms(node.left), *_top_level_terms(node.right)]
    return [node]


def _unwrap_normalization(node: ExprNode) -> ExprNode:
    current = node
    while (
        isinstance(current, FunctionCallNode)
        and current.name in NORMALIZATION_OPERATORS
        and len(current.args) == 1
    ):
        current = current.args[0]
    return current


def _collect_tokens(node: ExprNode, tokens: set[str]) -> None:
    if isinstance(node, IdentifierNode):
        tokens.add(f"field:{node.name}")
        return
    if isinstance(node, NumberNode):
        if float(node.value).is_integer() and int(node.value) > 0:
            tokens.add(f"window:{int(node.value)}")
        return
    if isinstance(node, FunctionCallNode):
        tokens.add(f"op:{node.name}")
    if isinstance(node, BinaryOpNode):
        tokens.add(f"op:{node.operator}")
    if isinstance(node, UnaryOpNode):
        tokens.add(f"op:unary_{node.operator}")
    for child in _children(node):
        _collect_tokens(child, tokens)


def _collect_subexpressions(node: ExprNode, subexpressions: set[str]) -> None:
    subexpressions.add(to_expression(node))
    for child in _children(node):
        _collect_subexpressions(child, subexpressions)


def _children(node: ExprNode) -> tuple[ExprNode, ...]:
    if isinstance(node, UnaryOpNode):
        return (node.operand,)
    if isinstance(node, BinaryOpNode):
        return (node.left, node.right)
    if isinstance(node, FunctionCallNode):
        return tuple(node.args)
    return ()


def _is_root_call(root: ExprNode, name: str) -> bool:
    return isinstance(root, FunctionCallNode) and root.name == name


def _is_redundant_normalization(expression: str) -> bool:
    try:
        root = parse_expression(expression)
    except ValueError:
        return False
    return (
        isinstance(root, FunctionCallNode)
        and root.name in NORMALIZATION_OPERATORS
        and len(root.args) == 1
        and isinstance(root.args[0], FunctionCallNode)
        and root.args[0].name in NORMALIZATION_OPERATORS
    )


def _similarity_penalty(similarity: float, *, threshold: float) -> float:
    if similarity <= threshold:
        return 0.0
    return float((similarity - threshold) / max(1e-9, 1.0 - threshold))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
