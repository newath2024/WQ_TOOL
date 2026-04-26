from __future__ import annotations

import random
from dataclasses import replace
from typing import TYPE_CHECKING, Sequence

from alpha.ast_nodes import BinaryOpNode, ExprNode, FunctionCallNode, IdentifierNode, NumberNode, UnaryOpNode, to_expression
from alpha.parser import parse_expression
from data.field_registry import FieldRegistry

if TYPE_CHECKING:
    from domain.candidate import AlphaCandidate


def mutate_expressions(
    parents: Sequence[AlphaCandidate],
    count: int,
    field_registry: FieldRegistry,
    lookbacks: list[int],
    normalization_wrappers: list[str],
    seed: int,
) -> list[tuple[str, tuple[str, ...], dict]]:
    randomizer = random.Random(seed)
    payload: list[tuple[str, tuple[str, ...], dict]] = []
    operator_swaps = {
        "ts_mean": ["ts_std_dev", "ts_decay_linear", "rolling_mean"],
        "ts_std_dev": ["ts_mean", "rolling_std"],
        "ts_delta": ["ts_delay", "ts_mean"],
        "ts_returns": ["ts_delay", "ts_mean"],
        "ts_corr": ["ts_covariance"],
        "ts_covariance": ["ts_corr"],
        "delta": ["ts_delay", "ts_mean"],
        "returns": ["ts_delay", "ts_mean"],
        "correlation": ["ts_covariance"],
        "covariance": ["ts_corr"],
        "decay_linear": ["ts_mean", "ts_std_dev"],
        "rank": ["zscore", "sign"],
        "zscore": ["rank", "sign"],
        "group_rank": ["group_zscore", "group_neutralize"],
        "group_zscore": ["group_rank", "group_neutralize"],
    }

    strategies = [
        ("lookback", lambda candidate: _swap_lookback(candidate.expression, lookbacks, randomizer)),
        ("field", lambda candidate: _swap_field(candidate.expression, field_registry, randomizer)),
        ("operator", lambda candidate: _swap_operator(candidate.expression, operator_swaps, randomizer)),
        ("wrapper", lambda candidate: _swap_wrapper(candidate.expression, normalization_wrappers, randomizer)),
        ("simplify", lambda candidate: _simplify_expression(candidate.expression, randomizer)),
    ]

    attempts = 0
    max_attempts = max(count * 12, 24)
    while len(payload) < count and attempts < max_attempts and parents:
        attempts += 1
        parent = randomizer.choice(list(parents))
        strategy_name, strategy = randomizer.choice(strategies)
        expression = strategy(parent)
        if not expression or expression == parent.expression:
            continue
        metadata = {
            "template_name": parent.template_name or str(parent.generation_metadata.get("template_name") or ""),
            "fields_used": list(parent.fields_used or parent.generation_metadata.get("fields_used") or []),
            "mutation_strategy": strategy_name,
            "parent_refs": [{"alpha_id": parent.alpha_id}],
        }
        payload.append((expression, (parent.alpha_id,), metadata))
    return payload


def _swap_lookback(expression: str, lookbacks: list[int], randomizer: random.Random) -> str | None:
    node = parse_expression(expression)
    mutated = _mutate_first(node, lambda current: _replace_lookback(current, lookbacks, randomizer))
    return to_expression(mutated) if mutated is not None else None


def _swap_field(expression: str, field_registry: FieldRegistry, randomizer: random.Random) -> str | None:
    node = parse_expression(expression)
    mutated = _mutate_first(node, lambda current: _replace_identifier(current, field_registry, randomizer))
    return to_expression(mutated) if mutated is not None else None


def _swap_operator(
    expression: str,
    operator_swaps: dict[str, list[str]],
    randomizer: random.Random,
) -> str | None:
    node = parse_expression(expression)
    mutated = _mutate_first(node, lambda current: _replace_operator(current, operator_swaps, randomizer))
    return to_expression(mutated) if mutated is not None else None


def _swap_wrapper(expression: str, normalization_wrappers: list[str], randomizer: random.Random) -> str:
    node = parse_expression(expression)
    if isinstance(node, FunctionCallNode) and len(node.args) == 1 and node.name in normalization_wrappers:
        alternative = randomizer.choice([name for name in normalization_wrappers if name != node.name] or normalization_wrappers)
        return to_expression(FunctionCallNode(name=alternative, args=node.args))
    wrapper = randomizer.choice(normalization_wrappers)
    return to_expression(FunctionCallNode(name=wrapper, args=(node,)))


def _simplify_expression(expression: str, randomizer: random.Random) -> str | None:
    node = parse_expression(expression)
    if isinstance(node, FunctionCallNode) and len(node.args) == 1:
        return to_expression(node.args[0])
    if isinstance(node, UnaryOpNode):
        return to_expression(node.operand)
    if isinstance(node, BinaryOpNode):
        return to_expression(randomizer.choice([node.left, node.right]))
    return None


def _mutate_first(node: ExprNode, transform) -> ExprNode | None:
    changed = transform(node)
    if changed is not None:
        return changed
    if isinstance(node, UnaryOpNode):
        child = _mutate_first(node.operand, transform)
        return replace(node, operand=child) if child is not None else None
    if isinstance(node, BinaryOpNode):
        left = _mutate_first(node.left, transform)
        if left is not None:
            return replace(node, left=left)
        right = _mutate_first(node.right, transform)
        return replace(node, right=right) if right is not None else None
    if isinstance(node, FunctionCallNode):
        for index, argument in enumerate(node.args):
            child = _mutate_first(argument, transform)
            if child is None:
                continue
            args = list(node.args)
            args[index] = child
            return replace(node, args=tuple(args))
    return None


def _replace_lookback(node: ExprNode, lookbacks: list[int], randomizer: random.Random) -> ExprNode | None:
    if not isinstance(node, FunctionCallNode) or not node.args:
        return None
    last = node.args[-1]
    if not isinstance(last, NumberNode) or len(lookbacks) <= 1:
        return None
    options = [value for value in lookbacks if int(value) != int(last.value)]
    if not options:
        return None
    args = list(node.args)
    args[-1] = NumberNode(float(randomizer.choice(options)))
    return replace(node, args=tuple(args))


def _replace_identifier(node: ExprNode, field_registry: FieldRegistry, randomizer: random.Random) -> ExprNode | None:
    if not isinstance(node, IdentifierNode):
        return None
    source = field_registry.get(node.name) if field_registry.contains(node.name) else None
    if source is not None and source.operator_type == "group":
        pool = field_registry.runtime_group_fields()
    else:
        pool = field_registry.runtime_numeric_fields()
    candidates = [spec for spec in pool if spec.name != node.name]
    if not candidates:
        return None
    weights = [
        max(
            1e-6,
            spec.field_score * (1.25 if source is not None and spec.category == source.category else 0.85),
        )
        for spec in candidates
    ]
    replacement = randomizer.choices(candidates, weights=weights, k=1)[0]
    return IdentifierNode(replacement.name)


def _replace_operator(
    node: ExprNode,
    operator_swaps: dict[str, list[str]],
    randomizer: random.Random,
) -> ExprNode | None:
    if not isinstance(node, FunctionCallNode) or node.name not in operator_swaps:
        return None
    replacement = randomizer.choice(operator_swaps[node.name])
    return replace(node, name=replacement)
