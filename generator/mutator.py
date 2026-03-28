from __future__ import annotations

import random
import re
from typing import Sequence


FIELD_PATTERN = re.compile(r"\b(open|high|low|close|volume|returns)\b")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_])(\d+)(?![A-Za-z_])")


def mutate_expressions(
    parents: Sequence[tuple[str, str]],
    count: int,
    fields: list[str],
    lookbacks: list[int],
    normalization_wrappers: list[str],
    seed: int,
) -> list[tuple[str, tuple[str, ...]]]:
    randomizer = random.Random(seed)
    expressions: list[tuple[str, tuple[str, ...]]] = []
    operator_swaps = {
        "ts_mean": ["ts_std", "ts_rank", "rolling_mean"],
        "ts_std": ["ts_mean", "rolling_std"],
        "delta": ["delay", "returns"],
        "correlation": ["covariance"],
        "covariance": ["correlation"],
        "rank": ["zscore", "sign"],
        "zscore": ["rank", "sign"],
    }

    while len(expressions) < count and parents:
        strategy = randomizer.choice(["lookback", "field", "wrapper", "operator", "combine"])
        parent_id, parent_expr = randomizer.choice(list(parents))
        if strategy == "lookback":
            mutated = _swap_lookback(parent_expr, lookbacks, randomizer)
            if mutated:
                expressions.append((mutated, (parent_id,)))
        elif strategy == "field":
            mutated = _swap_field(parent_expr, fields, randomizer)
            if mutated:
                expressions.append((mutated, (parent_id,)))
        elif strategy == "wrapper":
            wrapper = randomizer.choice(normalization_wrappers)
            expressions.append((f"{wrapper}({parent_expr})", (parent_id,)))
        elif strategy == "operator":
            mutated = _swap_operator(parent_expr, operator_swaps, randomizer)
            if mutated:
                expressions.append((mutated, (parent_id,)))
        else:
            other_id, other_expr = randomizer.choice(list(parents))
            expressions.append((f"rank(({parent_expr}) - ({other_expr}))", (parent_id, other_id)))
    return expressions


def _swap_lookback(expression: str, lookbacks: list[int], randomizer: random.Random) -> str | None:
    matches = list(NUMBER_PATTERN.finditer(expression))
    if not matches:
        return None
    match = randomizer.choice(matches)
    replacement = str(randomizer.choice(lookbacks))
    return expression[: match.start(1)] + replacement + expression[match.end(1) :]


def _swap_field(expression: str, fields: list[str], randomizer: random.Random) -> str | None:
    matches = list(FIELD_PATTERN.finditer(expression))
    if not matches:
        return None
    match = randomizer.choice(matches)
    replacement = randomizer.choice(fields)
    return expression[: match.start(0)] + replacement + expression[match.end(0) :]


def _swap_operator(
    expression: str,
    operator_swaps: dict[str, list[str]],
    randomizer: random.Random,
) -> str | None:
    candidates = [name for name in operator_swaps if f"{name}(" in expression]
    if not candidates:
        return None
    name = randomizer.choice(candidates)
    replacement = randomizer.choice(operator_swaps[name])
    return expression.replace(f"{name}(", f"{replacement}(", 1)
