from __future__ import annotations

from itertools import combinations


def generate_template_expressions(
    fields: list[str],
    lookbacks: list[int],
    normalization_wrappers: list[str],
) -> list[str]:
    expressions: list[str] = []
    value_fields = [field for field in fields if field != "returns"]
    volume_fallback = "volume" if "volume" in fields else fields[0]

    for field in fields:
        for window in lookbacks:
            expressions.append(f"rank(delta({field}, {window}))")
            expressions.append(f"zscore(ts_mean({field}, {window}))")
            expressions.append(f"ts_rank({field}, {window})")
            expressions.append(f"decay_linear(delta({field}, {window}), {window})")

    for left, right in combinations(value_fields, 2):
        for window in lookbacks:
            expressions.append(f"rank(correlation({left}, {right}, {window}))")
            expressions.append(f"rank(covariance({left}, {right}, {window}))")
            expressions.append(f"(sign(delta({left}, {window})) * zscore({right}))")

    for wrapper in normalization_wrappers:
        expressions.append(f"{wrapper}(returns(close, 1))")
        expressions.append(f"{wrapper}(zscore(ts_mean({volume_fallback}, {lookbacks[0]})))")

    deduped: list[str] = []
    seen: set[str] = set()
    for expression in expressions:
        if expression not in seen:
            deduped.append(expression)
            seen.add(expression)
    return deduped
