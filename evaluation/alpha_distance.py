from __future__ import annotations

from memory.pattern_memory import StructuralSignature


def _jaccard_distance(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - (len(left_set & right_set) / len(union))


def structural_distance(left: StructuralSignature, right: StructuralSignature) -> float:
    if left.family_signature == right.family_signature and left.operators == right.operators and left.fields == right.fields:
        return 0.0
    motif_distance = 0.0 if left.motif == right.motif else 1.0
    family_distance = _jaccard_distance(left.field_families, right.field_families)
    operator_path_distance = _jaccard_distance(left.operator_path, right.operator_path)
    wrapper_distance = _jaccard_distance(left.wrappers, right.wrappers)
    horizon_distance = 0.0 if left.horizon_bucket == right.horizon_bucket else 1.0
    turnover_distance = 0.0 if left.turnover_bucket == right.turnover_bucket else 1.0
    complexity_distance = 0.0 if left.complexity_bucket == right.complexity_bucket else 1.0
    return max(
        0.0,
        min(
            1.0,
            (
                0.20 * motif_distance
                + 0.20 * family_distance
                + 0.20 * operator_path_distance
                + 0.10 * wrapper_distance
                + 0.10 * horizon_distance
                + 0.10 * turnover_distance
                + 0.10 * complexity_distance
            ),
        ),
    )

