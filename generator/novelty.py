from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from evaluation.alpha_distance import structural_distance
from domain.metrics import StructuralSignature


@dataclass(frozen=True, slots=True)
class NoveltyScore:
    score: float
    min_distance: float
    average_distance: float
    nearest_family_signature: str = ""


class NoveltySearch:
    def score(
        self,
        signature: StructuralSignature,
        references: Iterable[StructuralSignature],
    ) -> NoveltyScore:
        distances = [
            (structural_distance(signature, reference), reference.family_signature)
            for reference in references
        ]
        if not distances:
            return NoveltyScore(score=1.0, min_distance=1.0, average_distance=1.0)
        min_distance, nearest_family_signature = min(distances, key=lambda item: item[0])
        average_distance = sum(item[0] for item in distances) / len(distances)
        score = max(0.0, min(1.0, (0.65 * min_distance) + (0.35 * average_distance)))
        return NoveltyScore(
            score=score,
            min_distance=min_distance,
            average_distance=average_distance,
            nearest_family_signature=nearest_family_signature,
        )

