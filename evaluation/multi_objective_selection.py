from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Iterable, TypeVar

from domain.metrics import ObjectiveVector


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RankedItem(Generic[T]):
    item: T
    objective_vector: ObjectiveVector
    family_signature: str
    primary_field_category: str
    horizon_bucket: str
    operator_path_key: str
    diversity_score: float
    exploration_candidate: bool
    crowding_distance: float = 0.0
    pareto_rank: int = 0
    rationale: dict[str, object] = field(default_factory=dict)


class MultiObjectiveSelectionService:
    BENEFIT_KEYS = ("fitness", "sharpe", "eligibility", "robustness", "novelty", "diversity")
    COST_KEYS = ("turnover_cost", "complexity_cost")

    def order(self, items: Iterable[RankedItem[T]]) -> list[RankedItem[T]]:
        population = list(items)
        if not population:
            return []
        fronts = self._non_dominated_sort(population)
        ordered: list[RankedItem[T]] = []
        for rank, front in enumerate(fronts):
            crowding = self._crowding_distance(front)
            enriched = [
                RankedItem(
                    item=item.item,
                    objective_vector=item.objective_vector,
                    family_signature=item.family_signature,
                    primary_field_category=item.primary_field_category,
                    horizon_bucket=item.horizon_bucket,
                    operator_path_key=item.operator_path_key,
                    diversity_score=item.diversity_score,
                    exploration_candidate=item.exploration_candidate,
                    crowding_distance=crowding.get(index, 0.0),
                    pareto_rank=rank,
                    rationale={
                        **dict(item.rationale),
                        "pareto_rank": rank,
                        "crowding_distance": crowding.get(index, 0.0),
                        "objective_vector": item.objective_vector.to_dict(),
                    },
                )
                for index, item in enumerate(front)
            ]
            ordered.extend(
                sorted(
                    enriched,
                    key=lambda item: (
                        -item.crowding_distance,
                        item.pareto_rank,
                        item.family_signature,
                    ),
                )
            )
        return ordered

    def dominates(self, left: ObjectiveVector, right: ObjectiveVector) -> bool:
        left_values = self._maximize_payload(left)
        right_values = self._maximize_payload(right)
        return all(l >= r for l, r in zip(left_values, right_values, strict=True)) and any(
            l > r for l, r in zip(left_values, right_values, strict=True)
        )

    def _non_dominated_sort(self, population: list[RankedItem[T]]) -> list[list[RankedItem[T]]]:
        remaining = list(population)
        fronts: list[list[RankedItem[T]]] = []
        while remaining:
            front: list[RankedItem[T]] = []
            for candidate in remaining:
                if not any(
                    self.dominates(other.objective_vector, candidate.objective_vector)
                    for other in remaining
                    if other is not candidate
                ):
                    front.append(candidate)
            fronts.append(front)
            remaining = [candidate for candidate in remaining if candidate not in front]
        return fronts

    def _crowding_distance(self, front: list[RankedItem[T]]) -> dict[int, float]:
        if len(front) <= 2:
            return {index: float("inf") for index in range(len(front))}
        distances = {index: 0.0 for index in range(len(front))}
        for key in (*self.BENEFIT_KEYS, *self.COST_KEYS):
            ordered = sorted(
                enumerate(front),
                key=lambda item: getattr(item[1].objective_vector, key),
            )
            distances[ordered[0][0]] = float("inf")
            distances[ordered[-1][0]] = float("inf")
            min_value = getattr(ordered[0][1].objective_vector, key)
            max_value = getattr(ordered[-1][1].objective_vector, key)
            scale = max(max_value - min_value, 1e-9)
            for index in range(1, len(ordered) - 1):
                prev_value = getattr(ordered[index - 1][1].objective_vector, key)
                next_value = getattr(ordered[index + 1][1].objective_vector, key)
                distances[ordered[index][0]] += abs(next_value - prev_value) / scale
        return distances

    def _maximize_payload(self, vector: ObjectiveVector) -> tuple[float, ...]:
        benefits = tuple(getattr(vector, key) for key in self.BENEFIT_KEYS)
        costs = tuple(-getattr(vector, key) for key in self.COST_KEYS)
        return benefits + costs
