from __future__ import annotations

import math
from collections import Counter
from typing import Generic, Iterable, TypeVar

from core.config import DiversityThresholdConfig
from services.multi_objective_selection import RankedItem


T = TypeVar("T")


class DiversityManager(Generic[T]):
    def __init__(self, config: DiversityThresholdConfig) -> None:
        self.config = config

    def select(self, ordered: Iterable[RankedItem[T]], *, batch_size: int) -> tuple[list[RankedItem[T]], list[RankedItem[T]]]:
        items = list(ordered)
        if batch_size <= 0 or not items:
            return [], items
        exploration_quota = max(1, int(math.ceil(batch_size * self.config.exploration_quota_fraction)))
        family_cap = max(1, int(math.ceil(batch_size * self.config.max_family_fraction)))
        field_cap = max(2, int(math.ceil(batch_size * self.config.max_field_category_fraction)))
        horizon_cap = max(2, int(math.ceil(batch_size * self.config.max_horizon_bucket_fraction)))
        operator_cap = max(2, int(math.ceil(batch_size * self.config.max_operator_path_fraction)))

        selected: list[RankedItem[T]] = []
        archived: list[RankedItem[T]] = []
        counts = {
            "family": Counter(),
            "field": Counter(),
            "horizon": Counter(),
            "operator": Counter(),
        }

        exploration_selected = 0
        for item in items:
            if exploration_selected >= exploration_quota:
                break
            if not item.exploration_candidate:
                continue
            if self._violates_cap(item, counts, family_cap, field_cap, horizon_cap, operator_cap):
                archived.append(item)
                continue
            self._accept(item, selected, counts)
            exploration_selected += 1
            if len(selected) >= batch_size:
                return selected[:batch_size], [candidate for candidate in items if candidate not in selected] + archived

        for item in items:
            if item in selected:
                continue
            if len(selected) >= batch_size:
                archived.append(item)
                continue
            if self._violates_cap(item, counts, family_cap, field_cap, horizon_cap, operator_cap):
                archived.append(item)
                continue
            self._accept(item, selected, counts)

        for item in items:
            if item not in selected and item not in archived:
                archived.append(item)
        return selected[:batch_size], archived

    def _accept(self, item: RankedItem[T], selected: list[RankedItem[T]], counts: dict[str, Counter]) -> None:
        selected.append(item)
        counts["family"][item.family_signature] += 1
        counts["field"][item.primary_field_category] += 1
        counts["horizon"][item.horizon_bucket] += 1
        counts["operator"][item.operator_path_key] += 1

    def _violates_cap(
        self,
        item: RankedItem[T],
        counts: dict[str, Counter],
        family_cap: int,
        field_cap: int,
        horizon_cap: int,
        operator_cap: int,
    ) -> bool:
        return (
            counts["family"][item.family_signature] >= family_cap
            or counts["field"][item.primary_field_category] >= field_cap
            or counts["horizon"][item.horizon_bucket] >= horizon_cap
            or counts["operator"][item.operator_path_key] >= operator_cap
        )
