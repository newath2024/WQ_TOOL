from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationGuardrails:
    blocked_group_neutralize_fields: frozenset[str] = frozenset()
    blocked_unit_category_pairs: frozenset[tuple[str, str]] = frozenset()

    @property
    def enabled(self) -> bool:
        return bool(self.blocked_group_neutralize_fields or self.blocked_unit_category_pairs)
