from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AlphaCandidate:
    alpha_id: str
    expression: str
    normalized_expression: str
    generation_mode: str
    parent_ids: tuple[str, ...]
    complexity: int
    created_at: str
    template_name: str = ""
    fields_used: tuple[str, ...] = ()
    operators_used: tuple[str, ...] = ()
    depth: int = 0
    generation_metadata: dict[str, Any] = field(default_factory=dict)
