from __future__ import annotations

from typing import Any


def _normalize_brain_enum(value: Any, *, true_value: str, false_value: str) -> str:
    if isinstance(value, bool):
        return true_value if value else false_value
    normalized = str(value or "").strip()
    lowered = normalized.lower()
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off"}
    if lowered in truthy:
        return true_value
    if lowered in falsy:
        return false_value
    return normalized.upper()


__all__ = ["_normalize_brain_enum"]
