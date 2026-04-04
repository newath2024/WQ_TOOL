from __future__ import annotations

import re

_UNKNOWN_VARIABLE_PATTERN = re.compile(r'unknown variable "([^"]+)"', re.IGNORECASE)
_INVALID_DATA_FIELD_PATTERN = re.compile(r"invalid data field ([A-Za-z0-9_]+)", re.IGNORECASE)


def extract_invalid_field_from_rejection(reason: str | None) -> str | None:
    text = str(reason or "").strip()
    if not text:
        return None
    for pattern in (_UNKNOWN_VARIABLE_PATTERN, _INVALID_DATA_FIELD_PATTERN):
        match = pattern.search(text)
        if not match:
            continue
        field_name = str(match.group(1) or "").strip()
        if field_name:
            return field_name
    return None
