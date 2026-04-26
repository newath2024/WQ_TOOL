from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StorageConfig:
    path: str


__all__ = ["StorageConfig"]
