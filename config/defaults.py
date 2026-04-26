from __future__ import annotations

DEFAULT_SIMULATION_PROFILES: tuple[dict[str, object], ...] = (
    {
        "name": "stable",
        "region": "USA",
        "universe": "TOP1000",
        "delay": 1,
        "neutralization": "SUBINDUSTRY",
        "decay": 3,
        "truncation": 0.01,
        "weight": 0.6,
    },
    {
        "name": "aggressive_short",
        "region": "USA",
        "universe": "TOP500",
        "delay": 1,
        "neutralization": "SUBINDUSTRY",
        "decay": 1,
        "truncation": 0.02,
        "weight": 0.4,
    },
)

__all__ = ["DEFAULT_SIMULATION_PROFILES"]
