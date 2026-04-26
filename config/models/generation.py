from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GenerationConfig:
    allowed_fields: list[str]
    allowed_operators: list[str]
    lookbacks: list[int]
    max_depth: int
    complexity_limit: int
    template_count: int
    grammar_count: int
    mutation_count: int
    normalization_wrappers: list[str]
    random_seed: int = 7
    field_catalog_paths: list[str] = field(default_factory=list)
    operator_catalog_paths: list[str] = field(default_factory=list)
    field_value_paths: list[str] = field(default_factory=list)
    allow_catalog_fields_without_runtime: bool = False
    field_score_weights: dict[str, float] = field(
        default_factory=lambda: {"coverage": 0.50, "usage": 0.30, "category": 0.20}
    )
    category_weights: dict[str, float] = field(
        default_factory=lambda: {
            "price": 1.00,
            "volume": 0.85,
            "fundamental": 0.95,
            "analyst": 0.90,
            "model": 0.85,
            "sentiment": 0.75,
            "risk": 0.70,
            "macro": 0.65,
            "group": 0.60,
            "other": 0.50,
        }
    )
    template_weights: dict[str, float] = field(default_factory=dict)
    template_pool_size: int = 200
    max_turnover_bias: float = 0.35
    engine_validation_cache_enabled: bool = True
    sim_neutralization: str = "none"
    sim_decay: int = 0


__all__ = ["GenerationConfig"]
