from __future__ import annotations

from types import SimpleNamespace

from core.config import load_config
from generator.mutation_policy import MutationPolicy
from memory.pattern_memory import PatternMemoryService


def test_mutation_outcome_weights_favor_positive_uplift_modes() -> None:
    config = load_config("config/dev.yaml")
    policy = MutationPolicy(
        config=config.generation,
        adaptive_config=config.adaptive_generation,
        memory_service=PatternMemoryService(),
        mutation_learning_records=[
            {"family_signature": "family-a", "mutation_mode": "structural", "outcome_delta": 0.6},
            {"family_signature": "family-a", "mutation_mode": "structural", "outcome_delta": 0.4},
            {"family_signature": "family-a", "mutation_mode": "structural", "outcome_delta": 0.5},
            {"family_signature": "family-a", "mutation_mode": "repair", "outcome_delta": -0.4},
            {"family_signature": "family-a", "mutation_mode": "repair", "outcome_delta": -0.2},
            {"family_signature": "family-a", "mutation_mode": "repair", "outcome_delta": -0.3},
        ],
    )

    multipliers = policy._mutation_outcome_multipliers(family_signature="family-a")

    assert multipliers["structural"] > 1.0
    assert multipliers["repair"] < multipliers["structural"]
    assert multipliers["repair"] >= 0.10


def test_mutation_outcome_learning_falls_back_when_support_is_low() -> None:
    config = load_config("config/dev.yaml")
    policy = MutationPolicy(
        config=config.generation,
        adaptive_config=config.adaptive_generation,
        memory_service=PatternMemoryService(),
        mutation_learning_records=[
            {"family_signature": "family-a", "mutation_mode": "structural", "outcome_delta": 0.6},
        ],
    )

    multipliers = policy._mutation_outcome_multipliers(family_signature="family-a")

    assert multipliers == {}
