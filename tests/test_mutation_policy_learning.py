from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
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


# ── Phase 3: _motif_success_weights tests ─────────────────────────────────────


def _make_record(motif: str, delta: float, days_ago: float = 0.0) -> dict:
    """Helper: build a mutation_learning_record with child_motif and created_at."""
    created_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        "mutation_mode": "structural",
        "family_signature": "family-x",
        "child_motif": motif,
        "outcome_delta": delta,
        "created_at": created_at,
    }


def test_motif_success_weights_favor_winning_motif() -> None:
    """Motifs with consistently positive uplifts should receive weight > 1."""
    config = load_config("config/dev.yaml")
    records = [
        _make_record("momentum", 0.5),
        _make_record("momentum", 0.4),
        _make_record("momentum", 0.6),
        _make_record("mean_reversion", -0.3),
        _make_record("mean_reversion", -0.2),
        _make_record("mean_reversion", -0.4),
    ]
    policy = MutationPolicy(
        config=config.generation,
        adaptive_config=config.adaptive_generation,
        memory_service=PatternMemoryService(),
        mutation_learning_records=records,
    )

    weights = policy._motif_success_weights()

    assert "momentum" in weights
    assert "mean_reversion" in weights
    assert weights["momentum"] > 1.0, "winning motif should have weight > 1.0"
    assert weights["momentum"] > weights["mean_reversion"], "winner should outweigh loser"
    assert weights["mean_reversion"] >= 0.10, "loser should still be >= floor 0.10"


def test_motif_success_weights_time_decay_reduces_old_records() -> None:
    """Records from 30 days ago should contribute less than recent records."""
    config = load_config("config/dev.yaml")
    # score_decay=0.98 -> after 30 days: each delta is scaled by 0.98^30 ≈ 0.545
    recent_records = [_make_record("momentum", 0.8, days_ago=0) for _ in range(3)]
    old_records = [_make_record("spread", 0.8, days_ago=30) for _ in range(3)]
    policy = MutationPolicy(
        config=config.generation,
        adaptive_config=config.adaptive_generation,
        memory_service=PatternMemoryService(),
        mutation_learning_records=recent_records + old_records,
    )

    weights = policy._motif_success_weights()

    assert "momentum" in weights
    assert "spread" in weights
    # Both have same raw delta, but old records are decayed → momentum should score higher
    assert weights["momentum"] > weights["spread"], "recent records should outweigh old ones"


def test_motif_success_weights_fallback_when_no_child_motif() -> None:
    """Records without child_motif should be silently skipped → empty weights."""
    config = load_config("config/dev.yaml")
    records = [
        {
            "mutation_mode": "structural",
            "family_signature": "f",
            "outcome_delta": 0.5,
            "child_motif": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "mutation_mode": "structural",
            "family_signature": "f",
            "outcome_delta": 0.6,
            "created_at": datetime.now(timezone.utc).isoformat(),
            # no child_motif key at all
        },
    ]
    policy = MutationPolicy(
        config=config.generation,
        adaptive_config=config.adaptive_generation,
        memory_service=PatternMemoryService(),
        mutation_learning_records=records,
    )

    weights = policy._motif_success_weights()

    assert weights == {}, "records with no child_motif should yield empty weights"


def test_motif_success_weights_min_support_respected() -> None:
    """Motifs with fewer records than min_support=3 should be excluded."""
    config = load_config("config/dev.yaml")
    records = [
        _make_record("momentum", 0.9),
        _make_record("momentum", 0.8),
        # Only 2 records → below min_support=3
    ]
    policy = MutationPolicy(
        config=config.generation,
        adaptive_config=config.adaptive_generation,
        memory_service=PatternMemoryService(),
        mutation_learning_records=records,
    )

    weights = policy._motif_success_weights()

    assert weights == {}, "should return empty when all motifs are below min_support"
