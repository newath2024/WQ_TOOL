from __future__ import annotations

from core.config import CrowdingConfig, DiversityThresholdConfig
from generator.engine import AlphaCandidate
from services.crowding_service import CrowdingService


def test_crowding_penalty_increases_with_family_concentration() -> None:
    service = CrowdingService(
        config=CrowdingConfig(),
        diversity_config=DiversityThresholdConfig(
            max_family_fraction=1.0,
            max_operator_path_fraction=1.0,
            max_lineage_branch_fraction=1.0,
        ),
    )
    sparse = [
        _candidate("alpha-1", family="family-a", motif="momentum", operator_path=("rank", "ts_mean")),
        _candidate("alpha-2", family="family-b", motif="mean_reversion", operator_path=("zscore", "ts_delta")),
    ]
    crowded = [
        _candidate("alpha-1", family="family-a", motif="momentum", operator_path=("rank", "ts_mean")),
        _candidate("alpha-2", family="family-a", motif="momentum", operator_path=("rank", "ts_mean")),
        _candidate("alpha-3", family="family-a", motif="momentum", operator_path=("rank", "ts_mean")),
    ]

    sparse_scores = service.score_pre_sim(
        sparse,
        run_id="run-1",
        round_index=1,
        effective_regime_key="regime",
        case_snapshot=None,
    )
    crowded_scores = service.score_pre_sim(
        crowded,
        run_id="run-1",
        round_index=1,
        effective_regime_key="regime",
        case_snapshot=None,
    )

    assert crowded_scores["alpha-1"].total_penalty > sparse_scores["alpha-1"].total_penalty


def test_family_saturation_can_hard_block_candidate() -> None:
    service = CrowdingService(
        config=CrowdingConfig(),
        diversity_config=DiversityThresholdConfig(
            max_family_fraction=0.25,
            max_operator_path_fraction=1.0,
            max_lineage_branch_fraction=1.0,
        ),
    )
    candidates = [
        _candidate("alpha-1", family="family-a", motif="momentum", operator_path=("rank", "ts_mean")),
        _candidate("alpha-2", family="family-a", motif="momentum", operator_path=("rank", "ts_mean")),
        _candidate("alpha-3", family="family-a", motif="momentum", operator_path=("rank", "ts_mean")),
        _candidate("alpha-4", family="family-b", motif="volatility", operator_path=("rank", "ts_std")),
    ]

    scores = service.score_pre_sim(
        candidates,
        run_id="run-1",
        round_index=1,
        effective_regime_key="regime",
        case_snapshot=None,
    )

    assert scores["alpha-1"].hard_blocked is True
    assert "family_saturation_cap" in scores["alpha-1"].reason_codes


def _candidate(
    alpha_id: str,
    *,
    family: str,
    motif: str,
    operator_path: tuple[str, ...],
) -> AlphaCandidate:
    return AlphaCandidate(
        alpha_id=alpha_id,
        expression=f"{motif}_{alpha_id}",
        normalized_expression=f"{motif}_{alpha_id}",
        generation_mode="template",
        parent_ids=(),
        complexity=3,
        created_at="2026-01-01T00:00:00+00:00",
        template_name=motif,
        fields_used=("close",),
        operators_used=operator_path,
        depth=2,
        generation_metadata={
            "family_signature": family,
            "motif": motif,
            "operator_path": list(operator_path),
            "lineage_branch_key": f"branch-{family}",
        },
    )
