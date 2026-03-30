from __future__ import annotations

from collections import Counter

from core.config import CrowdingConfig, DiversityThresholdConfig
from generator.engine import AlphaCandidate
from memory.case_memory import CaseMemorySnapshot
from services.models import CrowdingScore


class CrowdingService:
    def __init__(
        self,
        *,
        config: CrowdingConfig,
        diversity_config: DiversityThresholdConfig,
    ) -> None:
        self.config = config
        self.diversity_config = diversity_config

    def score_pre_sim(
        self,
        candidates: list[AlphaCandidate],
        *,
        run_id: str,
        round_index: int,
        effective_regime_key: str,
        case_snapshot: CaseMemorySnapshot | None,
        selected_so_far: list[AlphaCandidate] | None = None,
        stage: str = "pre_sim",
    ) -> dict[str, CrowdingScore]:
        del run_id, round_index, effective_regime_key
        if not self.config.enabled or not candidates:
            return {
                candidate.alpha_id: CrowdingScore(
                    alpha_id=candidate.alpha_id,
                    stage=stage,
                    total_penalty=0.0,
                )
                for candidate in candidates
            }

        selected_so_far = list(selected_so_far or [])
        batch_size = max(1, len(candidates) + len(selected_so_far))
        family_counts = Counter(
            str(candidate.generation_metadata.get("family_signature") or "")
            for candidate in [*selected_so_far, *candidates]
            if candidate.generation_metadata.get("family_signature")
        )
        motif_counts = Counter(
            str(candidate.generation_metadata.get("motif") or candidate.template_name or "")
            for candidate in [*selected_so_far, *candidates]
            if candidate.generation_metadata.get("motif") or candidate.template_name
        )
        operator_path_counts = Counter(
            ">".join((candidate.generation_metadata.get("operator_path") or [])[:4]) or "none"
            for candidate in [*selected_so_far, *candidates]
        )
        lineage_counts = Counter(
            str(candidate.generation_metadata.get("lineage_branch_key") or "")
            for candidate in [*selected_so_far, *candidates]
            if candidate.generation_metadata.get("lineage_branch_key")
        )

        family_stats = case_snapshot.stats_for_scope("family", scope="blended") if case_snapshot else {}
        motif_stats = case_snapshot.stats_for_scope("motif", scope="blended") if case_snapshot else {}
        operator_stats = case_snapshot.stats_for_scope("operator_path", scope="blended") if case_snapshot else {}
        family_sample_count = max(1, int(case_snapshot.sample_count if case_snapshot else 0))

        scores: dict[str, CrowdingScore] = {}
        for candidate in candidates:
            family_signature = str(candidate.generation_metadata.get("family_signature") or "")
            motif = str(candidate.generation_metadata.get("motif") or candidate.template_name or "")
            operator_path_key = ">".join((candidate.generation_metadata.get("operator_path") or [])[:4]) or "none"
            lineage_branch_key = str(candidate.generation_metadata.get("lineage_branch_key") or "")

            family_ratio = family_counts.get(family_signature, 0) / batch_size if family_signature else 0.0
            motif_ratio = motif_counts.get(motif, 0) / batch_size if motif else 0.0
            operator_ratio = operator_path_counts.get(operator_path_key, 0) / batch_size
            lineage_ratio = lineage_counts.get(lineage_branch_key, 0) / batch_size if lineage_branch_key else 0.0

            family_penalty = self.config.family_penalty_weight * family_ratio
            motif_penalty = self.config.motif_penalty_weight * motif_ratio
            operator_path_penalty = self.config.operator_path_penalty_weight * operator_ratio
            lineage_penalty = self.config.lineage_penalty_weight * lineage_ratio
            batch_penalty = self.config.batch_penalty_weight * max(family_ratio, motif_ratio, operator_ratio)

            historical_penalty = 0.0
            if family_signature and family_signature in family_stats:
                historical_penalty += (
                    self.config.historical_penalty_weight
                    * min(1.0, family_stats[family_signature].support / max(1, family_sample_count))
                )
            if motif and motif in motif_stats:
                historical_penalty += (
                    0.5
                    * self.config.historical_penalty_weight
                    * min(1.0, motif_stats[motif].support / max(1, family_sample_count))
                )
            if operator_path_key in operator_stats:
                historical_penalty += (
                    0.5
                    * self.config.historical_penalty_weight
                    * min(1.0, operator_stats[operator_path_key].support / max(1, family_sample_count))
                )
            total_penalty = float(
                family_penalty
                + motif_penalty
                + operator_path_penalty
                + lineage_penalty
                + batch_penalty
                + historical_penalty
            )
            hard_blocked = (
                family_ratio > float(self.diversity_config.max_family_fraction)
                or lineage_ratio > float(self.diversity_config.max_lineage_branch_fraction)
            )
            reason_codes: list[str] = []
            if family_ratio > float(self.diversity_config.max_family_fraction):
                reason_codes.append("family_saturation_cap")
            elif family_ratio >= 0.5 * float(self.diversity_config.max_family_fraction):
                reason_codes.append("family_saturation_penalty")
            if motif_ratio >= 0.50:
                reason_codes.append("motif_saturation_penalty")
            if operator_ratio >= float(self.diversity_config.max_operator_path_fraction):
                reason_codes.append("operator_path_concentration")
            if lineage_ratio > float(self.diversity_config.max_lineage_branch_fraction):
                reason_codes.append("lineage_branch_cap")
            elif lineage_ratio >= 0.5 * float(self.diversity_config.max_lineage_branch_fraction) and lineage_ratio > 0:
                reason_codes.append("lineage_branch_penalty")

            scores[candidate.alpha_id] = CrowdingScore(
                alpha_id=candidate.alpha_id,
                stage=stage,
                total_penalty=total_penalty,
                family_penalty=family_penalty,
                motif_penalty=motif_penalty,
                operator_path_penalty=operator_path_penalty,
                lineage_penalty=lineage_penalty,
                batch_penalty=batch_penalty,
                historical_penalty=historical_penalty,
                hard_blocked=hard_blocked,
                reason_codes=tuple(reason_codes),
                metrics={
                    "family_ratio": family_ratio,
                    "motif_ratio": motif_ratio,
                    "operator_ratio": operator_ratio,
                    "lineage_ratio": lineage_ratio,
                    "batch_size": batch_size,
                },
            )
        return scores
