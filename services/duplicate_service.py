from __future__ import annotations

import json
from dataclasses import dataclass

from core.config import DuplicateConfig
from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemoryService, StructuralSignature
from services.models import DedupBatchResult, DedupDecision
from storage.repository import SQLiteRepository


@dataclass(slots=True, frozen=True)
class _CandidateReference:
    run_id: str
    alpha_id: str
    normalized_expression: str
    signature: StructuralSignature | None
    scope: str


class DuplicateService:
    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        config: DuplicateConfig,
        memory_service: PatternMemoryService | None = None,
    ) -> None:
        self.repository = repository
        self.config = config
        self.memory_service = memory_service or PatternMemoryService()
        self._same_run_ref_cache: dict[tuple[str, int, str, int], tuple[_CandidateReference, ...]] = {}

    def filter_pre_sim(
        self,
        candidates: list[AlphaCandidate],
        *,
        run_id: str,
        round_index: int,
        legacy_regime_key: str,
        global_regime_key: str,
        stage: str = "pre_sim",
    ) -> DedupBatchResult:
        del legacy_regime_key
        if not self.config.enabled or not candidates:
            return DedupBatchResult(
                kept_candidates=tuple(candidates),
                blocked_candidates=(),
                decisions=tuple(
                    DedupDecision(
                        alpha_id=candidate.alpha_id,
                        normalized_expression=candidate.normalized_expression,
                        stage=stage,
                        decision="kept",
                        reason_code="duplicate_service_disabled",
                    )
                    for candidate in candidates
                ),
                stage_metrics={
                    "generated": len(candidates),
                    "kept_after_dedup": len(candidates),
                    "blocked_exact": 0,
                    "blocked_near": 0,
                    "blocked_cross_run": 0,
                },
            )

        current_alpha_ids = {candidate.alpha_id for candidate in candidates}
        existing_by_normalized = self.repository.get_existing_alpha_ids_by_normalized(
            run_id,
            (candidate.normalized_expression for candidate in candidates),
            exclude_alpha_ids=current_alpha_ids,
        )
        same_run_refs = (
            self._same_run_structural_refs(run_id=run_id, exclude_alpha_ids=current_alpha_ids)
            if self.config.structural_match_enabled
            else []
        )
        cross_run_refs = [
            _CandidateReference(
                run_id=str(row["run_id"]),
                alpha_id=str(row["alpha_id"]),
                normalized_expression=str(row["normalized_expression"]),
                signature=self._signature_from_payload(json.loads(row["structural_signature_json"] or "{}")),
                scope="global_profile",
            )
            for row in self.repository.get_cross_run_duplicate_references(
                run_id=run_id,
                global_regime_key=global_regime_key,
                limit=self.config.max_cross_run_references,
            )
        ] if self.config.cross_run_enabled else []

        kept: list[AlphaCandidate] = []
        blocked: list[AlphaCandidate] = []
        decisions: list[DedupDecision] = []
        kept_refs: list[_CandidateReference] = []
        blocked_exact = 0
        blocked_near = 0
        blocked_cross_run = 0

        for candidate in candidates:
            normalized_expression = candidate.normalized_expression
            if self.config.exact_match_enabled and normalized_expression in existing_by_normalized:
                blocked.append(candidate)
                blocked_exact += 1
                decisions.append(
                    DedupDecision(
                        alpha_id=candidate.alpha_id,
                        normalized_expression=normalized_expression,
                        stage=stage,
                        decision="blocked",
                        reason_code="exact_same_run",
                        matched_run_id=run_id,
                        matched_alpha_id=existing_by_normalized[normalized_expression],
                        matched_scope="same_run",
                        similarity_score=1.0,
                        normalized_match=True,
                    )
                )
                continue

            duplicate_in_kept = next(
                (reference for reference in kept_refs if reference.normalized_expression == normalized_expression),
                None,
            )
            if self.config.exact_match_enabled and duplicate_in_kept is not None:
                blocked.append(candidate)
                blocked_exact += 1
                decisions.append(
                    DedupDecision(
                        alpha_id=candidate.alpha_id,
                        normalized_expression=normalized_expression,
                        stage=stage,
                        decision="blocked",
                        reason_code="exact_same_run",
                        matched_run_id=duplicate_in_kept.run_id,
                        matched_alpha_id=duplicate_in_kept.alpha_id,
                        matched_scope=duplicate_in_kept.scope,
                        similarity_score=1.0,
                        normalized_match=True,
                    )
                )
                continue

            cross_exact = next(
                (reference for reference in cross_run_refs if reference.normalized_expression == normalized_expression),
                None,
            )
            if self.config.exact_match_enabled and cross_exact is not None:
                blocked.append(candidate)
                blocked_cross_run += 1
                decisions.append(
                    DedupDecision(
                        alpha_id=candidate.alpha_id,
                        normalized_expression=normalized_expression,
                        stage=stage,
                        decision="blocked",
                        reason_code="exact_cross_run",
                        matched_run_id=cross_exact.run_id,
                        matched_alpha_id=cross_exact.alpha_id,
                        matched_scope=cross_exact.scope,
                        similarity_score=1.0,
                        normalized_match=True,
                    )
                )
                continue

            signature = self._safe_extract_signature(
                candidate.normalized_expression or candidate.expression,
                generation_metadata=candidate.generation_metadata,
            )
            same_run_similarity, same_run_match = self._best_structural_match(signature, [*same_run_refs, *kept_refs])
            cross_run_similarity, cross_run_match = self._best_structural_match(signature, cross_run_refs)
            branch_reuse_similarity = self._branch_reuse_similarity(candidate, kept)

            if (
                self.config.structural_match_enabled
                and same_run_match is not None
                and same_run_similarity >= self.config.same_run_structural_threshold
            ):
                blocked.append(candidate)
                blocked_near += 1
                decisions.append(
                    DedupDecision(
                        alpha_id=candidate.alpha_id,
                        normalized_expression=normalized_expression,
                        stage=stage,
                        decision="blocked",
                        reason_code="near_structural_same_run",
                        matched_run_id=same_run_match.run_id,
                        matched_alpha_id=same_run_match.alpha_id,
                        matched_scope=same_run_match.scope,
                        similarity_score=same_run_similarity,
                        metrics={
                            "current_batch_similarity": same_run_similarity,
                            "cross_run_similarity": cross_run_similarity,
                            "branch_reuse_similarity": branch_reuse_similarity,
                        },
                    )
                )
                continue

            if (
                self.config.structural_match_enabled
                and self.config.cross_run_enabled
                and cross_run_match is not None
                and cross_run_similarity >= self.config.cross_run_structural_threshold
            ):
                blocked.append(candidate)
                blocked_cross_run += 1
                decisions.append(
                    DedupDecision(
                        alpha_id=candidate.alpha_id,
                        normalized_expression=normalized_expression,
                        stage=stage,
                        decision="blocked",
                        reason_code="near_structural_cross_run",
                        matched_run_id=cross_run_match.run_id,
                        matched_alpha_id=cross_run_match.alpha_id,
                        matched_scope=cross_run_match.scope,
                        similarity_score=cross_run_similarity,
                        metrics={
                            "current_batch_similarity": same_run_similarity,
                            "cross_run_similarity": cross_run_similarity,
                            "branch_reuse_similarity": branch_reuse_similarity,
                        },
                    )
                )
                continue

            kept.append(candidate)
            kept_ref = _CandidateReference(
                run_id=run_id,
                alpha_id=candidate.alpha_id,
                normalized_expression=normalized_expression,
                signature=signature,
                scope="same_run",
            )
            kept_refs.append(kept_ref)
            decisions.append(
                DedupDecision(
                    alpha_id=candidate.alpha_id,
                    normalized_expression=normalized_expression,
                    stage=stage,
                    decision="kept",
                    reason_code="unique",
                    similarity_score=max(same_run_similarity, cross_run_similarity),
                    normalized_match=False,
                    metrics={
                        "duplicate_risk": max(same_run_similarity, cross_run_similarity, branch_reuse_similarity),
                        "current_batch_similarity": same_run_similarity,
                        "cross_run_similarity": cross_run_similarity,
                        "branch_reuse_similarity": branch_reuse_similarity,
                    },
                )
            )

        return DedupBatchResult(
            kept_candidates=tuple(kept),
            blocked_candidates=tuple(blocked),
            decisions=tuple(decisions),
            stage_metrics={
                "generated": len(candidates),
                "blocked_by_exact_dedup": blocked_exact,
                "blocked_by_near_duplicate": blocked_near,
                "blocked_by_cross_run_dedup": blocked_cross_run,
                "kept_after_dedup": len(kept),
            },
        )

    def structural_similarity(self, left: StructuralSignature | None, right: StructuralSignature | None) -> float:
        if left is None or right is None:
            return 0.0
        weights = {
            "motif": 0.15,
            "operators": 0.15,
            "operator_path": 0.20,
            "fields": 0.10,
            "field_families": 0.15,
            "lookbacks": 0.10,
            "wrappers": 0.05,
            "horizon_bucket": 0.05,
            "turnover_bucket": 0.03,
            "complexity_bucket": 0.02,
        }
        score = 0.0
        score += weights["motif"] if left.motif == right.motif else 0.0
        score += weights["operators"] * self._jaccard(left.operators, right.operators)
        score += weights["operator_path"] * self._prefix_similarity(left.operator_path, right.operator_path)
        score += weights["fields"] * self._jaccard(left.fields, right.fields)
        score += weights["field_families"] * self._jaccard(left.field_families, right.field_families)
        score += weights["lookbacks"] * self._jaccard(tuple(str(item) for item in left.lookbacks), tuple(str(item) for item in right.lookbacks))
        score += weights["wrappers"] * self._jaccard(left.wrappers, right.wrappers)
        score += weights["horizon_bucket"] if left.horizon_bucket == right.horizon_bucket else 0.0
        score += weights["turnover_bucket"] if left.turnover_bucket == right.turnover_bucket else 0.0
        score += weights["complexity_bucket"] if left.complexity_bucket == right.complexity_bucket else 0.0
        return float(max(0.0, min(1.0, score)))

    def _best_structural_match(
        self,
        signature: StructuralSignature | None,
        references: list[_CandidateReference],
    ) -> tuple[float, _CandidateReference | None]:
        best_similarity = 0.0
        best_reference: _CandidateReference | None = None
        for reference in references:
            similarity = self.structural_similarity(signature, reference.signature)
            if similarity > best_similarity:
                best_similarity = similarity
                best_reference = reference
        return best_similarity, best_reference

    def _branch_reuse_similarity(self, candidate: AlphaCandidate, kept: list[AlphaCandidate]) -> float:
        primary_parent = str(candidate.generation_metadata.get("primary_parent_alpha_id") or "")
        if not primary_parent:
            return 0.0
        overlap = sum(
            1
            for kept_candidate in kept
            if str(kept_candidate.generation_metadata.get("primary_parent_alpha_id") or "") == primary_parent
        )
        return float(min(1.0, overlap / max(1, len(kept) or 1)))

    def _safe_extract_signature(
        self,
        expression: str,
        *,
        generation_metadata: dict | None,
    ) -> StructuralSignature | None:
        try:
            return self.memory_service.extract_signature(expression, generation_metadata=generation_metadata)
        except Exception:  # noqa: BLE001
            return None

    def _same_run_structural_refs(
        self,
        *,
        run_id: str,
        exclude_alpha_ids: set[str],
    ) -> list[_CandidateReference]:
        limit = max(int(self.config.same_run_structural_reference_limit), 0)
        if limit <= 0:
            return []
        alpha_count, max_created_at = self.repository.get_alpha_reference_marker(run_id)
        cache_key = (run_id, alpha_count, max_created_at, limit)
        cached = self._same_run_ref_cache.get(cache_key)
        if cached is None:
            rows = self.repository.get_same_run_structural_references(run_id=run_id, limit=limit)
            refs: list[_CandidateReference] = []
            fallback_remaining = min(100, limit)
            for row in rows:
                signature = self._signature_from_payload_json(str(row.get("structural_signature_json") or "{}"))
                if signature is None and fallback_remaining > 0:
                    signature = self._safe_extract_signature(
                        str(row.get("expression") or row.get("normalized_expression") or ""),
                        generation_metadata=self._record_metadata(str(row.get("generation_metadata") or "{}")),
                    )
                    fallback_remaining -= 1
                    if signature is not None:
                        self.repository.update_alpha_structural_signature(
                            run_id=str(row.get("run_id") or run_id),
                            alpha_id=str(row.get("alpha_id") or ""),
                            structural_signature_json=json.dumps(signature.to_dict(), sort_keys=True),
                        )
                refs.append(
                    _CandidateReference(
                        run_id=str(row.get("run_id") or run_id),
                        alpha_id=str(row.get("alpha_id") or ""),
                        normalized_expression=str(row.get("normalized_expression") or ""),
                        signature=signature,
                        scope="same_run",
                    )
                )
            cached = tuple(refs)
            self._same_run_ref_cache.clear()
            self._same_run_ref_cache[cache_key] = cached
        return [reference for reference in cached if reference.alpha_id not in exclude_alpha_ids]

    def _signature_from_payload_json(self, payload: str) -> StructuralSignature | None:
        try:
            parsed = json.loads(payload or "{}")
        except json.JSONDecodeError:
            return None
        return self._signature_from_payload(parsed if isinstance(parsed, dict) else {})

    @staticmethod
    def _record_metadata(payload: str) -> dict:
        try:
            parsed = json.loads(payload or "{}")
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _jaccard(left, right) -> float:
        left_set = set(left)
        right_set = set(right)
        if not left_set and not right_set:
            return 1.0
        if not left_set or not right_set:
            return 0.0
        return float(len(left_set & right_set) / len(left_set | right_set))

    @staticmethod
    def _prefix_similarity(left, right) -> float:
        left_items = tuple(left[:4])
        right_items = tuple(right[:4])
        max_len = max(len(left_items), len(right_items), 1)
        matches = 0
        for left_item, right_item in zip(left_items, right_items, strict=False):
            if left_item == right_item:
                matches += 1
            else:
                break
        return float(matches / max_len)

    @staticmethod
    def _signature_from_payload(payload: dict) -> StructuralSignature | None:
        if not payload:
            return None
        try:
            return StructuralSignature(
                operators=tuple(payload.get("operators", ())),
                operator_families=tuple(payload.get("operator_families", ())),
                operator_path=tuple(payload.get("operator_path", ())),
                fields=tuple(payload.get("fields", ())),
                field_families=tuple(payload.get("field_families", ())),
                lookbacks=tuple(payload.get("lookbacks", ())),
                wrappers=tuple(payload.get("wrappers", ())),
                depth=int(payload.get("depth", 0) or 0),
                complexity=int(payload.get("complexity", 0) or 0),
                complexity_bucket=str(payload.get("complexity_bucket", "")),
                horizon_bucket=str(payload.get("horizon_bucket", "")),
                turnover_bucket=str(payload.get("turnover_bucket", "")),
                motif=str(payload.get("motif", "")),
                family_signature=str(payload.get("family_signature", "")),
                subexpressions=tuple(payload.get("subexpressions", ())),
            )
        except Exception:  # noqa: BLE001
            return None
