from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from alpha.ast_nodes import (
    BinaryOpNode,
    ExprNode,
    FunctionCallNode,
    IdentifierNode,
    NumberNode,
    UnaryOpNode,
    iter_child_nodes,
    node_complexity,
    to_expression,
)
from alpha.parser import parse_expression
from core.config import AdaptiveGenerationConfig, GenerationConfig, QualityOptimizationConfig
from core.quality_score import MultiObjectiveQualityScorer
from data.field_registry import FieldRegistry
from domain.candidate import AlphaCandidate
from features.registry import OperatorRegistry, WINDOWED_OPERATORS
from generator.engine import AlphaGenerationEngine, GenerationSessionStats
from generator.guardrails import GenerationGuardrails
from services.elite_motifs import build_elite_seed_variants
from memory.pattern_memory import RegionLearningContext
from services.evaluation_service import alpha_candidate_from_record
from services.search_space_filter import expression_fields_and_operators
from storage.models import AlphaRecord
from storage.repository import SQLiteRepository


@dataclass(slots=True)
class QualityPolishStats:
    enabled: bool = True
    parent_count: int = 0
    eligible_parent_count: int = 0
    scanned_parent_count: int = 0
    saturated_parent_count: int = 0
    generated_count: int = 0
    selected_count: int = 0
    attempt_count: int = 0
    success_count: int = 0
    top_parent_quality: float | None = None
    transform_counts: Counter[str] = field(default_factory=Counter)
    failure_counts: Counter[str] = field(default_factory=Counter)
    failure_field_counts: dict[str, Counter[str]] = field(default_factory=dict)
    skipped_used_signature: int = 0
    skipped_used_parent_transform: int = 0
    skipped_existing_normalized: int = 0
    disabled_transform_counts: Counter[str] = field(default_factory=Counter)
    transform_cooldown_counts: Counter[str] = field(default_factory=Counter)
    transform_attempt_counts: Counter[str] = field(default_factory=Counter)
    transform_scores: dict[str, float] = field(default_factory=dict)
    cooldown_exempt_groups: list[str] = field(default_factory=list)
    external_elite_seed_count: int = 0
    external_elite_generated: int = 0
    search_space_filter_blocked: int = 0
    turnover_repair_generated: int = 0
    turnover_repair_attempt_count: int = 0
    turnover_repair_success_count: int = 0
    turnover_repair_selected: int = 0
    turnover_repair_transform_counts: Counter[str] = field(default_factory=Counter)
    generation_total_ms: float = 0.0

    def record_failure(
        self,
        reason: str | None,
        *,
        fields: Iterable[str] = (),
    ) -> None:
        normalized = str(reason or "unknown_failure").strip() or "unknown_failure"
        self.failure_counts[normalized] += 1
        normalized_fields = tuple(
            dict.fromkeys(str(field).strip() for field in fields if field is not None and str(field).strip())
        )
        if normalized_fields:
            self.failure_field_counts.setdefault(normalized, Counter()).update(normalized_fields)

    def to_metrics(self) -> dict[str, Any]:
        transform_success_rates = {
            transform: self.transform_counts.get(transform, 0) / attempt_count
            for transform, attempt_count in self.transform_attempt_counts.items()
            if attempt_count > 0
        }
        return {
            "quality_polish_enabled": bool(self.enabled),
            "quality_polish_parent_count": int(self.eligible_parent_count),
            "quality_polish_recent_completed_parent_count": int(self.parent_count),
            "quality_polish_scanned_parent_count": int(self.scanned_parent_count),
            "quality_polish_saturated_parent_count": int(self.saturated_parent_count),
            "quality_polish_generated": int(self.generated_count),
            "quality_polish_attempt_count": int(self.attempt_count),
            "quality_polish_success_count": int(self.success_count),
            "quality_polish_selected": int(self.selected_count),
            "quality_polish_top_parent_quality": self.top_parent_quality,
            "quality_polish_transform_counts": dict(self.transform_counts),
            "quality_polish_blocked_by_signature": int(self.skipped_used_signature),
            "quality_polish_blocked_by_recent_parent_transform": int(self.skipped_used_parent_transform),
            "quality_polish_skipped_used_signature": int(self.skipped_used_signature),
            "quality_polish_skipped_used_parent_transform": int(self.skipped_used_parent_transform),
            "quality_polish_skipped_existing_normalized": int(self.skipped_existing_normalized),
            "quality_polish_disabled_transform_counts": dict(self.disabled_transform_counts),
            "quality_polish_transform_cooldown_counts": dict(self.transform_cooldown_counts),
            "quality_polish_transform_attempt_counts": dict(self.transform_attempt_counts),
            "quality_polish_transform_success_rates": transform_success_rates,
            "quality_polish_transform_scores": dict(self.transform_scores),
            "quality_polish_cooldown_exempt_groups": list(self.cooldown_exempt_groups),
            "quality_polish_external_elite_seed_count": int(self.external_elite_seed_count),
            "quality_polish_external_elite_generated": int(self.external_elite_generated),
            "quality_polish_search_space_filter_blocked": int(self.search_space_filter_blocked),
            "quality_polish_failure_reason_counts": dict(self.failure_counts),
            "turnover_repair_generated": int(self.turnover_repair_generated),
            "turnover_repair_attempt_count": int(self.turnover_repair_attempt_count),
            "turnover_repair_success_count": int(self.turnover_repair_success_count),
            "turnover_repair_selected": int(self.turnover_repair_selected),
            "turnover_repair_transform_counts": dict(self.turnover_repair_transform_counts),
            "quality_polish_generation_total_ms": round(self.generation_total_ms, 3),
        }


@dataclass(slots=True)
class QualityPolishResult:
    candidates: list[AlphaCandidate]
    stats: QualityPolishStats


@dataclass(frozen=True, slots=True)
class _PolishParent:
    candidate: AlphaCandidate
    metrics: dict[str, Any]
    quality_score: float
    simulated_at: str


@dataclass(frozen=True, slots=True)
class _ExpressionVariant:
    expression: str
    transform: str
    transform_group: str
    priority: int


@dataclass(frozen=True, slots=True)
class _TransformProfile:
    scores: dict[str, float]
    cooldown_groups: set[str]


class QualityPolisher:
    def __init__(self, repository: SQLiteRepository) -> None:
        self.repository = repository

    def generate(
        self,
        *,
        config: QualityOptimizationConfig,
        adaptive_config: AdaptiveGenerationConfig,
        generation_config: GenerationConfig,
        registry: OperatorRegistry,
        field_registry: FieldRegistry,
        region_learning_context: RegionLearningContext,
        generation_guardrails: GenerationGuardrails,
        field_penalty_multipliers: dict[str, float] | None,
        blocked_fields: set[str],
        existing_normalized: set[str],
        run_id: str,
        round_index: int,
        count: int,
        allowed_fields: set[str] | None = None,
        lane_operator_allowlist: set[str] | None = None,
    ) -> QualityPolishResult:
        stats = QualityPolishStats(enabled=bool(config.enabled))
        if not config.enabled or count <= 0 or config.max_polish_candidates_per_round <= 0:
            return QualityPolishResult(candidates=[], stats=stats)

        started = time.perf_counter()
        raw_parent_count, parents = self._load_parents(
            config=config,
            run_id=run_id,
            blocked_fields=blocked_fields,
            allowed_fields=allowed_fields,
            lane_operator_allowlist=lane_operator_allowlist,
        )
        stats.parent_count = raw_parent_count
        elite_seed_lane_enabled = (
            bool(adaptive_config.elite_motifs.enabled)
            and bool(adaptive_config.elite_motifs.seed_expressions)
            and int(adaptive_config.elite_motifs.max_quality_polish_seeds_per_round) > 0
        )
        if len(parents) < int(config.min_completed_parent_count) and not elite_seed_lane_enabled:
            stats.generation_total_ms = (time.perf_counter() - started) * 1000.0
            return QualityPolishResult(candidates=[], stats=stats)
        scan_limit = max(
            int(config.max_polish_parents_per_round),
            int(config.max_polish_parents_per_round) * int(config.parent_scan_multiplier),
        )
        eligible = parents[:scan_limit]
        stats.eligible_parent_count = len(eligible)
        stats.scanned_parent_count = len(eligible)
        stats.top_parent_quality = eligible[0].quality_score if eligible else None

        target_count = min(int(config.max_polish_candidates_per_round), int(count))
        engine = AlphaGenerationEngine(
            config=generation_config,
            adaptive_config=adaptive_config,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=region_learning_context,
            generation_guardrails=generation_guardrails,
            field_penalty_multipliers=field_penalty_multipliers,
        )
        validation_ctx, prepare_ms, cache_hit = engine._get_or_build_validation_context()  # noqa: SLF001
        session = GenerationSessionStats(
            prepare_validation_context_ms=prepare_ms,
            validation_context_cache_hit=cache_hit,
        )
        candidates: list[AlphaCandidate] = []
        existing = set(existing_normalized)
        usage_keys = self.repository.list_quality_polish_usage_keys(run_id)
        used_signatures = set(usage_keys.get("signatures") or set())
        recent_parent_transform_counts = self._recent_parent_transform_counts(
            usage_rows=list(usage_keys.get("usage_rows") or []),
            current_round_index=int(round_index),
            config=config,
        )
        same_round_parent_transform_counts: Counter[str] = Counter()
        transform_profile = self._load_transform_profile(
            config=config,
            run_id=run_id,
            round_index=int(round_index),
        )
        stats.transform_scores = dict(transform_profile.scores)
        stats.cooldown_exempt_groups = list(config.cooldown_exempt_transform_groups or [])
        max_parent_transform_uses = int(config.max_parent_transform_uses_per_recent_window)

        if elite_seed_lane_enabled:
            elite_seed_target_count = _elite_seed_target_count(
                target_count=target_count,
                eligible_parent_count=len(eligible),
                min_completed_parent_count=int(config.min_completed_parent_count),
                max_quality_polish_seeds_per_round=int(
                    adaptive_config.elite_motifs.max_quality_polish_seeds_per_round
                ),
            )
            if elite_seed_target_count > 0:
                candidates.extend(
                    self._generate_elite_seed_candidates(
                        config=config,
                        adaptive_config=adaptive_config,
                        registry=registry,
                        engine=engine,
                        validation_ctx=validation_ctx,
                        session=session,
                        existing=existing,
                        run_id=run_id,
                        round_index=int(round_index),
                        target_count=elite_seed_target_count,
                        stats=stats,
                        allowed_fields=allowed_fields,
                        lane_operator_allowlist=lane_operator_allowlist,
                    )
                )

        for parent in eligible:
            if len(candidates) >= target_count:
                break
            parent_candidate_count_before = len(candidates)
            parent_skip_count_before = (
                stats.skipped_used_signature
                + stats.skipped_used_parent_transform
                + stats.skipped_existing_normalized
                + sum(stats.transform_cooldown_counts.values())
                + sum(stats.disabled_transform_counts.values())
            )
            variants = _unique_expression_variants(
                [
                    *self._turnover_repair_variants(
                        expression=parent.candidate.expression,
                        parent=parent,
                        registry=registry,
                        lookbacks=generation_config.lookbacks,
                    ),
                    *self._variant_expressions(
                        parent.candidate.expression,
                        registry=registry,
                        lookbacks=generation_config.lookbacks,
                        limit=int(config.variants_per_parent),
                        config=config,
                        stats=stats,
                        transform_scores=transform_profile.scores,
                        cooldown_groups=transform_profile.cooldown_groups,
                        existing_normalized=existing,
                    ),
                ]
            )
            for variant in variants:
                if len(candidates) >= target_count:
                    break
                normalized_variant = _normalize_variant_expression(variant.expression)
                polish_signature = _polish_signature(
                    run_id=run_id,
                    parent_alpha_id=parent.candidate.alpha_id,
                    transform=variant.transform,
                    normalized_expression=normalized_variant,
                )
                parent_transform_key = f"{parent.candidate.alpha_id}:{variant.transform}"
                if polish_signature in used_signatures:
                    stats.skipped_used_signature += 1
                    continue
                parent_transform_uses = (
                    recent_parent_transform_counts[parent_transform_key]
                    + same_round_parent_transform_counts[parent_transform_key]
                )
                if parent_transform_uses >= max_parent_transform_uses:
                    stats.skipped_used_parent_transform += 1
                    continue
                if normalized_variant in existing:
                    stats.skipped_existing_normalized += 1
                    stats.record_failure("duplicate_normalized_expression")
                    continue
                same_round_parent_transform_counts[parent_transform_key] += 1
                stats.attempt_count += 1
                stats.transform_attempt_counts[variant.transform_group] += 1
                if _is_turnover_repair_variant(variant):
                    stats.turnover_repair_attempt_count += 1
                    stats.turnover_repair_transform_counts[variant.transform] += 1
                session.record_attempt()
                metadata = self._variant_metadata(
                    parent=parent,
                    transform=variant.transform,
                    transform_group=variant.transform_group,
                    polish_signature=polish_signature,
                    parent_transform_key=parent_transform_key,
                    round_index=int(round_index),
                    transform_priority=variant.priority,
                    selection_prior_weight=float(config.selection_prior_weight),
                )
                result = engine._build_candidate_result(  # noqa: SLF001
                    expression=variant.expression,
                    mode="quality_polish",
                    parent_ids=(parent.candidate.alpha_id,),
                    generation_metadata=metadata,
                    validation_ctx=validation_ctx,
                )
                candidate = result.candidate
                if candidate is None:
                    stats.record_failure(result.failure_reason, fields=result.failure_fields)
                    session.record_failure(
                        result.failure_reason,
                        expression=variant.expression,
                        fields=result.failure_fields,
                    )
                    continue
                if candidate.normalized_expression in existing:
                    stats.skipped_existing_normalized += 1
                    stats.record_failure("duplicate_normalized_expression")
                    session.record_duplicate(
                        "duplicate_normalized_expression",
                        expression=candidate.normalized_expression,
                        mutation_mode="quality_polish",
                        motif=str(candidate.generation_metadata.get("motif") or ""),
                        operator_path=tuple(candidate.generation_metadata.get("operator_path") or candidate.operators_used),
                    )
                    continue
                existing.add(candidate.normalized_expression)
                used_signatures.add(polish_signature)
                candidates.append(candidate)
                stats.transform_counts[variant.transform_group] += 1
                stats.success_count += 1
                if _is_turnover_repair_variant(variant):
                    stats.turnover_repair_generated += 1
                    stats.turnover_repair_success_count += 1
                session.record_success()
            parent_skip_count_after = (
                stats.skipped_used_signature
                + stats.skipped_used_parent_transform
                + stats.skipped_existing_normalized
                + sum(stats.transform_cooldown_counts.values())
                + sum(stats.disabled_transform_counts.values())
            )
            if (
                len(candidates) == parent_candidate_count_before
                and (variants or parent_skip_count_after > parent_skip_count_before)
            ):
                stats.saturated_parent_count += 1

        if (
            len(candidates) < target_count
            and elite_seed_lane_enabled
            and stats.external_elite_generated <= 0
        ):
            candidates.extend(
                self._generate_elite_seed_candidates(
                    config=config,
                    adaptive_config=adaptive_config,
                    registry=registry,
                    engine=engine,
                    validation_ctx=validation_ctx,
                    session=session,
                    existing=existing,
                    run_id=run_id,
                    round_index=int(round_index),
                    target_count=target_count - len(candidates),
                    stats=stats,
                    allowed_fields=allowed_fields,
                    lane_operator_allowlist=lane_operator_allowlist,
                )
            )

        stats.generated_count = len(candidates)
        stats.generation_total_ms = (time.perf_counter() - started) * 1000.0
        return QualityPolishResult(candidates=candidates, stats=stats)

    def _generate_elite_seed_candidates(
        self,
        *,
        config: QualityOptimizationConfig,
        adaptive_config: AdaptiveGenerationConfig,
        registry: OperatorRegistry,
        engine: AlphaGenerationEngine,
        validation_ctx,
        session: GenerationSessionStats,
        existing: set[str],
        run_id: str,
        round_index: int,
        target_count: int,
        stats: QualityPolishStats,
        allowed_fields: set[str] | None,
        lane_operator_allowlist: set[str] | None,
    ) -> list[AlphaCandidate]:
        variants = build_elite_seed_variants(
            adaptive_config.elite_motifs,
            registry=registry,
            existing_normalized=existing,
        )
        stats.external_elite_seed_count = min(
            len(adaptive_config.elite_motifs.seed_expressions or []),
            int(adaptive_config.elite_motifs.max_quality_polish_seeds_per_round),
        )
        if target_count <= 0 or not variants:
            return []

        candidates: list[AlphaCandidate] = []
        for variant in variants:
            if len(candidates) >= target_count:
                break
            normalized_variant = _normalize_variant_expression(variant.expression)
            if not normalized_variant or normalized_variant in existing:
                stats.skipped_existing_normalized += 1
                stats.record_failure("duplicate_normalized_expression")
                continue
            if not _expression_allowed_by_search_space(
                variant.expression,
                allowed_fields=allowed_fields,
                lane_operator_allowlist=lane_operator_allowlist,
            ):
                stats.search_space_filter_blocked += 1
                stats.record_failure("search_space_filter_blocked")
                continue
            polish_signature = _polish_signature(
                run_id=run_id,
                parent_alpha_id=variant.seed_id,
                transform=variant.variant,
                normalized_expression=normalized_variant,
            )
            stats.attempt_count += 1
            stats.transform_attempt_counts["elite_seed_polish"] += 1
            session.record_attempt()
            metadata = {
                "generation_mode": "quality_polish",
                "generation_source": "quality_polish",
                "mutation_mode": "quality_polish",
                "motif": "elite_seed_polish",
                "template_name": "elite_seed_polish",
                "polish_transform": variant.variant,
                "polish_transform_group": "elite_seed_polish",
                "polish_signature": polish_signature,
                "polish_parent_transform_key": f"{variant.seed_id}:{variant.variant}",
                "polish_round_index": int(round_index),
                "polish_transform_priority": 8,
                "quality_polish_prior": float(config.selection_prior_weight),
                "elite_seed_id": variant.seed_id,
                "elite_seed_variant": variant.variant,
                "elite_seed_similarity": round(float(variant.similarity), 6),
                "elite_seed_similarity_penalty": round(
                    max(
                        0.0,
                        (
                            float(variant.similarity)
                            - float(adaptive_config.elite_motifs.clone_similarity_threshold)
                        )
                        / max(
                            1e-9,
                            1.0 - float(adaptive_config.elite_motifs.clone_similarity_threshold),
                        ),
                    ),
                    6,
                ),
                "elite_motif_match_score": round(float(variant.match_score), 6),
                "elite_motif_ids": [variant.seed_id],
                "parent_refs": [],
            }
            result = engine._build_candidate_result(  # noqa: SLF001
                expression=variant.expression,
                mode="quality_polish",
                parent_ids=(),
                generation_metadata=metadata,
                validation_ctx=validation_ctx,
            )
            candidate = result.candidate
            if candidate is None:
                stats.record_failure(result.failure_reason, fields=result.failure_fields)
                session.record_failure(
                    result.failure_reason,
                    expression=variant.expression,
                    fields=result.failure_fields,
                )
                continue
            if candidate.normalized_expression in existing:
                stats.skipped_existing_normalized += 1
                stats.record_failure("duplicate_normalized_expression")
                session.record_duplicate(
                    "duplicate_normalized_expression",
                    expression=candidate.normalized_expression,
                    mutation_mode="quality_polish",
                    motif="elite_seed_polish",
                    operator_path=tuple(candidate.operators_used),
                )
                continue
            existing.add(candidate.normalized_expression)
            candidates.append(candidate)
            stats.transform_counts["elite_seed_polish"] += 1
            stats.external_elite_generated += 1
            stats.success_count += 1
            session.record_success()
        return candidates

    def _load_parents(
        self,
        *,
        config: QualityOptimizationConfig,
        run_id: str,
        blocked_fields: set[str],
        allowed_fields: set[str] | None,
        lane_operator_allowlist: set[str] | None,
    ) -> tuple[int, list[_PolishParent]]:
        rows = self.repository.list_quality_polish_parent_rows(
            run_id=run_id,
            limit=int(config.lookback_completed_results),
        )
        parents: list[_PolishParent] = []
        for row in rows:
            parent = self._row_to_parent(row)
            if parent is None:
                continue
            parent_fields = _candidate_field_names(parent.candidate)
            if blocked_fields and not blocked_fields.isdisjoint(parent_fields):
                continue
            if (
                allowed_fields is not None
                and parent_fields
                and not parent_fields.issubset(allowed_fields)
            ):
                continue
            if not _candidate_operators_allowed_for_lane(
                parent.candidate,
                lane_operator_allowlist=lane_operator_allowlist,
            ):
                continue
            if not self._passes_thresholds(parent.metrics, config=config):
                continue
            parents.append(parent)
        return len(rows), sorted(parents, key=lambda item: (-item.quality_score, item.simulated_at, item.candidate.alpha_id))

    def _row_to_parent(self, row: dict[str, Any]) -> _PolishParent | None:
        if str(row.get("result_rejection_reason") or "").strip():
            return None
        metrics = {
            "fitness": _to_float(row.get("result_fitness")),
            "sharpe": _to_float(row.get("result_sharpe")),
            "turnover": _to_float(row.get("result_turnover")),
            "drawdown": _to_float(row.get("result_drawdown")),
            "returns": _to_float(row.get("result_returns")),
            "margin": _to_float(row.get("result_margin")),
            "submission_eligible": _to_optional_bool(row.get("result_submission_eligible")),
        }
        alpha_record = AlphaRecord(
            run_id=str(row.get("run_id") or ""),
            alpha_id=str(row.get("alpha_id") or ""),
            expression=str(row.get("expression") or ""),
            normalized_expression=str(row.get("normalized_expression") or ""),
            generation_mode=str(row.get("generation_mode") or ""),
            template_name=str(row.get("template_name") or ""),
            fields_used_json=str(row.get("fields_used_json") or "[]"),
            operators_used_json=str(row.get("operators_used_json") or "[]"),
            depth=int(row.get("depth") or 0),
            generation_metadata=str(row.get("generation_metadata") or "{}"),
            complexity=int(row.get("complexity") or 0),
            created_at=str(row.get("created_at") or ""),
            status=str(row.get("alpha_status") or ""),
            structural_signature_json=str(row.get("structural_signature_json") or "{}"),
        )
        try:
            candidate = alpha_candidate_from_record(alpha_record)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return _PolishParent(
            candidate=candidate,
            metrics=metrics,
            quality_score=quality_parent_score(metrics),
            simulated_at=str(row.get("result_simulated_at") or ""),
        )

    @staticmethod
    def _passes_thresholds(metrics: dict[str, float], *, config: QualityOptimizationConfig) -> bool:
        return (
            metrics["fitness"] >= float(config.min_parent_fitness)
            and metrics["sharpe"] >= float(config.min_parent_sharpe)
            and metrics["turnover"] <= float(config.max_parent_turnover)
            and metrics["drawdown"] <= float(config.max_parent_drawdown)
        )

    @staticmethod
    def _recent_parent_transform_counts(
        *,
        usage_rows: list[dict[str, Any]],
        current_round_index: int,
        config: QualityOptimizationConfig,
    ) -> Counter[str]:
        if int(config.parent_transform_recent_rounds) <= 0:
            return Counter()
        lower_bound = int(current_round_index) - int(config.parent_transform_recent_rounds)
        counts: Counter[str] = Counter()
        for row in usage_rows:
            row_round = _optional_int(row.get("polish_round_index"))
            if row_round is None or row_round < lower_bound:
                continue
            parent_transform_key = str(row.get("polish_parent_transform_key") or "").strip()
            if parent_transform_key:
                counts[parent_transform_key] += 1
        return counts

    def _load_transform_profile(
        self,
        *,
        config: QualityOptimizationConfig,
        run_id: str,
        round_index: int,
    ) -> _TransformProfile:
        if int(config.transform_score_lookback_rounds) <= 0:
            return _TransformProfile(scores={}, cooldown_groups=set())
        rows = self.repository.list_recent_generation_stage_metrics(
            run_id,
            limit=int(config.transform_score_lookback_rounds),
            before_round_index=int(round_index),
        )
        cooldown_exempt_groups = {
            str(item).strip()
            for item in (config.cooldown_exempt_transform_groups or [])
            if str(item).strip()
        }
        attempts: Counter[str] = Counter()
        successes: Counter[str] = Counter()
        for row in rows:
            metrics = _decode_json_object(row.get("metrics_json"))
            attempt_counts = _decode_json_object(metrics.get("quality_polish_transform_attempt_counts"))
            success_counts = _decode_json_object(metrics.get("quality_polish_transform_counts"))
            success_rates = _decode_json_object(metrics.get("quality_polish_transform_success_rates"))
            for transform_group, raw_count in attempt_counts.items():
                attempt_count = _to_non_negative_int(raw_count)
                if attempt_count <= 0:
                    continue
                group = str(transform_group or "").strip()
                if not group:
                    continue
                attempts[group] += attempt_count
                raw_success = success_counts.get(group)
                if raw_success is None:
                    raw_success = float(success_rates.get(group) or 0.0) * attempt_count
                successes[group] += max(0.0, _to_float(raw_success))
        scores: dict[str, float] = {}
        cooldown_groups: set[str] = set()
        for group, attempt_count in attempts.items():
            if attempt_count <= 0:
                continue
            score = max(0.0, min(1.0, float(successes[group]) / float(attempt_count)))
            scores[group] = round(score, 6)
            if (
                group not in cooldown_exempt_groups
                and attempt_count >= int(config.transform_cooldown_min_attempts)
                and score < float(config.transform_cooldown_success_rate_floor)
            ):
                cooldown_groups.add(group)
        return _TransformProfile(scores=scores, cooldown_groups=cooldown_groups)

    def _variant_expressions(
        self,
        expression: str,
        *,
        registry: OperatorRegistry,
        lookbacks: list[int],
        limit: int,
        config: QualityOptimizationConfig,
        stats: QualityPolishStats,
        transform_scores: dict[str, float],
        cooldown_groups: set[str],
        existing_normalized: set[str],
    ) -> list[_ExpressionVariant]:
        try:
            root = parse_expression(expression)
        except ValueError:
            return []

        variants: list[_ExpressionVariant] = []
        seen: set[str] = set()
        transform_counts: Counter[str] = Counter()
        max_by_transform = dict(config.max_variants_per_parent_by_transform or {})
        enabled_transforms = set(config.enabled_transforms or [])
        disabled_transforms = set(config.disabled_transforms or [])

        parent_normalized = _normalize_variant_expression(expression)

        def add(expr: str, transform: str, *, group: str, priority: int) -> None:
            if group in cooldown_groups or transform in cooldown_groups:
                stats.transform_cooldown_counts[group] += 1
                return
            if group in disabled_transforms or transform in disabled_transforms:
                stats.disabled_transform_counts[group] += 1
                return
            if enabled_transforms and group not in enabled_transforms and transform not in enabled_transforms:
                stats.disabled_transform_counts[group] += 1
                return
            if transform_counts[group] >= int(max_by_transform.get(group, limit) or limit):
                return
            normalized = _normalize_variant_expression(expr)
            if (
                not normalized
                or normalized == parent_normalized
                or normalized in existing_normalized
                or normalized in seen
                or _is_redundant_cross_sectional_wrapper(expr)
                or len(variants) >= limit
            ):
                return
            seen.add(normalized)
            variants.append(
                _ExpressionVariant(
                    expression=expr.strip(),
                    transform=transform,
                    transform_group=group,
                    priority=priority,
                )
            )
            transform_counts[group] += 1

        root_expr = to_expression(root)
        if registry.contains("rank") and not _is_root_call(root, "rank"):
            add(f"rank({root_expr})", "wrap_rank", group="wrap_rank", priority=1)
        if registry.contains("zscore") and not _is_root_call(root, "zscore"):
            add(f"zscore({root_expr})", "wrap_zscore", group="wrap_zscore", priority=2)
        cleaned = _cleanup_root_wrapper(root)
        if cleaned is not None:
            add(
                to_expression(cleaned),
                "cleanup_redundant_wrapper",
                group="cleanup_redundant_wrapper",
                priority=4,
            )

        preferred_windows = _preferred_windows(lookbacks)
        base_nodes = _smooth_base_nodes(_unwrap_outer_cross_sectional(root))
        if registry.contains("ts_mean"):
            for base_node in base_nodes:
                base_expr = to_expression(base_node)
                for window in preferred_windows:
                    inner = f"ts_mean({base_expr},{window})"
                    if registry.contains("rank"):
                        add(
                            f"rank({inner})",
                            f"smooth_ts_mean_rank_{window}",
                            group="smooth_ts_mean",
                            priority=5,
                        )
                    if registry.contains("zscore"):
                        add(
                            f"zscore({inner})",
                            f"smooth_ts_mean_zscore_{window}",
                            group="smooth_ts_mean",
                            priority=5,
                        )
        if registry.contains("ts_decay_linear") and registry.contains("rank"):
            for base_node in base_nodes:
                base_expr = to_expression(base_node)
                for window in preferred_windows:
                    inner = f"ts_decay_linear({base_expr},{window})"
                    add(
                        f"rank({inner})",
                        f"smooth_ts_decay_linear_rank_{window}",
                        group="smooth_ts_decay_linear",
                        priority=6,
                    )
                    if registry.contains("zscore"):
                        add(
                            f"zscore({inner})",
                            f"smooth_ts_decay_linear_zscore_{window}",
                            group="smooth_ts_decay_linear",
                            priority=6,
                        )
        if registry.contains("ts_rank") and registry.contains("rank"):
            base_expr = to_expression(base_nodes[0]) if base_nodes else to_expression(_unwrap_outer_cross_sectional(root))
            for window in preferred_windows:
                inner = f"ts_rank({base_expr},{window})"
                add(
                    f"rank({inner})",
                    f"smooth_ts_rank_{window}",
                    group="smooth_ts_rank",
                    priority=7,
                )

        for variant_node, transform in _window_perturbations(
            root,
            lookbacks=lookbacks,
            neighbor_count=int(config.window_perturb_neighbor_count),
        ):
            add(to_expression(variant_node), transform, group="window_perturb", priority=3)

        return sorted(
            variants,
            key=lambda item: (
                item.priority,
                -float(transform_scores.get(item.transform_group, 0.0)),
                item.transform,
                item.expression,
            ),
        )[:limit]

    def _turnover_repair_variants(
        self,
        *,
        expression: str,
        parent: _PolishParent,
        registry: OperatorRegistry,
        lookbacks: list[int],
    ) -> list[_ExpressionVariant]:
        source_turnover = _to_float(parent.metrics.get("turnover"))
        if source_turnover <= 0.70:
            return []
        try:
            root = parse_expression(expression)
        except ValueError:
            return []
        base_node = _unwrap_outer_cross_sectional(root)
        base_expr = to_expression(base_node)
        preferred_windows = [window for window in _preferred_windows(lookbacks) if window > 0]
        variants: list[_ExpressionVariant] = []
        if registry.contains("ts_mean") and preferred_windows:
            variants.append(
                _ExpressionVariant(
                    expression=f"rank(ts_mean({base_expr},{preferred_windows[0]}))",
                    transform=f"wrap_ts_mean_{preferred_windows[0]}",
                    transform_group="turnover_repair",
                    priority=0,
                )
            )
        if registry.contains("ts_decay_linear") and preferred_windows:
            variants.append(
                _ExpressionVariant(
                    expression=f"rank(ts_decay_linear({base_expr},{preferred_windows[0]}))",
                    transform=f"wrap_ts_decay_linear_{preferred_windows[0]}",
                    transform_group="turnover_repair",
                    priority=0,
                )
            )
        extension = _extend_existing_smoothing(base_node, preferred_windows)
        if extension is not None:
            variants.append(
                _ExpressionVariant(
                    expression=f"rank({extension})",
                    transform="extend_existing_smoothing",
                    transform_group="turnover_repair",
                    priority=0,
                )
            )
        return _unique_expression_variants(variants)[:2]

    @staticmethod
    def _variant_metadata(
        *,
        parent: _PolishParent,
        transform: str,
        transform_group: str,
        polish_signature: str,
        parent_transform_key: str,
        round_index: int,
        transform_priority: int,
        selection_prior_weight: float,
    ) -> dict[str, Any]:
        family_signature = str(parent.candidate.generation_metadata.get("family_signature") or "")
        return {
            "generation_mode": "quality_polish",
            "generation_source": "quality_polish",
            "mutation_mode": "quality_polish",
            "motif": "quality_polish",
            "template_name": "quality_polish",
            "polish_transform": transform,
            "polish_transform_group": transform_group,
            "polish_signature": polish_signature,
            "polish_parent_transform_key": parent_transform_key,
            "polish_round_index": int(round_index),
            "polish_transform_priority": int(transform_priority),
            "polish_parent_alpha_id": parent.candidate.alpha_id,
            "polish_parent_quality_score": float(parent.quality_score),
            "polish_parent_metrics": dict(parent.metrics),
            "quality_polish_prior": float(selection_prior_weight),
            "parent_refs": [
                {
                    "run_id": parent.candidate.generation_metadata.get("run_id", ""),
                    "alpha_id": parent.candidate.alpha_id,
                    "family_signature": family_signature,
                    "relation": "quality_polish",
                }
            ],
            **(
                {
                    "repair_reason": "turnover",
                    "repair_transform": transform,
                    "repair_source_turnover": float(parent.metrics.get("turnover") or 0.0),
                    "repair_target_profile": "low_turnover",
                }
                if transform_group == "turnover_repair"
                else {}
            ),
        }


def quality_parent_score(metrics: dict[str, Any]) -> float:
    return MultiObjectiveQualityScorer.score(
        metrics=metrics,
        submission_eligible=metrics.get("submission_eligible"),
        rejection_reason=None,
        status="completed",
    )


def _polish_signature(
    *,
    run_id: str,
    parent_alpha_id: str,
    transform: str,
    normalized_expression: str,
) -> str:
    payload = "\x1f".join((str(run_id), str(parent_alpha_id), str(transform), str(normalized_expression)))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _normalize_variant_expression(expression: str) -> str:
    try:
        return to_expression(parse_expression(expression))
    except ValueError:
        return str(expression or "").strip()


def _window_perturbations(
    root: ExprNode,
    *,
    lookbacks: list[int],
    neighbor_count: int,
) -> list[tuple[ExprNode, str]]:
    variants: list[tuple[ExprNode, str]] = []
    for index, replacement in enumerate(
        _replace_window_nodes(root, lookbacks=lookbacks, neighbor_count=neighbor_count)
    ):
        variants.append((replacement, f"window_perturb_{index + 1}"))
    return variants


def _replace_window_nodes(
    node: ExprNode,
    *,
    lookbacks: list[int],
    neighbor_count: int,
) -> list[ExprNode]:
    replacements: list[ExprNode] = []
    if isinstance(node, FunctionCallNode):
        args = list(node.args)
        if node.name in WINDOWED_OPERATORS and len(args) >= 2 and isinstance(args[1], NumberNode):
            current = int(args[1].value)
            for window in _neighbor_windows(current, lookbacks, limit=neighbor_count):
                updated_args = list(args)
                updated_args[1] = NumberNode(float(window))
                replacements.append(FunctionCallNode(name=node.name, args=tuple(updated_args)))
        for arg_index, arg in enumerate(args):
            for child_replacement in _replace_window_nodes(
                arg,
                lookbacks=lookbacks,
                neighbor_count=neighbor_count,
            ):
                updated_args = list(args)
                updated_args[arg_index] = child_replacement
                replacements.append(FunctionCallNode(name=node.name, args=tuple(updated_args)))
    return replacements


def _neighbor_windows(current: int, lookbacks: list[int], *, limit: int = 2) -> list[int]:
    candidates = [int(value) for value in lookbacks if int(value) > 0 and int(value) != current]
    return sorted(candidates, key=lambda value: (abs(value - current), value))[: max(1, int(limit))]


def _preferred_windows(lookbacks: list[int]) -> list[int]:
    normalized = [int(value) for value in lookbacks if int(value) > 0]
    return list(dict.fromkeys(normalized[:3] or [10, 20, 60]))


def _is_turnover_repair_variant(variant: _ExpressionVariant) -> bool:
    return str(variant.transform_group) == "turnover_repair"


def _elite_seed_target_count(
    *,
    target_count: int,
    eligible_parent_count: int,
    min_completed_parent_count: int,
    max_quality_polish_seeds_per_round: int,
) -> int:
    if target_count <= 0 or max_quality_polish_seeds_per_round <= 0:
        return 0
    if eligible_parent_count < max(1, min_completed_parent_count):
        return int(target_count)
    reserved = max(1, int(target_count) // 4)
    return min(int(target_count), int(max_quality_polish_seeds_per_round), reserved)


def _unique_expression_variants(variants: list[_ExpressionVariant]) -> list[_ExpressionVariant]:
    seen: set[str] = set()
    unique: list[_ExpressionVariant] = []
    for variant in variants:
        key = str(variant.expression).strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(variant)
    return unique


def _is_redundant_cross_sectional_wrapper(expression: str) -> bool:
    try:
        root = parse_expression(expression)
    except ValueError:
        return False
    if not isinstance(root, FunctionCallNode) or root.name not in {"rank", "zscore"} or len(root.args) != 1:
        return False
    child = root.args[0]
    return isinstance(child, FunctionCallNode) and child.name == root.name and len(child.args) == 1


def _candidate_operators_allowed_for_lane(
    candidate: AlphaCandidate,
    *,
    lane_operator_allowlist: set[str] | None,
) -> bool:
    if lane_operator_allowlist is None:
        return True
    _, parsed_operators = expression_fields_and_operators(candidate.expression)
    operators = parsed_operators or _policy_operator_names(candidate.operators_used)
    return not operators or operators.issubset(lane_operator_allowlist)


def _candidate_field_names(candidate: AlphaCandidate) -> set[str]:
    parsed_fields, _ = expression_fields_and_operators(candidate.expression)
    return parsed_fields or {str(field).strip() for field in candidate.fields_used if str(field).strip()}


def _policy_operator_names(operators: Iterable[str]) -> set[str]:
    return {
        str(operator).strip()
        for operator in operators
        if str(operator).strip()
        and not str(operator).strip().startswith(("binary:", "unary:"))
    }


def _unwrap_outer_cross_sectional(node: ExprNode) -> ExprNode:
    current = node
    while isinstance(current, FunctionCallNode) and current.name in {"rank", "zscore"} and len(current.args) == 1:
        current = current.args[0]
    return current


_CROSS_SECTIONAL_SMOOTHING_OPERATORS = frozenset(
    {"rank", "zscore", "quantile", "group_rank", "group_zscore", "group_neutralize", "normalize"}
)


def _smooth_base_nodes(node: ExprNode) -> list[ExprNode]:
    candidates: list[ExprNode] = []

    def visit(current: ExprNode) -> None:
        if isinstance(current, NumberNode):
            return
        if _contains_field(current) and not _contains_cross_sectional_operator(current):
            candidates.append(current)
        for child in iter_child_nodes(current):
            visit(child)

    visit(node)
    if not candidates and _contains_field(node):
        candidates.append(node)

    unique: dict[str, ExprNode] = {}
    for candidate in sorted(candidates, key=lambda item: (node_complexity(item), to_expression(item))):
        unique.setdefault(to_expression(candidate), candidate)
    return list(unique.values())[:4]


def _contains_field(node: ExprNode) -> bool:
    if isinstance(node, IdentifierNode):
        return True
    return any(_contains_field(child) for child in iter_child_nodes(node))


def _contains_cross_sectional_operator(node: ExprNode) -> bool:
    if isinstance(node, FunctionCallNode) and node.name in _CROSS_SECTIONAL_SMOOTHING_OPERATORS:
        return True
    if isinstance(node, (BinaryOpNode, FunctionCallNode, UnaryOpNode)):
        return any(_contains_cross_sectional_operator(child) for child in iter_child_nodes(node))
    return False


def _extend_existing_smoothing(node: ExprNode, lookbacks: list[int]) -> str | None:
    if not isinstance(node, FunctionCallNode) or len(node.args) < 2:
        return None
    if node.name not in {"ts_mean", "ts_decay_linear"}:
        return None
    if not isinstance(node.args[1], NumberNode):
        return None
    current_window = int(node.args[1].value)
    larger_windows = [window for window in lookbacks if int(window) > current_window]
    if not larger_windows:
        return None
    updated_args = list(node.args)
    updated_args[1] = NumberNode(float(max(larger_windows)))
    return to_expression(FunctionCallNode(name=node.name, args=tuple(updated_args)))


def _cleanup_root_wrapper(root: ExprNode) -> ExprNode | None:
    if isinstance(root, FunctionCallNode) and root.name in {"rank", "zscore"} and len(root.args) == 1:
        return root.args[0]
    return None


def _is_root_call(root: ExprNode, name: str) -> bool:
    return isinstance(root, FunctionCallNode) and root.name == name


def _expression_allowed_by_search_space(
    expression: str,
    *,
    allowed_fields: set[str] | None,
    lane_operator_allowlist: set[str] | None,
) -> bool:
    if allowed_fields is None and lane_operator_allowlist is None:
        return True
    fields, operators = expression_fields_and_operators(expression)
    if allowed_fields is not None and fields and not fields.issubset(allowed_fields):
        return False
    if lane_operator_allowlist is not None and operators and not operators.issubset(lane_operator_allowlist):
        return False
    return True


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _decode_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _to_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        normalized = str(value).strip().lower()
        if normalized in {"true", "yes", "y"}:
            return True
        if normalized in {"false", "no", "n"}:
            return False
        return None
