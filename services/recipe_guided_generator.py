from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from core.config import AdaptiveGenerationConfig, GenerationConfig, RecipeGenerationConfig
from core.quality_score import MultiObjectiveQualityScorer
from data.field_registry import FieldRegistry, FieldSpec
from features.registry import OperatorRegistry
from generator.engine import AlphaCandidate, AlphaGenerationEngine
from generator.guardrails import GenerationGuardrails
from memory.pattern_memory import RegionLearningContext
from services.evaluation_service import alpha_candidate_from_record
from storage.repository import SQLiteRepository

_ALLOWED_FIELD_CATEGORIES = frozenset({"fundamental", "analyst", "model"})
_RECIPE_TEMPLATE_BY_FAMILY = {
    "fundamental_quality": "recipe_fundamental_quality",
    "accrual_vs_cashflow": "recipe_accrual_vs_cashflow",
    "value_vs_growth": "recipe_value_vs_growth",
    "revision_surprise": "recipe_revision_surprise",
}
_RECIPE_KEYWORDS = {
    "fundamental_quality": {
        "primary": (
            "margin",
            "profitability",
            "profit",
            "roe",
            "roa",
            "cash",
            "cfo",
            "fcf",
            "ebit",
            "ebitda",
            "quality",
        ),
    },
    "accrual_vs_cashflow": {
        "cashflow": (
            "cfo",
            "ocf",
            "cash",
            "free_cash_flow",
            "fcf",
            "operating_cash",
        ),
        "accrual": (
            "accrual",
            "receivable",
            "working_capital",
            "inventory",
            "deferred",
            "reserve",
        ),
    },
    "value_vs_growth": {
        "value": (
            "book",
            "earnings_yield",
            "cashflow_yield",
            "sales_yield",
            "dividend",
            "value",
        ),
        "growth": (
            "growth",
            "revision",
            "estimate",
            "forecast",
            "expected",
            "surprise",
        ),
    },
    "revision_surprise": {
        "primary": (
            "revision",
            "surprise",
            "estimate",
            "forecast",
            "analyst",
        ),
    },
}


@dataclass(slots=True)
class RecipeGuidedStats:
    enabled: bool = True
    generated_count: int = 0
    selected_count: int = 0
    attempt_count: int = 0
    success_count: int = 0
    bucket_counts: Counter[str] = field(default_factory=Counter)
    selected_by_bucket: Counter[str] = field(default_factory=Counter)
    template_counts: Counter[str] = field(default_factory=Counter)
    failure_counts: Counter[str] = field(default_factory=Counter)
    parented_count: int = 0
    parentless_count: int = 0
    budget_allocations: dict[str, int] = field(default_factory=dict)
    yield_scores: dict[str, float] = field(default_factory=dict)
    floor_hits: Counter[str] = field(default_factory=Counter)
    generation_total_ms: float = 0.0

    def record_failure(self, reason: str | None) -> None:
        normalized = str(reason or "unknown_failure").strip() or "unknown_failure"
        if normalized == "expression_validation_failed":
            normalized = "validation_unknown_error"
        self.failure_counts[normalized] += 1

    def to_metrics(self) -> dict[str, Any]:
        return {
            "recipe_guided_generated": int(self.generated_count),
            "recipe_guided_attempt_count": int(self.attempt_count),
            "recipe_guided_success_count": int(self.success_count),
            "recipe_guided_selected": int(self.selected_count),
            "recipe_guided_bucket_counts": dict(self.bucket_counts),
            "recipe_guided_selected_by_bucket": dict(self.selected_by_bucket),
            "recipe_guided_template_counts": dict(self.template_counts),
            "recipe_guided_failure_reason_counts": dict(self.failure_counts),
            "recipe_guided_parented_count": int(self.parented_count),
            "recipe_guided_parentless_count": int(self.parentless_count),
            "recipe_bucket_budget_allocations": dict(self.budget_allocations),
            "recipe_bucket_yield_scores": dict(self.yield_scores),
            "recipe_bucket_floor_hits": dict(self.floor_hits),
            "recipe_guided_generation_total_ms": round(self.generation_total_ms, 3),
        }


@dataclass(frozen=True, slots=True)
class RecipeGuidedResult:
    candidates: list[AlphaCandidate]
    stats: RecipeGuidedStats


@dataclass(frozen=True, slots=True)
class _RecipeBucket:
    search_bucket_id: str
    recipe_family: str
    objective_profile: str
    template_name: str


@dataclass(frozen=True, slots=True)
class _RecipeParent:
    candidate: AlphaCandidate
    metrics: dict[str, Any]
    quality_score: float


@dataclass(frozen=True, slots=True)
class _RecipeDraft:
    expression: str
    recipe_variant: str


class RecipeGuidedGenerator:
    def __init__(self, repository: SQLiteRepository) -> None:
        self.repository = repository

    def generate(
        self,
        *,
        config: RecipeGenerationConfig,
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
    ) -> RecipeGuidedResult:
        stats = RecipeGuidedStats(enabled=bool(config.enabled))
        if not config.enabled or count <= 0 or config.max_recipe_candidates_per_round <= 0:
            return RecipeGuidedResult(candidates=[], stats=stats)

        started = time.perf_counter()
        target_count = min(int(count), int(config.max_recipe_candidates_per_round))
        active_buckets = _active_buckets(config=config, round_index=round_index)
        if not active_buckets:
            return RecipeGuidedResult(candidates=[], stats=stats)

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
        _ = prepare_ms
        _ = cache_hit

        parents = self._load_parents(
            run_id=run_id,
            limit=int(config.lookback_completed_results),
            blocked_fields=blocked_fields,
        )
        bucket_priors = self._load_bucket_priors(
            run_id=run_id,
            before_round_index=round_index,
            config=config,
        )
        bucket_yield_scores = self._load_bucket_yield_scores(
            run_id=run_id,
            before_round_index=round_index,
            config=config,
            active_buckets=active_buckets,
        )
        existing = set(existing_normalized)
        candidates: list[AlphaCandidate] = []
        planned_targets = self._planned_bucket_targets(
            config=config,
            target_count=target_count,
            active_buckets=active_buckets,
            bucket_yield_scores=bucket_yield_scores,
        )
        stats.budget_allocations = {
            bucket.search_bucket_id: int(planned_targets.get(bucket.search_bucket_id, 0))
            for bucket in active_buckets
            if int(planned_targets.get(bucket.search_bucket_id, 0)) > 0
        }
        stats.yield_scores = {
            bucket.search_bucket_id: float(bucket_yield_scores.get(bucket.search_bucket_id, _neutral_yield_score()))
            for bucket in active_buckets
        }
        if bool(config.dynamic_budget_enabled):
            for bucket_id, count in stats.budget_allocations.items():
                if int(count) >= max(1, int(config.bucket_exploration_floor)):
                    stats.floor_hits[bucket_id] += 1

        spill = 0
        for bucket in active_buckets:
            if len(candidates) >= target_count:
                break
            bucket_target = min(
                int(config.max_candidates_per_bucket),
                int(planned_targets.get(bucket.search_bucket_id, 0)) + spill,
                target_count - len(candidates),
            )
            if bucket_target <= 0:
                spill = 0
                continue
            before_bucket = len(candidates)
            bucket_candidates = self._generate_bucket_candidates(
                bucket=bucket,
                bucket_target=bucket_target,
                config=config,
                registry=registry,
                field_registry=field_registry,
                blocked_fields=blocked_fields,
                field_penalty_multipliers=field_penalty_multipliers or {},
                parents=parents,
                generation_config=generation_config,
                engine=engine,
                validation_ctx=validation_ctx,
                existing_normalized=existing,
                run_id=run_id,
                round_index=round_index,
                bucket_prior=bucket_priors.get(bucket.search_bucket_id, float(config.selection_prior_weight)),
                stats=stats,
            )
            candidates.extend(bucket_candidates)
            produced = len(candidates) - before_bucket
            spill = max(0, bucket_target - produced)

        stats.generation_total_ms = (time.perf_counter() - started) * 1000.0
        return RecipeGuidedResult(candidates=candidates, stats=stats)

    def _load_parents(
        self,
        *,
        run_id: str,
        limit: int,
        blocked_fields: set[str],
    ) -> list[_RecipeParent]:
        rows = self.repository.list_recipe_parent_rows(run_id=run_id, limit=limit)
        parents: list[_RecipeParent] = []
        for row in rows:
            alpha_record = SimpleNamespace(
                alpha_id=str(row.get("alpha_id") or ""),
                expression=str(row.get("expression") or ""),
                normalized_expression=str(row.get("normalized_expression") or ""),
                generation_mode=str(row.get("generation_mode") or ""),
                template_name=str(row.get("template_name") or ""),
                fields_used_json=str(row.get("fields_used_json") or "[]"),
                operators_used_json=str(row.get("operators_used_json") or "[]"),
                depth=int(row.get("depth") or 0),
                generation_metadata=str(row.get("generation_metadata") or "{}"),
                structural_signature_json=str(row.get("structural_signature_json") or "{}"),
                complexity=int(row.get("complexity") or 0),
                created_at=str(row.get("created_at") or ""),
                status=str(row.get("alpha_status") or "generated"),
            )
            candidate = alpha_candidate_from_record(alpha_record)
            if blocked_fields and any(field in blocked_fields for field in candidate.fields_used):
                continue
            metrics = {
                "fitness": _to_float(row.get("result_fitness")),
                "sharpe": _to_float(row.get("result_sharpe")),
                "turnover": _to_float(row.get("result_turnover")),
                "drawdown": _to_float(row.get("result_drawdown")),
                "returns": _to_float(row.get("result_returns")),
                "margin": _to_float(row.get("result_margin")),
                "submission_eligible": row.get("result_submission_eligible"),
                "rejection_reason": row.get("result_rejection_reason"),
                "status": row.get("result_status"),
            }
            quality_score = _to_float(row.get("result_quality_score"))
            if quality_score is None or abs(float(quality_score)) <= 1e-12:
                quality_score = MultiObjectiveQualityScorer.score_record(
                    SimpleNamespace(
                        fitness=metrics["fitness"],
                        sharpe=metrics["sharpe"],
                        turnover=metrics["turnover"],
                        drawdown=metrics["drawdown"],
                        returns=metrics["returns"],
                        margin=metrics["margin"],
                        submission_eligible=metrics["submission_eligible"],
                        rejection_reason=metrics["rejection_reason"],
                        status=metrics["status"],
                    )
                )
            parents.append(
                _RecipeParent(
                    candidate=candidate,
                    metrics=metrics,
                    quality_score=float(quality_score),
                )
            )
        return sorted(parents, key=lambda item: (-item.quality_score, item.candidate.alpha_id))

    def _load_bucket_priors(
        self,
        *,
        run_id: str,
        before_round_index: int,
        config: RecipeGenerationConfig,
    ) -> dict[str, float]:
        rows = self.repository.list_recipe_bucket_result_rows(
            run_id=run_id,
            before_round_index=int(before_round_index),
            lookback_rounds=int(config.yield_lookback_rounds),
        )
        bucket_scores: dict[str, list[float]] = {}
        for row in rows:
            if str(row.get("status") or "") != "completed":
                continue
            metadata = _decode_json_object(row.get("generation_metadata"))
            bucket_id = str(metadata.get("search_bucket_id") or "").strip()
            if not bucket_id:
                continue
            quality_score = _to_float(row.get("quality_score"))
            if quality_score is None or abs(float(quality_score)) <= 1e-12:
                quality_score = MultiObjectiveQualityScorer.score_record(
                    SimpleNamespace(
                        fitness=row.get("fitness"),
                        sharpe=row.get("sharpe"),
                        turnover=row.get("turnover"),
                        drawdown=row.get("drawdown"),
                        returns=row.get("returns"),
                        margin=row.get("margin"),
                        submission_eligible=row.get("submission_eligible"),
                        rejection_reason=row.get("rejection_reason"),
                        status=row.get("status"),
                    )
                )
            bucket_scores.setdefault(bucket_id, []).append(float(quality_score))
        priors: dict[str, float] = {}
        for bucket in _all_buckets(config):
            scores = bucket_scores.get(bucket.search_bucket_id, [])
            multiplier = 1.0
            if len(scores) >= int(config.min_bucket_support_for_penalty):
                avg_quality = sum(scores) / len(scores)
                if avg_quality > 0.05:
                    multiplier = 1.20
                elif avg_quality <= 0.0:
                    multiplier = 0.70
            priors[bucket.search_bucket_id] = float(config.selection_prior_weight) * multiplier
        return priors

    def _load_bucket_yield_scores(
        self,
        *,
        run_id: str,
        before_round_index: int,
        config: RecipeGenerationConfig,
        active_buckets: list[_RecipeBucket],
    ) -> dict[str, float]:
        result_rows = self.repository.list_recipe_bucket_result_rows(
            run_id=run_id,
            before_round_index=int(before_round_index),
            lookback_rounds=int(config.yield_lookback_rounds),
        )
        stage_metric_rows = self.repository.list_recent_generation_stage_metrics(
            run_id,
            limit=int(config.yield_lookback_rounds),
            before_round_index=int(before_round_index),
        )
        generated_by_bucket = Counter[str]()
        selected_by_bucket = Counter[str]()
        for row in stage_metric_rows:
            metrics = _decode_json_object(row.get("metrics_json"))
            generated_by_bucket.update(
                {
                    str(key): int(value or 0)
                    for key, value in _decode_json_object(metrics.get("recipe_guided_bucket_counts")).items()
                }
            )
            selected_by_bucket.update(
                {
                    str(key): int(value or 0)
                    for key, value in _decode_json_object(metrics.get("recipe_guided_selected_by_bucket")).items()
                }
            )

        completed_counts: Counter[str] = Counter()
        positive_quality_counts: Counter[str] = Counter()
        quality_sums: Counter[str] = Counter()
        for row in result_rows:
            if str(row.get("status") or "") != "completed":
                continue
            metadata = _decode_json_object(row.get("generation_metadata"))
            bucket_id = str(metadata.get("search_bucket_id") or "").strip()
            if not bucket_id:
                continue
            completed_counts[bucket_id] += 1
            quality_score = _to_float(row.get("quality_score"))
            if quality_score is None or abs(float(quality_score)) <= 1e-12:
                quality_score = MultiObjectiveQualityScorer.score_record(SimpleNamespace(**row))
            quality_sums[bucket_id] += float(quality_score)
            if float(quality_score) > 0.0:
                positive_quality_counts[bucket_id] += 1

        scores: dict[str, float] = {}
        for bucket in active_buckets:
            bucket_id = bucket.search_bucket_id
            generated_support = int(generated_by_bucket.get(bucket_id, 0))
            completed_support = int(completed_counts.get(bucket_id, 0))
            if (
                generated_support < int(config.dynamic_budget_min_generated_support)
                or completed_support < int(config.dynamic_budget_min_completed_support)
            ):
                scores[bucket_id] = _neutral_yield_score()
                continue
            selected_rate = float(selected_by_bucket.get(bucket_id, 0)) / max(1, generated_support)
            positive_quality_rate = float(positive_quality_counts.get(bucket_id, 0)) / max(1, completed_support)
            avg_quality_score = float(quality_sums.get(bucket_id, 0.0)) / max(1, completed_support)
            raw_score = _yield_score(
                selected_rate=selected_rate,
                positive_quality_rate=positive_quality_rate,
                avg_quality_score=avg_quality_score,
            )
            scores[bucket_id] = _adjusted_yield_score(
                raw_score=raw_score,
                strength=float(config.bucket_reallocation_strength),
            )
        return scores

    def _planned_bucket_targets(
        self,
        *,
        config: RecipeGenerationConfig,
        target_count: int,
        active_buckets: list[_RecipeBucket],
        bucket_yield_scores: dict[str, float],
    ) -> dict[str, int]:
        if target_count <= 0 or not active_buckets:
            return {}
        if not bool(config.dynamic_budget_enabled):
            targets = _planned_bucket_targets(
                bucket_count=len(active_buckets),
                target_count=target_count,
                max_candidates_per_bucket=int(config.max_candidates_per_bucket),
            )
            return {
                bucket.search_bucket_id: int(targets[index])
                for index, bucket in enumerate(active_buckets)
                if index < len(targets) and int(targets[index]) > 0
            }
        capacities = {
            bucket.search_bucket_id: int(config.max_candidates_per_bucket)
            for bucket in active_buckets
        }
        weights = {
            bucket.search_bucket_id: float(bucket_yield_scores.get(bucket.search_bucket_id, _neutral_yield_score()))
            for bucket in active_buckets
        }
        allocations = _allocate_integer_budget(
            total_budget=int(target_count),
            capacities=capacities,
            weights=weights,
            floor_targets={
                bucket.search_bucket_id: int(config.bucket_exploration_floor)
                for bucket in active_buckets
            },
        )
        return {bucket_id: int(value) for bucket_id, value in allocations.items() if int(value) > 0}

    def _generate_bucket_candidates(
        self,
        *,
        bucket: _RecipeBucket,
        bucket_target: int,
        config: RecipeGenerationConfig,
        registry: OperatorRegistry,
        field_registry: FieldRegistry,
        blocked_fields: set[str],
        field_penalty_multipliers: dict[str, float],
        parents: list[_RecipeParent],
        generation_config: GenerationConfig,
        engine: AlphaGenerationEngine,
        validation_ctx,
        existing_normalized: set[str],
        run_id: str,
        round_index: int,
        bucket_prior: float,
        stats: RecipeGuidedStats,
    ) -> list[AlphaCandidate]:
        if bucket_target <= 0:
            return []
        parent = self._select_parent_for_bucket(bucket=bucket, parents=parents)
        parent_fields = set(parent.candidate.fields_used) if parent is not None else set()
        field_pool = self._eligible_fields(
            field_registry=field_registry,
            blocked_fields=blocked_fields,
            field_penalty_multipliers=field_penalty_multipliers,
        )
        draft_bundle = self._recipe_drafts_for_bucket(
            bucket=bucket,
            field_pool=field_pool,
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
            registry=registry,
            generation_config=generation_config,
        )
        parent_used = bool(parent is not None and draft_bundle["parent_used"])
        parent_refs = []
        parent_ids: tuple[str, ...] = ()
        recipe_parent_alpha_id = ""
        if parent_used and parent is not None:
            family_signature = str(parent.candidate.generation_metadata.get("family_signature") or "")
            parent_refs = [
                {
                    "run_id": run_id,
                    "alpha_id": parent.candidate.alpha_id,
                    "family_signature": family_signature,
                }
            ]
            parent_ids = (parent.candidate.alpha_id,)
            recipe_parent_alpha_id = parent.candidate.alpha_id

        candidates: list[AlphaCandidate] = []
        for draft in draft_bundle["drafts"]:
            if len(candidates) >= bucket_target:
                break
            stats.attempt_count += 1
            metadata = {
                "generation_mode": "recipe_guided",
                "generation_source": "recipe_guided",
                "mutation_mode": "recipe_guided",
                "template_name": bucket.template_name,
                "motif": bucket.template_name,
                "search_bucket_id": bucket.search_bucket_id,
                "recipe_family": bucket.recipe_family,
                "dataset_family": "fundamental",
                "objective_profile": bucket.objective_profile,
                "recipe_variant": draft.recipe_variant,
                "recipe_parent_alpha_id": recipe_parent_alpha_id,
                "recipe_prior": float(bucket_prior),
                "recipe_bucket_prior": float(bucket_prior),
                "recipe_bucket_prior_multiplier": float(
                    float(bucket_prior) / max(1e-6, float(config.selection_prior_weight))
                ),
                "parent_refs": parent_refs,
            }
            result = engine._build_candidate_result(  # noqa: SLF001
                expression=draft.expression,
                mode="recipe_guided",
                parent_ids=parent_ids,
                generation_metadata=metadata,
                validation_ctx=validation_ctx,
            )
            candidate = result.candidate
            if candidate is None:
                stats.record_failure(result.failure_reason)
                continue
            if candidate.normalized_expression in existing_normalized:
                stats.record_failure("duplicate_normalized_expression")
                continue
            existing_normalized.add(candidate.normalized_expression)
            candidates.append(candidate)
            stats.generated_count += 1
            stats.success_count += 1
            stats.bucket_counts[bucket.search_bucket_id] += 1
            stats.template_counts[bucket.template_name] += 1
            if parent_used:
                stats.parented_count += 1
            else:
                stats.parentless_count += 1
        return candidates

    def _select_parent_for_bucket(
        self,
        *,
        bucket: _RecipeBucket,
        parents: list[_RecipeParent],
    ) -> _RecipeParent | None:
        eligible: list[_RecipeParent] = []
        for parent in parents:
            if not _parent_meets_base_thresholds(parent.metrics):
                continue
            if not _parent_meets_profile_thresholds(parent.metrics, bucket.objective_profile):
                continue
            eligible.append(parent)
        return eligible[0] if eligible else None

    def _eligible_fields(
        self,
        *,
        field_registry: FieldRegistry,
        blocked_fields: set[str],
        field_penalty_multipliers: dict[str, float],
    ) -> list[FieldSpec]:
        candidates = [
            spec
            for spec in field_registry.fields.values()
            if spec.operator_type == "matrix"
            and spec.category in _ALLOWED_FIELD_CATEGORIES
            and spec.name not in blocked_fields
        ]
        return sorted(
            candidates,
            key=lambda spec: (
                _field_priority_score(spec, field_penalty_multipliers=field_penalty_multipliers),
                spec.coverage,
                spec.alpha_usage_count,
                spec.name,
            ),
            reverse=True,
        )

    def _recipe_drafts_for_bucket(
        self,
        *,
        bucket: _RecipeBucket,
        field_pool: list[FieldSpec],
        field_penalty_multipliers: dict[str, float],
        parent_fields: set[str],
        registry: OperatorRegistry,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        if bucket.recipe_family == "fundamental_quality":
            return self._fundamental_quality_drafts(
                objective_profile=bucket.objective_profile,
                field_pool=field_pool,
                field_penalty_multipliers=field_penalty_multipliers,
                parent_fields=parent_fields,
                registry=registry,
                generation_config=generation_config,
            )
        if bucket.recipe_family == "accrual_vs_cashflow":
            return self._paired_recipe_drafts(
                recipe_family=bucket.recipe_family,
                left_side="cashflow",
                right_side="accrual",
                objective_profile=bucket.objective_profile,
                field_pool=field_pool,
                field_penalty_multipliers=field_penalty_multipliers,
                parent_fields=parent_fields,
                registry=registry,
                generation_config=generation_config,
            )
        if bucket.recipe_family == "value_vs_growth":
            return self._paired_recipe_drafts(
                recipe_family=bucket.recipe_family,
                left_side="value",
                right_side="growth",
                objective_profile=bucket.objective_profile,
                field_pool=field_pool,
                field_penalty_multipliers=field_penalty_multipliers,
                parent_fields=parent_fields,
                registry=registry,
                generation_config=generation_config,
            )
        return self._revision_surprise_drafts(
            objective_profile=bucket.objective_profile,
            field_pool=field_pool,
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
            registry=registry,
            generation_config=generation_config,
        )

    def _fundamental_quality_drafts(
        self,
        *,
        objective_profile: str,
        field_pool: list[FieldSpec],
        field_penalty_multipliers: dict[str, float],
        parent_fields: set[str],
        registry: OperatorRegistry,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        ordered_fields = _ordered_fields_for_side(
            field_pool=field_pool,
            recipe_family="fundamental_quality",
            side="primary",
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
        )
        lookbacks = _ordered_lookbacks(generation_config.lookbacks, objective_profile)
        drafts: list[_RecipeDraft] = []
        parent_used = any(spec.name in parent_fields for spec in ordered_fields[:2])
        for field in ordered_fields[:2]:
            if objective_profile in {"quality", "low_turnover"} and registry.contains("ts_mean"):
                for lookback in lookbacks[:2]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"rank(ts_mean({field.name},{lookback}))",
                            recipe_variant="rank_ts_mean",
                        )
                    )
                    if objective_profile != "low_turnover" and registry.contains("zscore"):
                        drafts.append(
                            _RecipeDraft(
                                expression=f"zscore(ts_mean({field.name},{lookback}))",
                                recipe_variant="zscore_ts_mean",
                            )
                        )
            if objective_profile != "low_turnover":
                drafts.append(_RecipeDraft(expression=f"rank({field.name})", recipe_variant="rank_raw"))
                if objective_profile == "balanced" and registry.contains("ts_mean"):
                    for lookback in lookbacks[:1]:
                        drafts.append(
                            _RecipeDraft(
                                expression=f"rank(ts_mean({field.name},{lookback}))",
                                recipe_variant="rank_ts_mean",
                            )
                        )
                        if registry.contains("zscore"):
                            drafts.append(
                                _RecipeDraft(
                                    expression=f"zscore(ts_mean({field.name},{lookback}))",
                                    recipe_variant="zscore_ts_mean",
                                )
                            )
        return {"drafts": _unique_drafts(drafts), "parent_used": parent_used}

    def _paired_recipe_drafts(
        self,
        *,
        recipe_family: str,
        left_side: str,
        right_side: str,
        objective_profile: str,
        field_pool: list[FieldSpec],
        field_penalty_multipliers: dict[str, float],
        parent_fields: set[str],
        registry: OperatorRegistry,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        left_fields = _ordered_fields_for_side(
            field_pool=field_pool,
            recipe_family=recipe_family,
            side=left_side,
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
        )
        right_fields = _ordered_fields_for_side(
            field_pool=field_pool,
            recipe_family=recipe_family,
            side=right_side,
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
        )
        left, right = _pick_distinct_pair(left_fields, right_fields)
        if left is None or right is None:
            return {"drafts": [], "parent_used": False}
        lookbacks = _ordered_lookbacks(generation_config.lookbacks, objective_profile)
        drafts: list[_RecipeDraft] = []
        parent_used = left.name in parent_fields or right.name in parent_fields
        if objective_profile != "low_turnover":
            drafts.append(
                _RecipeDraft(
                    expression=f"rank({left.name}) - rank({right.name})",
                    recipe_variant="rank_diff",
                )
            )
        if registry.contains("ts_mean"):
            for lookback in lookbacks[:2]:
                drafts.append(
                    _RecipeDraft(
                        expression=(
                            f"rank(ts_mean({left.name},{lookback})) - rank(ts_mean({right.name},{lookback}))"
                        ),
                        recipe_variant="rank_ts_mean_diff",
                    )
                )
        return {"drafts": _unique_drafts(drafts), "parent_used": parent_used}

    def _revision_surprise_drafts(
        self,
        *,
        objective_profile: str,
        field_pool: list[FieldSpec],
        field_penalty_multipliers: dict[str, float],
        parent_fields: set[str],
        registry: OperatorRegistry,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        ordered_fields = _ordered_fields_for_side(
            field_pool=field_pool,
            recipe_family="revision_surprise",
            side="primary",
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
        )
        lookbacks = _ordered_lookbacks(generation_config.lookbacks, objective_profile)
        drafts: list[_RecipeDraft] = []
        parent_used = any(spec.name in parent_fields for spec in ordered_fields[:2])
        for field in ordered_fields[:2]:
            if objective_profile == "quality" and registry.contains("ts_mean"):
                for lookback in lookbacks[:2]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"rank(ts_mean({field.name},{lookback}))",
                            recipe_variant="rank_ts_mean",
                        )
                    )
            if objective_profile != "low_turnover":
                drafts.append(_RecipeDraft(expression=f"rank({field.name})", recipe_variant="rank_raw"))
                if registry.contains("ts_delta"):
                    for lookback in lookbacks[:1]:
                        drafts.append(
                            _RecipeDraft(
                                expression=f"rank(ts_delta({field.name},{lookback}))",
                                recipe_variant="rank_ts_delta",
                            )
                        )
            if registry.contains("ts_mean"):
                for lookback in lookbacks[:2]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"rank(ts_mean({field.name},{lookback}))",
                            recipe_variant="rank_ts_mean",
                        )
                    )
        return {"drafts": _unique_drafts(drafts), "parent_used": parent_used}


def _all_buckets(config: RecipeGenerationConfig) -> list[_RecipeBucket]:
    buckets: list[_RecipeBucket] = []
    for recipe_family in config.enabled_recipe_families:
        template_name = _RECIPE_TEMPLATE_BY_FAMILY.get(recipe_family)
        if not template_name:
            continue
        for objective_profile in config.objective_profiles:
            buckets.append(
                _RecipeBucket(
                    search_bucket_id=f"{recipe_family}|fundamental|{objective_profile}",
                    recipe_family=recipe_family,
                    objective_profile=objective_profile,
                    template_name=template_name,
                )
            )
    return buckets


def _active_buckets(config: RecipeGenerationConfig, round_index: int) -> list[_RecipeBucket]:
    buckets = _all_buckets(config)
    if not buckets:
        return []
    active_count = max(1, min(int(config.active_bucket_count), len(buckets)))
    offset = (int(round_index) * active_count) % len(buckets)
    return [buckets[(offset + index) % len(buckets)] for index in range(active_count)]


def _planned_bucket_targets(*, bucket_count: int, target_count: int, max_candidates_per_bucket: int) -> list[int]:
    if bucket_count <= 0 or target_count <= 0:
        return []
    base = target_count // bucket_count
    remainder = target_count % bucket_count
    targets = [base + (1 if index < remainder else 0) for index in range(bucket_count)]
    return [min(int(max_candidates_per_bucket), max(0, target)) for target in targets]


def _ordered_fields_for_side(
    *,
    field_pool: list[FieldSpec],
    recipe_family: str,
    side: str,
    field_penalty_multipliers: dict[str, float],
    parent_fields: set[str],
) -> list[FieldSpec]:
    keywords = _RECIPE_KEYWORDS.get(recipe_family, {}).get(side, ())
    scored: list[tuple[float, FieldSpec]] = []
    for spec in field_pool:
        text = " ".join(
            [
                str(spec.name or ""),
                str(spec.description or ""),
                str(spec.subcategory or ""),
                str(spec.category or ""),
            ]
        ).lower()
        score = _field_priority_score(spec, field_penalty_multipliers=field_penalty_multipliers)
        if spec.name in parent_fields:
            score *= 1.25
        if keywords:
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > 0:
                score *= 1.0 + min(1.2, 0.40 * matches)
        if recipe_family == "revision_surprise" and spec.category == "analyst":
            score *= 1.20
        scored.append((score, spec))
    scored.sort(key=lambda item: (item[0], item[1].coverage, item[1].name), reverse=True)
    return [spec for _, spec in scored]


def _pick_distinct_pair(
    left_fields: list[FieldSpec],
    right_fields: list[FieldSpec],
) -> tuple[FieldSpec | None, FieldSpec | None]:
    for left in left_fields[:3]:
        for right in right_fields[:3]:
            if left.name != right.name:
                return left, right
    return None, None


def _ordered_lookbacks(lookbacks: list[int], objective_profile: str) -> list[int]:
    unique = list(dict.fromkeys(int(value) for value in lookbacks if int(value) > 0))
    if not unique:
        return [5]
    if objective_profile in {"quality", "low_turnover"}:
        return sorted(unique, reverse=True)
    return unique


def _parent_meets_base_thresholds(metrics: dict[str, Any]) -> bool:
    return (
        _to_float(metrics.get("fitness")) >= 0.02
        and _to_float(metrics.get("sharpe")) >= 0.03
        and _to_float(metrics.get("drawdown")) <= 0.75
        and not str(metrics.get("rejection_reason") or "").strip()
        and str(metrics.get("status") or "") == "completed"
    )


def _parent_meets_profile_thresholds(metrics: dict[str, Any], objective_profile: str) -> bool:
    if objective_profile == "quality":
        return (
            _to_float(metrics.get("fitness")) >= 0.05
            and _to_float(metrics.get("sharpe")) >= 0.08
            and _to_float(metrics.get("drawdown")) <= 0.50
        )
    if objective_profile == "low_turnover":
        return _to_float(metrics.get("turnover")) <= 0.70
    return True


def _field_priority_score(
    spec: FieldSpec,
    *,
    field_penalty_multipliers: dict[str, float],
) -> float:
    multiplier = float(field_penalty_multipliers.get(spec.name, 1.0) or 1.0)
    return max(1e-6, float(spec.field_score or 0.0) * multiplier)


def _unique_drafts(drafts: list[_RecipeDraft]) -> list[_RecipeDraft]:
    unique: list[_RecipeDraft] = []
    seen: set[str] = set()
    for draft in drafts:
        key = str(draft.expression).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(draft)
    return unique


def _decode_json_object(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str) and payload.strip():
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _neutral_yield_score() -> float:
    return 0.5


def _yield_score(*, selected_rate: float, positive_quality_rate: float, avg_quality_score: float) -> float:
    return float(
        0.50 * max(0.0, min(1.0, float(selected_rate)))
        + 0.35 * max(0.0, min(1.0, float(positive_quality_rate)))
        + 0.15 * max(0.0, float(avg_quality_score))
    )


def _adjusted_yield_score(*, raw_score: float, strength: float) -> float:
    neutral = _neutral_yield_score()
    return float(neutral + max(0.0, min(1.0, float(strength))) * (float(raw_score) - neutral))


def _allocate_integer_budget(
    *,
    total_budget: int,
    capacities: dict[str, int],
    weights: dict[str, float],
    floor_targets: dict[str, int],
) -> dict[str, int]:
    allocation = {key: 0 for key, cap in capacities.items() if int(cap) > 0}
    if total_budget <= 0 or not allocation:
        return allocation

    positive_floor_keys = [
        key for key in allocation if int(floor_targets.get(key, 0)) > 0 and int(capacities.get(key, 0)) > 0
    ]
    if total_budget < len(positive_floor_keys):
        ranked = sorted(
            positive_floor_keys,
            key=lambda key: (-float(weights.get(key, _neutral_yield_score())), key),
        )
        for key in ranked[: int(total_budget)]:
            allocation[key] = 1
        return allocation

    for key in positive_floor_keys:
        allocation[key] = min(int(capacities.get(key, 0)), max(1, int(floor_targets.get(key, 0))))

    allocated = sum(allocation.values())
    if allocated > total_budget:
        ranked = sorted(
            allocation,
            key=lambda key: (-float(weights.get(key, _neutral_yield_score())), key),
        )
        trimmed = {key: 0 for key in allocation}
        for key in ranked:
            if sum(trimmed.values()) >= total_budget:
                break
            trimmed[key] = 1
        return trimmed

    remaining = max(0, int(total_budget) - allocated)
    capacity_remaining = {
        key: max(0, int(capacities.get(key, 0)) - int(allocation.get(key, 0)))
        for key in allocation
    }
    eligible = [key for key, cap in capacity_remaining.items() if cap > 0]
    if remaining <= 0 or not eligible:
        return allocation

    total_weight = sum(max(0.0, float(weights.get(key, _neutral_yield_score()))) for key in eligible)
    if total_weight <= 0.0:
        total_weight = float(len(eligible))
        base_weights = {key: 1.0 for key in eligible}
    else:
        base_weights = {key: max(0.0, float(weights.get(key, _neutral_yield_score()))) for key in eligible}

    extras = {key: 0 for key in eligible}
    remainders: list[tuple[float, str]] = []
    for key in eligible:
        ideal = float(remaining) * float(base_weights[key]) / float(total_weight)
        extra = min(int(capacity_remaining[key]), int(ideal))
        extras[key] = extra
        remainder = ideal - float(extra)
        remainders.append((remainder, key))

    used = sum(extras.values())
    leftover = max(0, remaining - used)
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if leftover <= 0:
            break
        available = max(0, capacity_remaining[key] - extras[key])
        if available <= 0:
            continue
        extras[key] += 1
        leftover -= 1

    for key, extra in extras.items():
        allocation[key] += int(extra)
    return allocation
