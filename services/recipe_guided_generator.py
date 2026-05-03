from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from core.brain_checks import has_structural_risk_blocker, parse_names_json
from core.config import AdaptiveGenerationConfig, GenerationConfig, RecipeGenerationConfig
from core.quality_score import MultiObjectiveQualityScorer
from data.field_registry import FieldRegistry, FieldSpec
from domain.candidate import AlphaCandidate
from features.registry import OperatorRegistry
from generator.engine import AlphaGenerationEngine
from generator.guardrails import GenerationGuardrails
from memory.pattern_memory import RegionLearningContext
from services.evaluation_service import alpha_candidate_from_record
from storage.repository import SQLiteRepository

_ALLOWED_FIELD_CATEGORIES = frozenset({"fundamental", "analyst", "model"})
_RETURN_FIELD_CATEGORIES = frozenset({"price", "price_volume", "volume", "returns", "pv", "model"})
_RECIPE_TEMPLATE_BY_FAMILY = {
    "fundamental_quality": "recipe_fundamental_quality",
    "accrual_vs_cashflow": "recipe_accrual_vs_cashflow",
    "value_vs_growth": "recipe_value_vs_growth",
    "revision_surprise": "recipe_revision_surprise",
    "analyst_estimate_recency": "recipe_analyst_estimate_recency",
    "analyst_estimate_stability": "recipe_analyst_estimate_stability",
    "analyst_profitability_spread": "recipe_analyst_profitability_spread",
    "returns_term_structure": "recipe_returns_term_structure",
}
_RECIPE_DATASET_FAMILY_BY_FAMILY = {
    "analyst_estimate_recency": "analyst",
    "analyst_estimate_stability": "analyst",
    "analyst_profitability_spread": "analyst",
    "returns_term_structure": "returns",
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
    "analyst_estimate_recency": {
        "primary": (
            "estimate",
            "expected",
            "report",
            "eps",
            "ebit",
            "net",
            "profit",
            "sales",
            "forecast",
            "analyst",
        ),
    },
    "analyst_estimate_stability": {
        "primary": (
            "estimate",
            "eps",
            "ebit",
            "net",
            "profit",
            "sales",
            "stddev",
            "standard",
            "number",
            "count",
            "analyst",
        ),
    },
    "analyst_profitability_spread": {
        "profitability": (
            "eps",
            "ebit",
            "ebitda",
            "net",
            "profit",
            "income",
            "earnings",
        ),
        "anchor": (
            "sales",
            "dividend",
            "dps",
            "cash",
            "revenue",
            "estimate",
        ),
    },
    "returns_term_structure": {
        "short": (
            "return",
            "ret",
            "price",
            "close",
            "short",
            "daily",
            "5d",
            "4w",
        ),
        "long": (
            "return",
            "ret",
            "sigma",
            "volatility",
            "monthly",
            "60m",
            "90",
            "factor",
        ),
    },
}

GROUP_RELATIVE_SOURCE = "group_relative"
GROUP_RELATIVE_MAX_FRACTION = 0.25
GROUP_RELATIVE_PRIMARY_FIELD_CAP = 3
GROUP_RECIPE_RETRY_LIMIT = 3
_GROUP_RELATIVE_TEMPLATE_NAME = "recipe_group_relative"
_GROUP_RELATIVE_SEARCH_BUCKET_PREFIX = "group_relative"
_GROUP_RELATIVE_FUNDAMENTAL_FIELDS = (
    "anl39_agrosmgn",
    "anl39_agrosmgn2",
    "anl39_epschngin",
    "anl39_ttmepsincx",
    "anl39_qepsinclxo",
    "anl39_rasv2_atotd2eq",
    "anl46_sentiment",
    "anl46_performancepercentile",
    "anl69_roe_best_cur_fiscal_year_period",
    "anl69_roa_best_cur_fiscal_year_period",
    "anl69_eps_best_eeps_nxt_yr",
    "anl69_ebit_best_eeps_nxt_yr",
)
_GROUP_RELATIVE_EPS_FIELDS = (
    "anl69_eps_best_eeps_nxt_yr",
    "anl69_eps_best_eeps_cur_yr",
    "anl69_roe_best_eeps_nxt_yr",
    "anl69_roa_best_eeps_nxt_yr",
    "anl39_epschngin",
)
_GROUP_RELATIVE_PRIMARY_GROUP_KEYS = ("subindustry", "sector")
_GROUP_RELATIVE_FALLBACK_GROUP_KEYS = ("industry", "country")
_GROUP_RELATIVE_WEIGHTED_PRIMARY_KEYS = (
    "subindustry",
    "subindustry",
    "subindustry",
    "subindustry",
    "subindustry",
    "subindustry",
    "subindustry",
    "sector",
    "sector",
    "sector",
)
_GROUP_RELATIVE_LOOKBACKS = (5, 10, 20)
_GROUP_RELATIVE_SHORT_WINDOWS = (5, 10)


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
    duplicate_retry_count: int = 0
    duplicate_retry_counts_by_bucket: Counter[str] = field(default_factory=Counter)
    exhausted_bucket_counts: Counter[str] = field(default_factory=Counter)
    unique_draft_count: int = 0
    unique_draft_counts_by_bucket: Counter[str] = field(default_factory=Counter)
    field_usage_counts: Counter[str] = field(default_factory=Counter)
    pair_usage_counts: Counter[str] = field(default_factory=Counter)
    bucket_biases: dict[str, float] = field(default_factory=dict)
    suppressed_bucket_caps: dict[str, int] = field(default_factory=dict)
    spilled_to_fresh: int = 0
    generation_total_ms: float = 0.0
    group_relative_generated_count: int = 0
    group_relative_selected_count: int = 0
    group_relative_attempt_count: int = 0
    group_relative_skipped_count: int = 0
    group_relative_group_counts: Counter[str] = field(default_factory=Counter)
    group_relative_skipped_by_group: Counter[str] = field(default_factory=Counter)
    group_relative_skip_reason_counts: Counter[str] = field(default_factory=Counter)
    group_relative_field_usage_counts: Counter[str] = field(default_factory=Counter)

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
            "recipe_guided_duplicate_retry_count": int(self.duplicate_retry_count),
            "recipe_guided_duplicate_retry_counts_by_bucket": dict(self.duplicate_retry_counts_by_bucket),
            "recipe_guided_exhausted_bucket_counts": dict(self.exhausted_bucket_counts),
            "recipe_guided_unique_draft_count": int(self.unique_draft_count),
            "recipe_guided_unique_draft_counts_by_bucket": dict(self.unique_draft_counts_by_bucket),
            "recipe_guided_field_usage_counts": dict(self.field_usage_counts),
            "recipe_guided_pair_usage_counts": dict(self.pair_usage_counts),
            "recipe_guided_bucket_biases": dict(self.bucket_biases),
            "recipe_guided_suppressed_bucket_caps": dict(self.suppressed_bucket_caps),
            "recipe_guided_spilled_to_fresh": int(self.spilled_to_fresh),
            "recipe_guided_generation_total_ms": round(self.generation_total_ms, 3),
            "group_relative_generated": int(self.group_relative_generated_count),
            "group_relative_attempt_count": int(self.group_relative_attempt_count),
            "group_relative_skipped": int(self.group_relative_skipped_count),
            "group_relative_selected": int(self.group_relative_selected_count),
            "group_relative_group_counts": dict(self.group_relative_group_counts),
            "group_relative_skipped_by_group": dict(self.group_relative_skipped_by_group),
            "group_relative_skip_reason_counts": dict(self.group_relative_skip_reason_counts),
            "group_relative_field_usage_counts": dict(self.group_relative_field_usage_counts),
        }


@dataclass(frozen=True, slots=True)
class RecipeGuidedResult:
    candidates: list[AlphaCandidate]
    stats: RecipeGuidedStats


@dataclass(frozen=True, slots=True)
class _RecipeBucket:
    search_bucket_id: str
    recipe_family: str
    dataset_family: str
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
    fields: tuple[str, ...] = ()
    pair_key: str = ""
    source: str = "recipe_guided"
    group_recipe_group: str = ""
    primary_field: str = ""
    group_key: str = ""


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
        suppressed_bucket_caps = self._load_bucket_suppression_caps(
            run_id=run_id,
            before_round_index=round_index,
            config=config,
            active_buckets=active_buckets,
        )
        recent_usage = self._load_recent_recipe_usage(
            run_id=run_id,
            before_round_index=round_index,
            config=config,
        )
        existing = set(existing_normalized)
        candidates: list[AlphaCandidate] = []
        group_target = min(
            target_count,
            int(float(target_count) * GROUP_RELATIVE_MAX_FRACTION),
        )
        if group_target > 0:
            group_candidates = self._generate_group_relative_candidates(
                group_target=group_target,
                config=config,
                registry=registry,
                field_registry=field_registry,
                blocked_fields=blocked_fields,
                field_penalty_multipliers=field_penalty_multipliers or {},
                generation_config=generation_config,
                engine=engine,
                validation_ctx=validation_ctx,
                existing_normalized=existing,
                run_id=run_id,
                round_index=round_index,
                stats=stats,
            )
            candidates.extend(group_candidates)

        bucket_target_total = max(0, target_count - len(candidates))
        planned_targets = self._planned_bucket_targets(
            config=config,
            target_count=bucket_target_total,
            active_buckets=active_buckets,
            bucket_yield_scores=bucket_yield_scores,
            suppressed_bucket_caps=suppressed_bucket_caps,
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
        stats.bucket_biases = {
            bucket.search_bucket_id: float(_bucket_bias(config=config, bucket_id=bucket.search_bucket_id))
            for bucket in active_buckets
        }
        stats.suppressed_bucket_caps = dict(suppressed_bucket_caps)
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
                adaptive_config=adaptive_config,
                registry=registry,
                field_registry=field_registry,
                blocked_fields=blocked_fields,
                field_penalty_multipliers=field_penalty_multipliers or {},
                recent_field_usage_counts=recent_usage["field_counts"],
                recent_pair_usage_counts=recent_usage["pair_counts"],
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

        stats.spilled_to_fresh = max(0, target_count - len(candidates))
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
                "hard_fail_checks": parse_names_json(row.get("result_hard_fail_checks_json")),
                "blocking_warning_checks": parse_names_json(row.get("result_blocking_warning_checks_json")),
                "check_summary_json": row.get("result_check_summary_json"),
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
                        check_summary_json=metrics["check_summary_json"],
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
                        check_summary_json=row.get("check_summary_json"),
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

    def _load_recent_recipe_usage(
        self,
        *,
        run_id: str,
        before_round_index: int,
        config: RecipeGenerationConfig,
    ) -> dict[str, Counter[str]]:
        if not bool(config.enable_field_rotation):
            return {"field_counts": Counter(), "pair_counts": Counter()}
        rows = self.repository.list_recent_recipe_guided_usage_rows(
            run_id=run_id,
            before_round_index=int(before_round_index),
            lookback_rounds=int(config.field_rotation_lookback_rounds),
        )
        field_counts: Counter[str] = Counter()
        pair_counts: Counter[str] = Counter()
        for row in rows:
            metadata = _decode_json_object(row.get("generation_metadata"))
            metadata_fields = metadata.get("recipe_field_names")
            if isinstance(metadata_fields, list):
                fields = [str(field).strip() for field in metadata_fields if str(field).strip()]
            else:
                fields = _decode_json_list(row.get("fields_used_json"))
            field_counts.update(fields)
            pair_key = str(metadata.get("recipe_pair_key") or "").strip()
            if pair_key:
                pair_counts[pair_key] += 1
        return {"field_counts": field_counts, "pair_counts": pair_counts}

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
                scores[bucket_id] = _apply_bucket_bias(
                    score=_neutral_yield_score(),
                    config=config,
                    bucket_id=bucket_id,
                )
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
            scores[bucket_id] = _apply_bucket_bias(
                score=scores[bucket_id],
                config=config,
                bucket_id=bucket_id,
            )
        return scores

    def _load_bucket_suppression_caps(
        self,
        *,
        run_id: str,
        before_round_index: int,
        config: RecipeGenerationConfig,
        active_buckets: list[_RecipeBucket],
    ) -> dict[str, int]:
        if not bool(config.bucket_suppression_enabled):
            return {}
        active_bucket_ids = {bucket.search_bucket_id for bucket in active_buckets}
        if not active_bucket_ids:
            return {}
        result_rows = self.repository.list_recipe_bucket_result_rows(
            run_id=run_id,
            before_round_index=int(before_round_index),
            lookback_rounds=int(config.yield_lookback_rounds),
        )
        support_counts: Counter[str] = Counter()
        sharpe_sums: Counter[str] = Counter()
        fitness_sums: Counter[str] = Counter()
        for row in result_rows:
            if str(row.get("status") or "") != "completed":
                continue
            metadata = _decode_json_object(row.get("generation_metadata"))
            bucket_id = str(metadata.get("search_bucket_id") or "").strip()
            if bucket_id not in active_bucket_ids:
                continue
            support_counts[bucket_id] += 1
            sharpe_sums[bucket_id] += _to_float(row.get("sharpe"))
            fitness_sums[bucket_id] += _to_float(row.get("fitness"))

        suppressed: dict[str, int] = {}
        for bucket_id, support in support_counts.items():
            if int(support) < int(config.bucket_suppression_min_support):
                continue
            avg_sharpe = float(sharpe_sums[bucket_id]) / max(1, int(support))
            avg_fitness = float(fitness_sums[bucket_id]) / max(1, int(support))
            if (
                avg_sharpe < float(config.bucket_suppression_sharpe_floor)
                or avg_fitness < float(config.bucket_suppression_fitness_floor)
            ):
                suppressed[bucket_id] = int(config.bucket_suppression_max_candidates)
        return suppressed

    def _planned_bucket_targets(
        self,
        *,
        config: RecipeGenerationConfig,
        target_count: int,
        active_buckets: list[_RecipeBucket],
        bucket_yield_scores: dict[str, float],
        suppressed_bucket_caps: dict[str, int] | None = None,
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
        suppressed_bucket_caps = dict(suppressed_bucket_caps or {})
        capacities = {}
        for bucket in active_buckets:
            cap = int(config.max_candidates_per_bucket)
            if bucket.search_bucket_id in suppressed_bucket_caps:
                cap = min(cap, max(0, int(suppressed_bucket_caps[bucket.search_bucket_id])))
            capacities[bucket.search_bucket_id] = cap
        weights = {
            bucket.search_bucket_id: float(bucket_yield_scores.get(bucket.search_bucket_id, _neutral_yield_score()))
            for bucket in active_buckets
        }
        allocations = _allocate_integer_budget(
            total_budget=int(target_count),
            capacities=capacities,
            weights=weights,
            floor_targets={
                bucket.search_bucket_id: min(
                    int(config.bucket_exploration_floor),
                    int(capacities.get(bucket.search_bucket_id, 0)),
                )
                for bucket in active_buckets
            },
        )
        return {bucket_id: int(value) for bucket_id, value in allocations.items() if int(value) > 0}

    def _generate_group_relative_candidates(
        self,
        *,
        group_target: int,
        config: RecipeGenerationConfig,
        registry: OperatorRegistry,
        field_registry: FieldRegistry,
        blocked_fields: set[str],
        field_penalty_multipliers: dict[str, float],
        generation_config: GenerationConfig,
        engine: AlphaGenerationEngine,
        validation_ctx,
        existing_normalized: set[str],
        run_id: str,
        round_index: int,
        stats: RecipeGuidedStats,
    ) -> list[AlphaCandidate]:
        if group_target <= 0:
            return []
        drafts = self._build_group_recipes(
            registry=registry,
            field_registry=field_registry,
            blocked_fields=blocked_fields,
            field_penalty_multipliers=field_penalty_multipliers,
            generation_config=generation_config,
            round_index=round_index,
        )
        stats.unique_draft_count += len(drafts)
        stats.unique_draft_counts_by_bucket[GROUP_RELATIVE_SOURCE] += len(drafts)
        if not drafts:
            stats.group_relative_skipped_count += int(group_target)
            stats.group_relative_skip_reason_counts["group_relative_no_eligible_drafts"] += int(group_target)
            return []

        candidates: list[AlphaCandidate] = []
        primary_field_counts: Counter[str] = Counter()
        draft_index = 0
        while len(candidates) < group_target and draft_index < len(drafts):
            slot_succeeded = False
            for _ in range(GROUP_RECIPE_RETRY_LIMIT):
                if draft_index >= len(drafts):
                    break
                draft = drafts[draft_index]
                draft_index += 1
                group_name = draft.group_recipe_group or "unknown"
                primary_field = draft.primary_field or (draft.fields[0] if draft.fields else "")
                if (
                    primary_field
                    and int(primary_field_counts.get(primary_field, 0)) >= GROUP_RELATIVE_PRIMARY_FIELD_CAP
                ):
                    self._record_group_relative_skip(
                        stats=stats,
                        reason="group_relative_primary_field_cap",
                        group_name=group_name,
                    )
                    continue

                stats.attempt_count += 1
                stats.group_relative_attempt_count += 1
                search_bucket_id = f"{_GROUP_RELATIVE_SEARCH_BUCKET_PREFIX}|{group_name.lower()}"
                metadata = {
                    "generation_mode": "recipe_guided",
                    "generation_source": GROUP_RELATIVE_SOURCE,
                    "mutation_mode": GROUP_RELATIVE_SOURCE,
                    "template_name": _GROUP_RELATIVE_TEMPLATE_NAME,
                    "motif": "group_relative_signal",
                    "search_bucket_id": search_bucket_id,
                    "recipe_family": GROUP_RELATIVE_SOURCE,
                    "dataset_family": "group",
                    "objective_profile": GROUP_RELATIVE_SOURCE,
                    "recipe_variant": draft.recipe_variant,
                    "recipe_round_index": int(round_index),
                    "recipe_field_names": list(draft.fields),
                    "recipe_pair_key": draft.pair_key,
                    "recipe_parent_alpha_id": "",
                    "recipe_prior": float(config.selection_prior_weight),
                    "recipe_bucket_prior": float(config.selection_prior_weight),
                    "recipe_bucket_prior_multiplier": 1.0,
                    "group_recipe_group": group_name,
                    "group_recipe_primary_field": primary_field,
                    "group_recipe_group_key": draft.group_key,
                    "parent_refs": [],
                    "run_id": run_id,
                }
                result = engine._build_candidate_result(  # noqa: SLF001
                    expression=draft.expression,
                    mode="recipe_guided",
                    parent_ids=(),
                    generation_metadata=metadata,
                    validation_ctx=validation_ctx,
                )
                candidate = result.candidate
                if candidate is None:
                    reason = result.failure_reason or "group_relative_candidate_build_failed"
                    stats.record_failure(reason)
                    self._record_group_relative_skip(
                        stats=stats,
                        reason=reason,
                        group_name=group_name,
                    )
                    continue
                if candidate.normalized_expression in existing_normalized:
                    stats.record_failure("duplicate_normalized_expression")
                    stats.duplicate_retry_count += 1
                    stats.duplicate_retry_counts_by_bucket[search_bucket_id] += 1
                    self._record_group_relative_skip(
                        stats=stats,
                        reason="duplicate_normalized_expression",
                        group_name=group_name,
                    )
                    continue

                existing_normalized.add(candidate.normalized_expression)
                candidates.append(candidate)
                primary_field_counts[primary_field] += 1
                stats.generated_count += 1
                stats.success_count += 1
                stats.group_relative_generated_count += 1
                stats.group_relative_group_counts[group_name] += 1
                stats.group_relative_field_usage_counts[primary_field] += 1
                stats.bucket_counts[search_bucket_id] += 1
                stats.template_counts[_GROUP_RELATIVE_TEMPLATE_NAME] += 1
                stats.field_usage_counts.update(draft.fields)
                stats.parentless_count += 1
                slot_succeeded = True
                break
            if not slot_succeeded and draft_index >= len(drafts):
                break
        return candidates

    def _record_group_relative_skip(
        self,
        *,
        stats: RecipeGuidedStats,
        reason: str,
        group_name: str,
    ) -> None:
        normalized = str(reason or "unknown_failure").strip() or "unknown_failure"
        stats.group_relative_skipped_count += 1
        stats.group_relative_skip_reason_counts[normalized] += 1
        stats.group_relative_skipped_by_group[str(group_name or "unknown")] += 1

    def _build_group_recipes(
        self,
        *,
        registry: OperatorRegistry,
        field_registry: FieldRegistry,
        blocked_fields: set[str],
        field_penalty_multipliers: dict[str, float],
        generation_config: GenerationConfig,
        round_index: int,
    ) -> list[_RecipeDraft]:
        group_keys = _resolved_group_key_names(
            field_registry=field_registry,
            generation_config=generation_config,
        )
        if not group_keys:
            return []
        weighted_group_keys = _weighted_group_key_cycle(group_keys)
        subindustry_key = "subindustry" if "subindustry" in group_keys else ""
        fundamental_fields = _resolved_group_matrix_field_names(
            candidate_names=_GROUP_RELATIVE_FUNDAMENTAL_FIELDS,
            field_registry=field_registry,
            generation_config=generation_config,
            blocked_fields=blocked_fields,
            field_penalty_multipliers=field_penalty_multipliers,
            round_index=round_index,
        )
        eps_fields = _resolved_group_matrix_field_names(
            candidate_names=_GROUP_RELATIVE_EPS_FIELDS,
            field_registry=field_registry,
            generation_config=generation_config,
            blocked_fields=blocked_fields,
            field_penalty_multipliers=field_penalty_multipliers,
            round_index=round_index + 1,
        )
        returns_field = _resolved_single_group_matrix_field(
            "returns",
            field_registry=field_registry,
            generation_config=generation_config,
            blocked_fields=blocked_fields,
        )
        close_field = _resolved_single_group_matrix_field(
            "close",
            field_registry=field_registry,
            generation_config=generation_config,
            blocked_fields=blocked_fields,
        )
        volume_field = _resolved_single_group_matrix_field(
            "volume",
            field_registry=field_registry,
            generation_config=generation_config,
            blocked_fields=blocked_fields,
        )

        drafts_by_group: dict[str, list[_RecipeDraft]] = {
            "A": [],
            "B": [],
            "C": [],
            "D": [],
        }
        drafts_by_group["A"].extend(
            _group_a_relative_fundamental_drafts(
                fields=fundamental_fields,
                group_keys=weighted_group_keys,
                registry=registry,
            )
        )
        drafts_by_group["B"].extend(
            _group_b_momentum_drafts(
                returns_field=returns_field,
                close_field=close_field,
                group_keys=weighted_group_keys,
                registry=registry,
            )
        )
        drafts_by_group["C"].extend(
            _group_c_earnings_drafts(
                fields=eps_fields,
                group_keys=weighted_group_keys,
                subindustry_key=subindustry_key,
                registry=registry,
            )
        )
        drafts_by_group["D"].extend(
            _group_d_liquidity_drafts(
                volume_field=volume_field,
                group_keys=weighted_group_keys,
                subindustry_key=subindustry_key,
                registry=registry,
            )
        )
        return _interleave_group_drafts(drafts_by_group)

    def _generate_bucket_candidates(
        self,
        *,
        bucket: _RecipeBucket,
        bucket_target: int,
        config: RecipeGenerationConfig,
        adaptive_config: AdaptiveGenerationConfig,
        registry: OperatorRegistry,
        field_registry: FieldRegistry,
        blocked_fields: set[str],
        field_penalty_multipliers: dict[str, float],
        recent_field_usage_counts: Counter[str],
        recent_pair_usage_counts: Counter[str],
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
            recipe_family=bucket.recipe_family,
            generation_config=generation_config,
        )
        draft_bundle = self._recipe_drafts_for_bucket(
            bucket=bucket,
            adaptive_config=adaptive_config,
            field_pool=field_pool,
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
            recent_field_usage_counts=recent_field_usage_counts,
            recent_pair_usage_counts=recent_pair_usage_counts,
            round_index=round_index,
            config=config,
            registry=registry,
            generation_config=generation_config,
        )
        drafts = list(draft_bundle["drafts"])
        stats.unique_draft_count += len(drafts)
        stats.unique_draft_counts_by_bucket[bucket.search_bucket_id] += len(drafts)
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
        max_attempts = min(
            len(drafts),
            max(int(bucket_target), int(bucket_target) * int(config.duplicate_retry_multiplier)),
        )
        for draft in drafts[:max_attempts]:
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
                "dataset_family": bucket.dataset_family,
                "objective_profile": bucket.objective_profile,
                "recipe_variant": draft.recipe_variant,
                "recipe_round_index": int(round_index),
                "recipe_field_names": list(draft.fields),
                "recipe_pair_key": draft.pair_key,
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
                stats.duplicate_retry_count += 1
                stats.duplicate_retry_counts_by_bucket[bucket.search_bucket_id] += 1
                continue
            existing_normalized.add(candidate.normalized_expression)
            candidates.append(candidate)
            stats.generated_count += 1
            stats.success_count += 1
            stats.bucket_counts[bucket.search_bucket_id] += 1
            stats.template_counts[bucket.template_name] += 1
            stats.field_usage_counts.update(draft.fields)
            if draft.pair_key:
                stats.pair_usage_counts[draft.pair_key] += 1
            if parent_used:
                stats.parented_count += 1
            else:
                stats.parentless_count += 1
        if len(candidates) < bucket_target:
            stats.exhausted_bucket_counts[bucket.search_bucket_id] += 1
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
        recipe_family: str,
        generation_config: GenerationConfig,
    ) -> list[FieldSpec]:
        allowed_categories = (
            _RETURN_FIELD_CATEGORIES
            if recipe_family == "returns_term_structure"
            else _ALLOWED_FIELD_CATEGORIES
        )
        configured_allowed_fields = set(generation_config.allowed_fields or [])
        candidates = [
            spec
            for spec in field_registry.fields.values()
            if spec.operator_type == "matrix"
            and spec.category in allowed_categories
            and (not configured_allowed_fields or spec.name in configured_allowed_fields)
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
        adaptive_config: AdaptiveGenerationConfig,
        field_pool: list[FieldSpec],
        field_penalty_multipliers: dict[str, float],
        parent_fields: set[str],
        recent_field_usage_counts: Counter[str],
        recent_pair_usage_counts: Counter[str],
        round_index: int,
        config: RecipeGenerationConfig,
        registry: OperatorRegistry,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        if bucket.recipe_family == "fundamental_quality":
            return self._fundamental_quality_drafts(
                objective_profile=bucket.objective_profile,
                field_pool=field_pool,
                field_penalty_multipliers=field_penalty_multipliers,
                parent_fields=parent_fields,
                recent_field_usage_counts=recent_field_usage_counts,
                round_index=round_index,
                config=config,
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
                recent_field_usage_counts=recent_field_usage_counts,
                recent_pair_usage_counts=recent_pair_usage_counts,
                round_index=round_index,
                config=config,
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
                recent_field_usage_counts=recent_field_usage_counts,
                recent_pair_usage_counts=recent_pair_usage_counts,
                round_index=round_index,
                config=config,
                registry=registry,
                generation_config=generation_config,
            )
        if bucket.recipe_family == "analyst_estimate_recency":
            return self._analyst_estimate_recency_drafts(
                objective_profile=bucket.objective_profile,
                field_pool=field_pool,
                field_penalty_multipliers=field_penalty_multipliers,
                parent_fields=parent_fields,
                recent_field_usage_counts=recent_field_usage_counts,
                round_index=round_index,
                config=config,
                adaptive_config=adaptive_config,
                registry=registry,
                generation_config=generation_config,
            )
        if bucket.recipe_family == "analyst_estimate_stability":
            return self._analyst_estimate_stability_drafts(
                objective_profile=bucket.objective_profile,
                field_pool=field_pool,
                field_penalty_multipliers=field_penalty_multipliers,
                parent_fields=parent_fields,
                recent_field_usage_counts=recent_field_usage_counts,
                round_index=round_index,
                config=config,
                adaptive_config=adaptive_config,
                registry=registry,
                generation_config=generation_config,
            )
        if bucket.recipe_family == "analyst_profitability_spread":
            return self._paired_elite_recipe_drafts(
                recipe_family=bucket.recipe_family,
                left_side="profitability",
                right_side="anchor",
                objective_profile=bucket.objective_profile,
                field_pool=field_pool,
                field_penalty_multipliers=field_penalty_multipliers,
                parent_fields=parent_fields,
                recent_field_usage_counts=recent_field_usage_counts,
                recent_pair_usage_counts=recent_pair_usage_counts,
                round_index=round_index,
                config=config,
                adaptive_config=adaptive_config,
                registry=registry,
                generation_config=generation_config,
            )
        if bucket.recipe_family == "returns_term_structure":
            return self._paired_elite_recipe_drafts(
                recipe_family=bucket.recipe_family,
                left_side="short",
                right_side="long",
                objective_profile=bucket.objective_profile,
                field_pool=field_pool,
                field_penalty_multipliers=field_penalty_multipliers,
                parent_fields=parent_fields,
                recent_field_usage_counts=recent_field_usage_counts,
                recent_pair_usage_counts=recent_pair_usage_counts,
                round_index=round_index,
                config=config,
                adaptive_config=adaptive_config,
                registry=registry,
                generation_config=generation_config,
            )
        return self._revision_surprise_drafts(
            objective_profile=bucket.objective_profile,
            field_pool=field_pool,
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
            recent_field_usage_counts=recent_field_usage_counts,
            round_index=round_index,
            config=config,
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
        recent_field_usage_counts: Counter[str],
        round_index: int,
        config: RecipeGenerationConfig,
        registry: OperatorRegistry,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        ordered_fields = _ordered_fields_for_side(
            field_pool=field_pool,
            recipe_family="fundamental_quality",
            side="primary",
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
            recent_field_usage_counts=recent_field_usage_counts,
        )
        ordered_fields = _rotated_top_fields(
            ordered_fields,
            limit=int(config.max_field_candidates_per_side),
            round_index=round_index if bool(config.enable_field_rotation) else 0,
        )
        lookbacks = _ordered_lookbacks(generation_config.lookbacks, objective_profile)
        drafts: list[_RecipeDraft] = []
        parent_used = any(spec.name in parent_fields for spec in ordered_fields)
        for field in ordered_fields:
            if objective_profile in {"quality", "low_turnover"} and registry.contains("ts_mean"):
                for lookback in lookbacks[:2]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"rank(ts_mean({field.name},{lookback}))",
                            recipe_variant="rank_ts_mean",
                            fields=(field.name,),
                        )
                    )
                    if objective_profile != "low_turnover" and registry.contains("zscore"):
                        drafts.append(
                            _RecipeDraft(
                                expression=f"zscore(ts_mean({field.name},{lookback}))",
                                recipe_variant="zscore_ts_mean",
                                fields=(field.name,),
                            )
                        )
            if objective_profile != "low_turnover":
                drafts.append(
                    _RecipeDraft(
                        expression=f"rank({field.name})",
                        recipe_variant="rank_raw",
                        fields=(field.name,),
                    )
                )
                if objective_profile == "balanced" and registry.contains("ts_mean"):
                    for lookback in lookbacks[:1]:
                        drafts.append(
                            _RecipeDraft(
                                expression=f"rank(ts_mean({field.name},{lookback}))",
                                recipe_variant="rank_ts_mean",
                                fields=(field.name,),
                            )
                        )
                        if registry.contains("zscore"):
                            drafts.append(
                                _RecipeDraft(
                                    expression=f"zscore(ts_mean({field.name},{lookback}))",
                                    recipe_variant="zscore_ts_mean",
                                    fields=(field.name,),
                                )
                            )
        return {
            "drafts": _unique_drafts(drafts)[: int(config.max_drafts_per_bucket)],
            "parent_used": parent_used,
        }

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
        recent_field_usage_counts: Counter[str],
        recent_pair_usage_counts: Counter[str],
        round_index: int,
        config: RecipeGenerationConfig,
        registry: OperatorRegistry,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        left_fields = _ordered_fields_for_side(
            field_pool=field_pool,
            recipe_family=recipe_family,
            side=left_side,
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
            recent_field_usage_counts=recent_field_usage_counts,
        )
        right_fields = _ordered_fields_for_side(
            field_pool=field_pool,
            recipe_family=recipe_family,
            side=right_side,
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
            recent_field_usage_counts=recent_field_usage_counts,
        )
        left_fields = _rotated_top_fields(
            left_fields,
            limit=int(config.max_field_candidates_per_side),
            round_index=round_index if bool(config.enable_field_rotation) else 0,
        )
        right_fields = _rotated_top_fields(
            right_fields,
            limit=int(config.max_field_candidates_per_side),
            round_index=round_index + 1 if bool(config.enable_field_rotation) else 0,
        )
        pairs = _distinct_pairs(
            left_fields,
            right_fields,
            limit=int(config.max_pair_candidates_per_bucket),
            pair_usage_counts=recent_pair_usage_counts,
        )
        if not pairs:
            return {"drafts": [], "parent_used": False}
        lookbacks = _ordered_lookbacks(generation_config.lookbacks, objective_profile)
        drafts: list[_RecipeDraft] = []
        parent_used = any(left.name in parent_fields or right.name in parent_fields for left, right in pairs)
        for left, right in pairs:
            pair_key = _pair_key(left, right)
            if objective_profile != "low_turnover":
                drafts.append(
                    _RecipeDraft(
                        expression=f"rank({left.name}) - rank({right.name})",
                        recipe_variant="rank_diff",
                        fields=(left.name, right.name),
                        pair_key=pair_key,
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
                            fields=(left.name, right.name),
                            pair_key=pair_key,
                        )
                    )
        return {
            "drafts": _unique_drafts(drafts)[: int(config.max_drafts_per_bucket)],
            "parent_used": parent_used,
        }

    def _analyst_estimate_recency_drafts(
        self,
        *,
        objective_profile: str,
        field_pool: list[FieldSpec],
        field_penalty_multipliers: dict[str, float],
        parent_fields: set[str],
        recent_field_usage_counts: Counter[str],
        round_index: int,
        config: RecipeGenerationConfig,
        adaptive_config: AdaptiveGenerationConfig,
        registry: OperatorRegistry,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        ordered_fields = _ordered_fields_for_side(
            field_pool=field_pool,
            recipe_family="analyst_estimate_recency",
            side="primary",
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
            recent_field_usage_counts=recent_field_usage_counts,
        )
        ordered_fields = _rotated_top_fields(
            ordered_fields,
            limit=int(config.max_field_candidates_per_side),
            round_index=round_index if bool(config.enable_field_rotation) else 0,
        )
        lookbacks = _elite_recipe_lookbacks(adaptive_config, generation_config, objective_profile)
        drafts: list[_RecipeDraft] = []
        parent_used = any(spec.name in parent_fields for spec in ordered_fields)
        for field in ordered_fields:
            if registry.contains("days_from_last_change") and registry.contains("rank"):
                drafts.append(
                    _RecipeDraft(
                        expression=f"-rank(days_from_last_change({field.name}))",
                        recipe_variant="neg_rank_days_from_last_change",
                        fields=(field.name,),
                    )
                )
            if objective_profile != "low_turnover" and registry.contains("ts_arg_max") and registry.contains("rank"):
                for lookback in lookbacks[:2]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"-rank(ts_arg_max({field.name},{lookback}))",
                            recipe_variant="neg_rank_ts_arg_max",
                            fields=(field.name,),
                        )
                    )
            if registry.contains("ts_av_diff") and registry.contains("rank"):
                for lookback in lookbacks[:2]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"rank(ts_av_diff({field.name},{lookback}))",
                            recipe_variant="rank_ts_av_diff",
                            fields=(field.name,),
                        )
                    )
        return {
            "drafts": _unique_drafts(drafts)[: int(config.max_drafts_per_bucket)],
            "parent_used": parent_used,
        }

    def _analyst_estimate_stability_drafts(
        self,
        *,
        objective_profile: str,
        field_pool: list[FieldSpec],
        field_penalty_multipliers: dict[str, float],
        parent_fields: set[str],
        recent_field_usage_counts: Counter[str],
        round_index: int,
        config: RecipeGenerationConfig,
        adaptive_config: AdaptiveGenerationConfig,
        registry: OperatorRegistry,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        ordered_fields = _ordered_fields_for_side(
            field_pool=field_pool,
            recipe_family="analyst_estimate_stability",
            side="primary",
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
            recent_field_usage_counts=recent_field_usage_counts,
        )
        ordered_fields = _rotated_top_fields(
            ordered_fields,
            limit=int(config.max_field_candidates_per_side),
            round_index=round_index if bool(config.enable_field_rotation) else 0,
        )
        lookbacks = _elite_recipe_lookbacks(adaptive_config, generation_config, objective_profile)
        drafts: list[_RecipeDraft] = []
        parent_used = any(spec.name in parent_fields for spec in ordered_fields)
        for field in ordered_fields:
            if registry.contains("ts_scale") and registry.contains("rank"):
                for lookback in lookbacks[:2]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"rank(ts_scale({field.name},{lookback}))",
                            recipe_variant="rank_ts_scale",
                            fields=(field.name,),
                        )
                    )
            if registry.contains("ts_count_nans") and registry.contains("rank"):
                for lookback in lookbacks[:2]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"-rank(ts_count_nans({field.name},{lookback}))",
                            recipe_variant="neg_rank_ts_count_nans",
                            fields=(field.name,),
                        )
                    )
            if objective_profile != "low_turnover" and registry.contains("ts_std_dev") and registry.contains("rank"):
                for lookback in lookbacks[:1]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"-rank(ts_std_dev({field.name},{lookback}))",
                            recipe_variant="neg_rank_ts_std_dev",
                            fields=(field.name,),
                        )
                    )
        return {
            "drafts": _unique_drafts(drafts)[: int(config.max_drafts_per_bucket)],
            "parent_used": parent_used,
        }

    def _paired_elite_recipe_drafts(
        self,
        *,
        recipe_family: str,
        left_side: str,
        right_side: str,
        objective_profile: str,
        field_pool: list[FieldSpec],
        field_penalty_multipliers: dict[str, float],
        parent_fields: set[str],
        recent_field_usage_counts: Counter[str],
        recent_pair_usage_counts: Counter[str],
        round_index: int,
        config: RecipeGenerationConfig,
        adaptive_config: AdaptiveGenerationConfig,
        registry: OperatorRegistry,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        left_fields = _ordered_fields_for_side(
            field_pool=field_pool,
            recipe_family=recipe_family,
            side=left_side,
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
            recent_field_usage_counts=recent_field_usage_counts,
        )
        right_fields = _ordered_fields_for_side(
            field_pool=field_pool,
            recipe_family=recipe_family,
            side=right_side,
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
            recent_field_usage_counts=recent_field_usage_counts,
        )
        left_fields = _rotated_top_fields(
            left_fields,
            limit=int(config.max_field_candidates_per_side),
            round_index=round_index if bool(config.enable_field_rotation) else 0,
        )
        right_fields = _rotated_top_fields(
            right_fields,
            limit=int(config.max_field_candidates_per_side),
            round_index=round_index + 1 if bool(config.enable_field_rotation) else 0,
        )
        pairs = _distinct_pairs(
            left_fields,
            right_fields,
            limit=int(config.max_pair_candidates_per_bucket),
            pair_usage_counts=recent_pair_usage_counts,
        )
        if not pairs:
            return {"drafts": [], "parent_used": False}

        lookbacks = _elite_recipe_lookbacks(adaptive_config, generation_config, objective_profile)
        drafts: list[_RecipeDraft] = []
        parent_used = any(left.name in parent_fields or right.name in parent_fields for left, right in pairs)
        for left, right in pairs:
            pair_key = _pair_key(left, right)
            if registry.contains("rank"):
                drafts.append(
                    _RecipeDraft(
                        expression=f"rank({left.name}) - rank({right.name})",
                        recipe_variant="rank_spread",
                        fields=(left.name, right.name),
                        pair_key=pair_key,
                    )
                )
            if registry.contains("ts_mean") and registry.contains("rank"):
                for lookback in lookbacks[:2]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"rank(ts_mean({left.name},{lookback})) - rank(ts_mean({right.name},{lookback}))",
                            recipe_variant="rank_ts_mean_spread",
                            fields=(left.name, right.name),
                            pair_key=pair_key,
                        )
                    )
            if registry.contains("ts_av_diff") and registry.contains("rank"):
                for lookback in lookbacks[:1]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"rank(ts_av_diff(({left.name}-{right.name}),{lookback}))",
                            recipe_variant="rank_ts_av_diff_spread",
                            fields=(left.name, right.name),
                            pair_key=pair_key,
                        )
                    )
            if recipe_family == "returns_term_structure" and registry.contains("reverse") and registry.contains("rank"):
                drafts.append(
                    _RecipeDraft(
                        expression=f"rank(reverse({left.name})) - rank(reverse({right.name}))",
                        recipe_variant="reverse_rank_term_spread",
                        fields=(left.name, right.name),
                        pair_key=pair_key,
                    )
                )
            if recipe_family == "returns_term_structure" and registry.contains("ts_arg_max") and registry.contains("ts_arg_min"):
                for lookback in lookbacks[:1]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"rank(ts_arg_max({left.name},{lookback})) - rank(ts_arg_min({right.name},{lookback}))",
                            recipe_variant="arg_extreme_term_spread",
                            fields=(left.name, right.name),
                            pair_key=pair_key,
                        )
                    )
        return {
            "drafts": _unique_drafts(drafts)[: int(config.max_drafts_per_bucket)],
            "parent_used": parent_used,
        }

    def _revision_surprise_drafts(
        self,
        *,
        objective_profile: str,
        field_pool: list[FieldSpec],
        field_penalty_multipliers: dict[str, float],
        parent_fields: set[str],
        recent_field_usage_counts: Counter[str],
        round_index: int,
        config: RecipeGenerationConfig,
        registry: OperatorRegistry,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        ordered_fields = _ordered_fields_for_side(
            field_pool=field_pool,
            recipe_family="revision_surprise",
            side="primary",
            field_penalty_multipliers=field_penalty_multipliers,
            parent_fields=parent_fields,
            recent_field_usage_counts=recent_field_usage_counts,
        )
        ordered_fields = _rotated_top_fields(
            ordered_fields,
            limit=int(config.max_field_candidates_per_side),
            round_index=round_index if bool(config.enable_field_rotation) else 0,
        )
        lookbacks = _ordered_lookbacks(generation_config.lookbacks, objective_profile)
        drafts: list[_RecipeDraft] = []
        parent_used = any(spec.name in parent_fields for spec in ordered_fields)
        for field in ordered_fields:
            if objective_profile == "quality" and registry.contains("ts_mean"):
                for lookback in lookbacks[:2]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"rank(ts_mean({field.name},{lookback}))",
                            recipe_variant="rank_ts_mean",
                            fields=(field.name,),
                        )
                    )
            if objective_profile != "low_turnover":
                drafts.append(
                    _RecipeDraft(
                        expression=f"rank({field.name})",
                        recipe_variant="rank_raw",
                        fields=(field.name,),
                    )
                )
                if registry.contains("ts_delta"):
                    for lookback in lookbacks[:1]:
                        drafts.append(
                            _RecipeDraft(
                                expression=f"rank(ts_delta({field.name},{lookback}))",
                                recipe_variant="rank_ts_delta",
                                fields=(field.name,),
                            )
                        )
            if registry.contains("ts_mean"):
                for lookback in lookbacks[:2]:
                    drafts.append(
                        _RecipeDraft(
                            expression=f"rank(ts_mean({field.name},{lookback}))",
                            recipe_variant="rank_ts_mean",
                            fields=(field.name,),
                        )
                    )
        return {
            "drafts": _unique_drafts(drafts)[: int(config.max_drafts_per_bucket)],
            "parent_used": parent_used,
        }


def _all_buckets(config: RecipeGenerationConfig) -> list[_RecipeBucket]:
    buckets: list[_RecipeBucket] = []
    for recipe_family in config.enabled_recipe_families:
        template_name = _RECIPE_TEMPLATE_BY_FAMILY.get(recipe_family)
        if not template_name:
            continue
        dataset_family = _dataset_family_for_recipe(recipe_family)
        for objective_profile in config.objective_profiles:
            buckets.append(
                _RecipeBucket(
                    search_bucket_id=f"{recipe_family}|{dataset_family}|{objective_profile}",
                    recipe_family=recipe_family,
                    dataset_family=dataset_family,
                    objective_profile=objective_profile,
                    template_name=template_name,
                )
            )
    return buckets


def _dataset_family_for_recipe(recipe_family: str) -> str:
    return str(_RECIPE_DATASET_FAMILY_BY_FAMILY.get(recipe_family, "fundamental"))


def _active_buckets(config: RecipeGenerationConfig, round_index: int) -> list[_RecipeBucket]:
    buckets = _all_buckets(config)
    if not buckets:
        return []
    active_count = max(1, min(int(config.active_bucket_count), len(buckets)))
    offset = (int(round_index) * active_count) % len(buckets)
    return [buckets[(offset + index) % len(buckets)] for index in range(active_count)]


def _resolved_group_key_names(
    *,
    field_registry: FieldRegistry,
    generation_config: GenerationConfig,
) -> list[str]:
    include_catalog = bool(getattr(generation_config, "allow_catalog_fields_without_runtime", False))
    group_specs = field_registry.generation_group_key_fields(include_catalog_fields=include_catalog)
    by_name = {
        spec.name: spec
        for spec in group_specs
        if spec.operator_type == "group"
        and spec.name in {*_GROUP_RELATIVE_PRIMARY_GROUP_KEYS, *_GROUP_RELATIVE_FALLBACK_GROUP_KEYS}
    }
    return [
        name
        for name in (*_GROUP_RELATIVE_PRIMARY_GROUP_KEYS, *_GROUP_RELATIVE_FALLBACK_GROUP_KEYS)
        if name in by_name
    ]


def _weighted_group_key_cycle(group_keys: list[str]) -> list[str]:
    available = set(group_keys)
    weighted = [name for name in _GROUP_RELATIVE_WEIGHTED_PRIMARY_KEYS if name in available]
    if weighted:
        return weighted
    return [
        name
        for name in (*_GROUP_RELATIVE_PRIMARY_GROUP_KEYS, *_GROUP_RELATIVE_FALLBACK_GROUP_KEYS)
        if name in available
    ]


def _resolved_group_matrix_field_names(
    *,
    candidate_names: tuple[str, ...],
    field_registry: FieldRegistry,
    generation_config: GenerationConfig,
    blocked_fields: set[str],
    field_penalty_multipliers: dict[str, float],
    round_index: int,
) -> list[str]:
    configured_allowed_fields = set(generation_config.allowed_fields or [])
    ranked: list[tuple[float, int, str]] = []
    for index, name in enumerate(candidate_names):
        spec = field_registry.fields.get(name)
        if spec is None:
            continue
        if spec.operator_type != "matrix":
            continue
        if name in blocked_fields:
            continue
        if configured_allowed_fields and name not in configured_allowed_fields:
            continue
        ranked.append(
            (
                _field_priority_score(spec, field_penalty_multipliers=field_penalty_multipliers),
                index,
                name,
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return _rotated_names([name for _, _, name in ranked], round_index=round_index)


def _resolved_single_group_matrix_field(
    name: str,
    *,
    field_registry: FieldRegistry,
    generation_config: GenerationConfig,
    blocked_fields: set[str],
) -> str:
    spec = field_registry.fields.get(name)
    configured_allowed_fields = set(generation_config.allowed_fields or [])
    if spec is None or spec.operator_type != "matrix":
        return ""
    if name in blocked_fields:
        return ""
    if configured_allowed_fields and name not in configured_allowed_fields:
        return ""
    return name


def _rotated_names(names: list[str], *, round_index: int) -> list[str]:
    if len(names) <= 1:
        return names
    offset = int(round_index) % len(names)
    return names[offset:] + names[:offset]


def _group_a_relative_fundamental_drafts(
    *,
    fields: list[str],
    group_keys: list[str],
    registry: OperatorRegistry,
) -> list[_RecipeDraft]:
    if not fields or not group_keys:
        return []
    drafts: list[_RecipeDraft] = []
    for index, field in enumerate(fields):
        group_key = group_keys[index % len(group_keys)]
        if _registry_contains_all(registry, ("rank", "group_zscore")):
            drafts.append(
                _group_relative_draft(
                    expression=f"rank(group_zscore({field},{group_key}))",
                    recipe_variant="group_a_rank_group_zscore_fundamental",
                    group_name="A",
                    primary_field=field,
                    group_key=group_key,
                )
            )
        if _registry_contains_all(registry, ("group_rank",)):
            drafts.append(
                _group_relative_draft(
                    expression=f"group_rank({field},{group_key})",
                    recipe_variant="group_a_group_rank_fundamental",
                    group_name="A",
                    primary_field=field,
                    group_key=group_key,
                )
            )
        if _registry_contains_all(registry, ("zscore", "group_rank")):
            drafts.append(
                _group_relative_draft(
                    expression=f"zscore(group_rank({field},{group_key}))",
                    recipe_variant="group_a_zscore_group_rank_fundamental",
                    group_name="A",
                    primary_field=field,
                    group_key=group_key,
                )
            )
    return drafts


def _group_b_momentum_drafts(
    *,
    returns_field: str,
    close_field: str,
    group_keys: list[str],
    registry: OperatorRegistry,
) -> list[_RecipeDraft]:
    if not group_keys:
        return []
    drafts: list[_RecipeDraft] = []
    if returns_field and _registry_contains_all(registry, ("rank", "group_neutralize", "ts_mean")):
        for index, lookback in enumerate(_GROUP_RELATIVE_LOOKBACKS):
            group_key = group_keys[index % len(group_keys)]
            drafts.append(
                _group_relative_draft(
                    expression=f"rank(group_neutralize(ts_mean({returns_field},{lookback}),{group_key}))",
                    recipe_variant="group_b_rank_group_neutralize_ts_mean_returns",
                    group_name="B",
                    primary_field=returns_field,
                    group_key=group_key,
                )
            )
    if returns_field and _registry_contains_all(registry, ("rank", "group_neutralize", "ts_decay_linear")):
        for index, lookback in enumerate(_GROUP_RELATIVE_LOOKBACKS):
            group_key = group_keys[(index + 1) % len(group_keys)]
            drafts.append(
                _group_relative_draft(
                    expression=f"rank(group_neutralize(ts_decay_linear({returns_field},{lookback}),{group_key}))",
                    recipe_variant="group_b_rank_group_neutralize_ts_decay_returns",
                    group_name="B",
                    primary_field=returns_field,
                    group_key=group_key,
                )
            )
    if close_field and _registry_contains_all(registry, ("group_neutralize", "ts_rank")):
        for index, lookback in enumerate(_GROUP_RELATIVE_LOOKBACKS):
            group_key = group_keys[(index + 2) % len(group_keys)]
            drafts.append(
                _group_relative_draft(
                    expression=f"group_neutralize(ts_rank({close_field},{lookback}),{group_key})",
                    recipe_variant="group_b_group_neutralize_ts_rank_close",
                    group_name="B",
                    primary_field=close_field,
                    group_key=group_key,
                )
            )
    return drafts


def _group_c_earnings_drafts(
    *,
    fields: list[str],
    group_keys: list[str],
    subindustry_key: str,
    registry: OperatorRegistry,
) -> list[_RecipeDraft]:
    if not fields or not group_keys:
        return []
    drafts: list[_RecipeDraft] = []
    for field_index, field in enumerate(fields):
        for window_index, short_window in enumerate(_GROUP_RELATIVE_SHORT_WINDOWS):
            group_key = group_keys[(field_index + window_index) % len(group_keys)]
            if _registry_contains_all(registry, ("group_zscore", "ts_delta")):
                drafts.append(
                    _group_relative_draft(
                        expression=f"group_zscore(ts_delta({field},{short_window}),{group_key})",
                        recipe_variant="group_c_group_zscore_ts_delta_eps",
                        group_name="C",
                        primary_field=field,
                        group_key=group_key,
                    )
                )
            if _registry_contains_all(registry, ("rank", "group_neutralize", "ts_delta")):
                drafts.append(
                    _group_relative_draft(
                        expression=f"rank(group_neutralize(ts_delta({field},{short_window}),{group_key}))",
                        recipe_variant="group_c_rank_group_neutralize_ts_delta_eps",
                        group_name="C",
                        primary_field=field,
                        group_key=group_key,
                    )
                )
        if subindustry_key and _registry_contains_all(registry, ("group_rank", "ts_mean")):
            drafts.append(
                _group_relative_draft(
                    expression=f"group_rank(ts_mean({field},20),{subindustry_key})",
                    recipe_variant="group_c_group_rank_ts_mean_eps_subindustry",
                    group_name="C",
                    primary_field=field,
                    group_key=subindustry_key,
                )
            )
    return drafts


def _group_d_liquidity_drafts(
    *,
    volume_field: str,
    group_keys: list[str],
    subindustry_key: str,
    registry: OperatorRegistry,
) -> list[_RecipeDraft]:
    if not volume_field or not group_keys:
        return []
    drafts: list[_RecipeDraft] = []
    if _registry_contains_all(registry, ("group_zscore", "ts_mean")):
        for index, lookback in enumerate(_GROUP_RELATIVE_LOOKBACKS):
            group_key = group_keys[index % len(group_keys)]
            drafts.append(
                _group_relative_draft(
                    expression=f"group_zscore(ts_mean({volume_field},{lookback}),{group_key})",
                    recipe_variant="group_d_group_zscore_ts_mean_volume",
                    group_name="D",
                    primary_field=volume_field,
                    group_key=group_key,
                )
            )
    if _registry_contains_all(registry, ("rank", "group_neutralize", "ts_rank")):
        for index, lookback in enumerate(_GROUP_RELATIVE_LOOKBACKS):
            group_key = group_keys[(index + 1) % len(group_keys)]
            drafts.append(
                _group_relative_draft(
                    expression=f"rank(group_neutralize(ts_rank({volume_field},{lookback}),{group_key}))",
                    recipe_variant="group_d_rank_group_neutralize_ts_rank_volume",
                    group_name="D",
                    primary_field=volume_field,
                    group_key=group_key,
                )
            )
    if subindustry_key and _registry_contains_all(registry, ("group_rank",)):
        drafts.append(
            _group_relative_draft(
                expression=f"group_rank({volume_field},{subindustry_key})",
                recipe_variant="group_d_group_rank_volume_subindustry",
                group_name="D",
                primary_field=volume_field,
                group_key=subindustry_key,
            )
        )
    return drafts


def _group_relative_draft(
    *,
    expression: str,
    recipe_variant: str,
    group_name: str,
    primary_field: str,
    group_key: str,
) -> _RecipeDraft:
    return _RecipeDraft(
        expression=expression,
        recipe_variant=recipe_variant,
        fields=(primary_field, group_key),
        source=GROUP_RELATIVE_SOURCE,
        group_recipe_group=group_name,
        primary_field=primary_field,
        group_key=group_key,
    )


def _interleave_group_drafts(drafts_by_group: dict[str, list[_RecipeDraft]]) -> list[_RecipeDraft]:
    ordered_groups = ("A", "B", "C", "D")
    max_len = max((len(drafts_by_group.get(group, [])) for group in ordered_groups), default=0)
    interleaved: list[_RecipeDraft] = []
    for index in range(max_len):
        for group in ordered_groups:
            drafts = drafts_by_group.get(group, [])
            if index < len(drafts):
                interleaved.append(drafts[index])
    return _unique_drafts(interleaved)


def _registry_contains_all(registry: OperatorRegistry, names: tuple[str, ...]) -> bool:
    return all(registry.contains(name) for name in names)


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
    recent_field_usage_counts: Counter[str],
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
        recent_usage = int(recent_field_usage_counts.get(spec.name, 0))
        if recent_usage > 0:
            score /= 1.0 + 0.35 * float(recent_usage)
        if keywords:
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > 0:
                score *= 1.0 + min(1.2, 0.40 * matches)
        if recipe_family == "revision_surprise" and spec.category == "analyst":
            score *= 1.20
        scored.append((score, spec))
    scored.sort(key=lambda item: (item[0], item[1].coverage, item[1].name), reverse=True)
    return [spec for _, spec in scored]


def _rotated_top_fields(fields: list[FieldSpec], *, limit: int, round_index: int) -> list[FieldSpec]:
    selected = list(fields[: max(1, int(limit))])
    if len(selected) <= 1:
        return selected
    offset = int(round_index) % len(selected)
    return selected[offset:] + selected[:offset]


def _distinct_pairs(
    left_fields: list[FieldSpec],
    right_fields: list[FieldSpec],
    *,
    limit: int,
    pair_usage_counts: Counter[str],
) -> list[tuple[FieldSpec, FieldSpec]]:
    pairs: list[tuple[int, int, str, FieldSpec, FieldSpec]] = []
    seen: set[str] = set()
    rank_index = 0
    for left in left_fields:
        for right in right_fields:
            if left.name == right.name:
                continue
            pair_key = _pair_key(left, right)
            if pair_key in seen:
                continue
            seen.add(pair_key)
            pairs.append((int(pair_usage_counts.get(pair_key, 0)), rank_index, pair_key, left, right))
            rank_index += 1
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(left, right) for _, _, _, left, right in pairs[: max(1, int(limit))]]


def _pair_key(left: FieldSpec, right: FieldSpec) -> str:
    return f"{left.name}:{right.name}"


def _ordered_lookbacks(lookbacks: list[int], objective_profile: str) -> list[int]:
    unique = list(dict.fromkeys(int(value) for value in lookbacks if int(value) > 0))
    if not unique:
        return [5]
    if objective_profile in {"quality", "low_turnover"}:
        return sorted(unique, reverse=True)
    return unique


def _elite_recipe_lookbacks(
    adaptive_config: AdaptiveGenerationConfig,
    generation_config: GenerationConfig,
    objective_profile: str,
) -> list[int]:
    elite_lookbacks = list(getattr(adaptive_config.elite_motifs, "lookbacks", []) or [])
    if elite_lookbacks:
        return _ordered_lookbacks(elite_lookbacks, objective_profile)
    return _ordered_lookbacks(generation_config.lookbacks, objective_profile)


def _parent_meets_base_thresholds(metrics: dict[str, Any]) -> bool:
    return (
        _to_float(metrics.get("fitness")) >= 0.02
        and _to_float(metrics.get("sharpe")) >= 0.03
        and _to_float(metrics.get("drawdown")) <= 0.75
        and not str(metrics.get("rejection_reason") or "").strip()
        and not has_structural_risk_blocker(
            tuple(metrics.get("hard_fail_checks") or ()),
            tuple(metrics.get("blocking_warning_checks") or ()),
        )
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


def _bucket_bias(*, config: RecipeGenerationConfig, bucket_id: str) -> float:
    return max(1e-6, float((config.bucket_biases or {}).get(bucket_id, 1.0) or 1.0))


def _apply_bucket_bias(*, score: float, config: RecipeGenerationConfig, bucket_id: str) -> float:
    return float(score) * _bucket_bias(config=config, bucket_id=bucket_id)


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


def _decode_json_list(payload: Any) -> list[str]:
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    if isinstance(payload, str) and payload.strip():
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded if str(item).strip()]
    return []


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
