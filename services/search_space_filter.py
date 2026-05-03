from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any

from alpha.ast_nodes import BinaryOpNode, FunctionCallNode, IdentifierNode, UnaryOpNode
from alpha.parser import parse_expression
from core.config import AppConfig, GenerationConfig
from core.brain_checks import (
    classify_timeout_cause,
    has_structural_risk_blocker,
    parse_names_json,
    robustness_check_names,
    structural_risk_check_names,
)
from data.field_registry import FieldRegistry
from storage.repository import SQLiteRepository


logger = logging.getLogger(__name__)


_TEMPLATE_OPERATORS: dict[str, tuple[str, ...]] = {
    "momentum": ("rank", "ts_mean"),
    "mean_reversion": ("rank", "ts_mean"),
    "volatility": ("rank", "ts_std_dev"),
    "ratio": ("rank", "ts_mean"),
    "delta": ("rank", "ts_delta"),
    "cross_field_spread": ("rank",),
    "correlation_pair": ("rank", "ts_corr", "ts_covariance"),
    "group_relative": ("rank", "group_neutralize"),
    "smoothed_momentum": ("rank", "ts_decay_linear", "ts_delta"),
    "fundamental_vs_price_confirmation": ("rank", "ts_mean", "ts_delta"),
}

_OPERATIONAL_REJECTION_MARKERS = (
    "timeout",
    "poll_timeout_after_downtime",
    "persona",
    "telegram",
    "network",
    "connection",
    "remote disconnected",
    "connecttimeout",
    "read timed out",
)
_ROBUSTNESS_CHECK_FIELD_MIN_SUPPORT = 8
_ROBUSTNESS_CHECK_FIELD_MULTIPLIER = 0.85
_WINNER_PRIOR_CACHE_LOCK = RLock()
_WINNER_PRIOR_CACHE: dict[tuple[Any, ...], tuple[float, "_WinnerPriorResult"]] = {}


@dataclass(frozen=True, slots=True)
class _WinnerPriorResult:
    field_multipliers: dict[str, float]
    operator_multipliers: dict[str, float]
    promoted_field_counts: Counter[str]
    demoted_field_counts: Counter[str]
    promoted_operator_counts: Counter[str]
    demoted_operator_counts: Counter[str]
    timeout_cause_counts: Counter[str]
    timeout_demoted_field_counts: Counter[str]
    timeout_demoted_operator_counts: Counter[str]
    source: str = "none"
    lookback_rounds: int = 0
    recent_quality_count: int = 0
    all_time_quality_count: int = 0
    fields_with_prior: int = 0
    fields_neutral: int = 0
    avg_boost: float = 1.0
    avg_penalty: float = 1.0


@dataclass(frozen=True, slots=True)
class _WinnerPriorObservation:
    field_labels: dict[str, Counter[str]]
    operator_labels: dict[str, Counter[str]]
    field_support: Counter[str]
    operator_support: Counter[str]
    timeout_cause_counts: Counter[str]
    timeout_field_counts: Counter[str]
    timeout_operator_counts: Counter[str]
    quality_count: int


def _empty_winner_prior_result() -> _WinnerPriorResult:
    return _WinnerPriorResult({}, {}, Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter())


def invalidate_winner_prior_cache(run_id: str | None = None) -> None:
    """Clear cached winner priors after BRAIN results are persisted."""
    with _WINNER_PRIOR_CACHE_LOCK:
        if run_id is None:
            _WINNER_PRIOR_CACHE.clear()
            return
        normalized_run_id = str(run_id)
        for key in list(_WINNER_PRIOR_CACHE):
            if len(key) > 1 and key[1] == normalized_run_id:
                _WINNER_PRIOR_CACHE.pop(key, None)


@dataclass(slots=True)
class SearchSpaceFilterContext:
    enabled: bool
    field_registry: FieldRegistry
    active_field_names: set[str]
    hard_blocked_fields: set[str]
    soft_penalized_fields: set[str]
    field_multipliers: dict[str, float]
    operator_multipliers: dict[str, float]
    lane_field_pools: dict[str, set[str]]
    lane_operator_allowlists: dict[str, set[str]]
    metrics: dict[str, Any]

    def field_registry_for_lane(self, lane: str) -> FieldRegistry:
        pool = self.lane_field_pools.get(lane)
        if not pool:
            return self.field_registry
        fields = {
            name: spec
            for name, spec in self.field_registry.fields.items()
            if name in pool or spec.operator_type == "group"
        }
        return FieldRegistry(fields=fields)

    def generation_config_for_lane(self, generation_config: GenerationConfig, lane: str) -> GenerationConfig:
        allowed_fields = list(generation_config.allowed_fields or [])
        pool = self.lane_field_pools.get(lane)
        if pool:
            if allowed_fields:
                allowed_fields = [field for field in allowed_fields if field in pool]
            else:
                allowed_fields = sorted(pool)

        allowed_operators = list(generation_config.allowed_operators or [])
        operator_allowlist = self.lane_operator_allowlists.get(lane)
        if operator_allowlist:
            allowed_operators = [operator for operator in allowed_operators if operator in operator_allowlist]

        template_weights = dict(generation_config.template_weights or {})
        if self.operator_multipliers:
            template_weights = _penalized_template_weights(template_weights, self.operator_multipliers)

        return replace(
            generation_config,
            allowed_fields=allowed_fields,
            allowed_operators=allowed_operators,
            template_weights=template_weights,
        )

    def expression_allowed_for_lane(self, expression: str, lane: str) -> bool:
        fields, operators = expression_fields_and_operators(expression)
        if fields and not fields.issubset(self.active_field_names):
            return False
        operator_allowlist = self.lane_operator_allowlists.get(lane)
        if operator_allowlist and operators and not operators.issubset(operator_allowlist):
            return False
        return True

    def to_metrics(self) -> dict[str, Any]:
        return dict(self.metrics)


def build_search_space_filter_context(
    *,
    repository: SQLiteRepository,
    config: AppConfig,
    field_registry: FieldRegistry,
    run_id: str,
    round_index: int,
    blocked_fields: set[str],
) -> SearchSpaceFilterContext:
    filter_config = config.adaptive_generation.search_space_filter
    base_active_fields = field_registry.generation_allowed_fields(
        config.generation.allowed_fields,
        include_catalog_fields=config.generation.allow_catalog_fields_without_runtime,
    )
    if not bool(filter_config.enabled):
        return SearchSpaceFilterContext(
            enabled=False,
            field_registry=field_registry,
            active_field_names=set(base_active_fields),
            hard_blocked_fields=set(blocked_fields),
            soft_penalized_fields=set(),
            field_multipliers={},
            operator_multipliers={},
            lane_field_pools={},
            lane_operator_allowlists={},
            metrics={
                "search_space_filter_enabled": False,
                "search_space_filter_active_field_count": len(base_active_fields),
                "search_space_filter_hard_blocked_field_count": len(blocked_fields),
                "search_space_filter_soft_penalized_field_count": 0,
                "search_space_filter_lane_field_pool_counts": {},
                "search_space_filter_lane_field_pool_before_counts": {},
                "search_space_filter_lane_field_pool_after_counts": {},
                "search_space_filter_operator_penalty_counts": {},
                "search_space_filter_field_hard_fail_penalty_counts": {},
                "search_space_filter_operator_hard_fail_penalty_counts": {},
                "search_space_filter_field_blocking_warning_penalty_counts": {},
                "search_space_filter_operator_blocking_warning_penalty_counts": {},
                "search_space_filter_field_robustness_penalty_counts": {},
                "search_space_filter_promoted_field_count": 0,
                "search_space_filter_demoted_field_count": 0,
                "search_space_filter_promoted_operator_count": 0,
                "search_space_filter_demoted_operator_count": 0,
                "search_space_filter_top_promoted_fields": [],
                "search_space_filter_top_promoted_operators": [],
                "search_space_filter_timeout_cause_counts": {},
                "search_space_filter_timeout_prior_demoted_field_counts": {},
                "search_space_filter_timeout_prior_demoted_operator_counts": {},
                "search_space_filter_diagnostic_field_multiplier_count": 0,
                "search_space_filter_top_diagnostic_fields": [],
                "search_space_filter_field_floor_activation_count": 0,
                "search_space_filter_field_floor_activated_fields": [],
                "search_space_filter_operator_floor_activation_count": 0,
                "hard_blocked_field_count": len(blocked_fields),
                "soft_penalized_field_count": 0,
                "lane_field_pool_counts": {},
                "lane_field_pool_before_counts": {},
                "lane_field_pool_after_counts": {},
                "operator_penalty_counts": {},
            },
        )

    validation_counts = _recent_validation_field_counts(
        repository=repository,
        run_id=run_id,
        round_index=round_index,
        lookback_rounds=int(filter_config.completed_lookback_rounds),
    )
    (
        field_result_penalties,
        operator_result_penalties,
        field_hard_fail_penalties,
        operator_hard_fail_penalties,
        field_blocking_warning_penalties,
        operator_blocking_warning_penalties,
        field_robustness_penalties,
    ) = _completed_result_penalties(
        repository=repository,
        run_id=run_id,
        round_index=round_index,
        lookback_rounds=int(filter_config.completed_lookback_rounds),
        min_support=int(filter_config.min_completed_support),
        check_penalty_min_support=int(filter_config.check_penalty_min_support),
        sharpe_floor=float(filter_config.sharpe_floor),
        fitness_floor=float(filter_config.fitness_floor),
    )
    winner_prior = _completed_result_priors(
        repository=repository,
        config=config,
        run_id=run_id,
        round_index=round_index,
    )
    diagnostic_multipliers = _field_diagnostic_multipliers(
        repository=repository,
        config=config,
        run_id=run_id,
    )
    updated_fields = {}
    soft_penalized_fields: set[str] = set()
    field_multipliers: dict[str, float] = {}
    field_floor_activation_count = 0
    field_floor_activated_fields: list[str] = []
    for name, spec in field_registry.fields.items():
        if name in blocked_fields:
            continue
        multiplier = _profile_multiplier(spec, config=config)
        if validation_counts.get(name, 0) >= int(filter_config.validation_field_min_count):
            multiplier *= float(filter_config.validation_field_multiplier)
        if name in field_result_penalties:
            multiplier *= float(filter_config.field_result_multiplier)
        if name in field_hard_fail_penalties:
            multiplier *= float(filter_config.hard_fail_field_multiplier)
        if name in field_blocking_warning_penalties:
            multiplier *= float(filter_config.blocking_warning_field_multiplier)
        if name in field_robustness_penalties:
            multiplier *= _ROBUSTNESS_CHECK_FIELD_MULTIPLIER
        if name in winner_prior.field_multipliers:
            multiplier *= float(winner_prior.field_multipliers[name])
        if name in diagnostic_multipliers:
            multiplier *= float(diagnostic_multipliers[name])
        multiplier = max(1e-6, float(multiplier))
        base_score = max(1e-6, float(spec.field_score or 0.0))
        raw_score = base_score * multiplier
        floor = compute_field_floor(spec, filter_config)
        final_score = max(raw_score, floor)
        if raw_score < floor:
            field_floor_activation_count += 1
            field_floor_activated_fields.append(name)
            logger.debug(
                "Floor activated for field %s: raw=%.3f floor=%.3f catalog=%.3f",
                name,
                raw_score,
                floor,
                float(spec.field_score or 0.0),
            )
        effective_multiplier = final_score / base_score
        if effective_multiplier < 0.999999:
            soft_penalized_fields.add(name)
        if abs(effective_multiplier - 1.0) > 1e-9:
            field_multipliers[name] = round(effective_multiplier, 6)
        updated_fields[name] = replace(spec, field_score=max(1e-6, final_score))

    filtered_registry = FieldRegistry(fields=updated_fields)
    active_fields = filtered_registry.generation_allowed_fields(
        config.generation.allowed_fields,
        include_catalog_fields=config.generation.allow_catalog_fields_without_runtime,
    )
    active_numeric_fields = filtered_registry.generation_numeric_fields(
        config.generation.allowed_fields,
        include_catalog_fields=config.generation.allow_catalog_fields_without_runtime,
    )
    if not active_numeric_fields:
        raise ValueError("search_space_filter removed all numeric generation fields")
    lane_field_pools, lane_field_pool_before_counts = _lane_field_pools(
        field_registry=filtered_registry,
        catalog_field_registry=field_registry,
        config=config,
        active_fields=active_fields,
    )
    lane_operator_allowlists = {
        lane: set(values)
        for lane, values in dict(filter_config.lane_operator_allowlists or {}).items()
        if values
    }
    operator_multipliers = {
        operator: float(filter_config.operator_result_multiplier)
        for operator in operator_result_penalties
    }
    for operator in operator_hard_fail_penalties:
        operator_multipliers[operator] = round(
            float(operator_multipliers.get(operator, 1.0)) * float(filter_config.hard_fail_operator_multiplier),
            6,
        )
    for operator in operator_blocking_warning_penalties:
        operator_multipliers[operator] = round(
            float(operator_multipliers.get(operator, 1.0))
            * float(filter_config.blocking_warning_operator_multiplier),
            6,
        )
    for operator, multiplier in winner_prior.operator_multipliers.items():
        operator_multipliers[operator] = round(float(operator_multipliers.get(operator, 1.0)) * float(multiplier), 6)
    operator_floor_activation_count = 0
    for operator, raw_multiplier in list(operator_multipliers.items()):
        floored_multiplier = apply_operator_floor(raw_multiplier, filter_config)
        if floored_multiplier > float(raw_multiplier):
            operator_floor_activation_count += 1
        operator_multipliers[operator] = round(floored_multiplier, 6)
    lane_field_pool_counts = {
        lane: len(pool) for lane, pool in sorted(lane_field_pools.items())
    }
    operator_penalty_counts = dict(sorted(operator_result_penalties.items()))
    field_result_penalty_counts = dict(sorted(field_result_penalties.items()))
    field_hard_fail_penalty_counts = dict(sorted(field_hard_fail_penalties.items()))
    operator_hard_fail_penalty_counts = dict(sorted(operator_hard_fail_penalties.items()))
    field_blocking_warning_penalty_counts = dict(sorted(field_blocking_warning_penalties.items()))
    operator_blocking_warning_penalty_counts = dict(sorted(operator_blocking_warning_penalties.items()))
    field_robustness_penalty_counts = dict(sorted(field_robustness_penalties.items()))
    validation_field_penalty_counts = dict(sorted(validation_counts.items()))
    top_promoted_fields = _top_prior_items(winner_prior.field_multipliers, winner_prior.promoted_field_counts)
    top_promoted_operators = _top_prior_items(winner_prior.operator_multipliers, winner_prior.promoted_operator_counts)
    if field_floor_activation_count:
        logger.debug(
            "Search space field floor activations: count=%d fields=%s",
            field_floor_activation_count,
            field_floor_activated_fields[:25],
        )
    return SearchSpaceFilterContext(
        enabled=True,
        field_registry=filtered_registry,
        active_field_names=set(active_fields),
        hard_blocked_fields=set(blocked_fields),
        soft_penalized_fields=soft_penalized_fields,
        field_multipliers=field_multipliers,
        operator_multipliers=operator_multipliers,
        lane_field_pools=lane_field_pools,
        lane_operator_allowlists=lane_operator_allowlists,
        metrics={
            "search_space_filter_enabled": True,
            "search_space_filter_active_field_count": len(active_fields),
            "search_space_filter_hard_blocked_field_count": len(blocked_fields),
            "search_space_filter_soft_penalized_field_count": len(soft_penalized_fields),
            "search_space_filter_lane_field_pool_counts": lane_field_pool_counts,
            "search_space_filter_lane_field_pool_before_counts": lane_field_pool_before_counts,
            "search_space_filter_lane_field_pool_after_counts": lane_field_pool_counts,
            "search_space_filter_operator_penalty_counts": operator_penalty_counts,
            "search_space_filter_field_result_penalty_counts": field_result_penalty_counts,
            "search_space_filter_field_hard_fail_penalty_counts": field_hard_fail_penalty_counts,
            "search_space_filter_operator_hard_fail_penalty_counts": operator_hard_fail_penalty_counts,
            "search_space_filter_field_blocking_warning_penalty_counts": field_blocking_warning_penalty_counts,
            "search_space_filter_operator_blocking_warning_penalty_counts": operator_blocking_warning_penalty_counts,
            "search_space_filter_field_robustness_penalty_counts": field_robustness_penalty_counts,
            "search_space_filter_validation_field_penalty_counts": validation_field_penalty_counts,
            "search_space_filter_promoted_field_count": len(winner_prior.promoted_field_counts),
            "search_space_filter_demoted_field_count": len(winner_prior.demoted_field_counts),
            "search_space_filter_promoted_operator_count": len(winner_prior.promoted_operator_counts),
            "search_space_filter_demoted_operator_count": len(winner_prior.demoted_operator_counts),
            "search_space_filter_top_promoted_fields": top_promoted_fields,
            "search_space_filter_top_promoted_operators": top_promoted_operators,
            "search_space_filter_timeout_cause_counts": dict(sorted(winner_prior.timeout_cause_counts.items())),
            "search_space_filter_timeout_prior_demoted_field_counts": dict(
                sorted(winner_prior.timeout_demoted_field_counts.items())
            ),
            "search_space_filter_timeout_prior_demoted_operator_counts": dict(
                sorted(winner_prior.timeout_demoted_operator_counts.items())
            ),
            "search_space_filter_diagnostic_field_multiplier_count": len(diagnostic_multipliers),
            "search_space_filter_top_diagnostic_fields": _top_multiplier_items(diagnostic_multipliers),
            "search_space_filter_field_floor_activation_count": field_floor_activation_count,
            "search_space_filter_field_floor_activated_fields": field_floor_activated_fields[:25],
            "search_space_filter_operator_floor_activation_count": operator_floor_activation_count,
            "hard_blocked_field_count": len(blocked_fields),
            "soft_penalized_field_count": len(soft_penalized_fields),
            "lane_field_pool_counts": lane_field_pool_counts,
            "lane_field_pool_before_counts": lane_field_pool_before_counts,
            "lane_field_pool_after_counts": lane_field_pool_counts,
            "operator_penalty_counts": operator_penalty_counts,
        },
    )


def expression_fields_and_operators(expression: str) -> tuple[set[str], set[str]]:
    try:
        root = parse_expression(expression)
    except ValueError:
        return set(), set()
    fields: set[str] = set()
    operators: set[str] = set()
    _collect_expression_parts(root, fields=fields, operators=operators)
    return fields, operators


def _collect_expression_parts(node, *, fields: set[str], operators: set[str]) -> None:
    if isinstance(node, IdentifierNode):
        fields.add(node.name)
        return
    if isinstance(node, FunctionCallNode):
        operators.add(node.name)
        for arg in node.args:
            _collect_expression_parts(arg, fields=fields, operators=operators)
        return
    if isinstance(node, BinaryOpNode):
        _collect_expression_parts(node.left, fields=fields, operators=operators)
        _collect_expression_parts(node.right, fields=fields, operators=operators)
        return
    if isinstance(node, UnaryOpNode):
        _collect_expression_parts(node.operand, fields=fields, operators=operators)


def _profile_multiplier(spec, *, config: AppConfig) -> float:
    if spec.runtime_available:
        return 1.0
    filter_config = config.adaptive_generation.search_space_filter
    region = str(spec.region or "").strip().upper()
    universe = str(spec.universe or "").strip().upper()
    preferred_region = str(config.brain.region or "").strip().upper()
    preferred_universe = str(config.brain.universe or "").strip().upper()
    multiplier = 1.0
    if not region or not universe:
        multiplier *= float(filter_config.unknown_profile_multiplier)
    if region and preferred_region and region not in {preferred_region, "GLB", "GLOBAL"}:
        multiplier *= float(filter_config.profile_mismatch_multiplier)
    if universe and preferred_universe and universe != preferred_universe:
        multiplier *= float(filter_config.profile_mismatch_multiplier)
    return multiplier


def _lane_field_pools(
    *,
    field_registry: FieldRegistry,
    catalog_field_registry: FieldRegistry | None = None,
    config: AppConfig,
    active_fields: set[str],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    pools: dict[str, set[str]] = {}
    before_counts: dict[str, int] = {}
    caps = dict(config.adaptive_generation.search_space_filter.lane_field_caps or {})
    if not caps:
        return pools, before_counts
    numeric = [
        spec
        for spec in field_registry.generation_numeric_fields(
            config.generation.allowed_fields,
            include_catalog_fields=config.generation.allow_catalog_fields_without_runtime,
        )
        if spec.name in active_fields
    ]
    min_count = int(config.adaptive_generation.search_space_filter.lane_field_min_count)
    for lane, raw_cap in caps.items():
        cap = int(raw_cap)
        if cap <= 0:
            continue
        before_counts[lane] = len(numeric)
        target_count = max(cap, min_count)
        pools[lane] = _select_lane_field_pool(
            numeric,
            target_count=target_count,
            filter_config=config.adaptive_generation.search_space_filter,
            catalog_field_registry=catalog_field_registry,
        )
    return pools, before_counts


def _select_lane_field_pool(
    fields: list,
    *,
    target_count: int,
    filter_config,
    catalog_field_registry: FieldRegistry | None = None,
) -> set[str]:
    if target_count <= 0 or not fields:
        return set()
    target_count = min(int(target_count), len(fields))
    exploration_count = min(
        target_count,
        max(0, int(target_count * float(getattr(filter_config, "exploration_budget_pct", 0.15) or 0.0))),
    )
    exploit_count = max(0, target_count - exploration_count)
    adjusted_ranked = sorted(
        fields,
        key=lambda spec: (float(spec.field_score or 0.0), float(spec.coverage or 0.0), int(spec.alpha_usage_count or 0), spec.name),
        reverse=True,
    )
    selected: list[str] = []
    for spec in adjusted_ranked[:exploit_count]:
        selected.append(spec.name)
    selected_names = set(selected)
    if exploration_count > 0:
        catalog_ranked = sorted(
            fields,
            key=lambda spec: (
                _catalog_field_score(spec.name, spec, catalog_field_registry),
                float(spec.coverage or 0.0),
                int(spec.alpha_usage_count or 0),
                spec.name,
            ),
            reverse=True,
        )
        for spec in catalog_ranked:
            if len(selected) >= target_count:
                break
            if spec.name in selected_names:
                continue
            selected.append(spec.name)
            selected_names.add(spec.name)
    if len(selected) < target_count:
        for spec in adjusted_ranked:
            if len(selected) >= target_count:
                break
            if spec.name in selected_names:
                continue
            selected.append(spec.name)
            selected_names.add(spec.name)
    return set(selected)


def compute_field_floor(field, filter_config) -> float:
    catalog_score = max(0.0, float(getattr(field, "field_score", 0.0) or 0.0))
    ratio = float(getattr(filter_config, "field_floor_ratio", 0.30) or 0.0)
    absolute_min = float(getattr(filter_config, "field_floor_absolute_min", 0.10) or 0.0)
    return max(catalog_score * ratio, absolute_min)


def compute_field_weight(field, multiplier: float, filter_config, *, hard_blocked: bool = False) -> float:
    if hard_blocked:
        return 0.0
    raw_score = max(1e-6, float(getattr(field, "field_score", 0.0) or 0.0)) * max(0.0, float(multiplier))
    return max(raw_score, compute_field_floor(field, filter_config))


def apply_operator_floor(raw_weight: float, filter_config, *, hard_blocked: bool = False) -> float:
    if hard_blocked:
        return 0.0
    absolute_min = float(getattr(filter_config, "operator_floor_absolute_min", 0.05) or 0.0)
    return max(max(0.0, float(raw_weight)), absolute_min)


def _catalog_field_score(name: str, fallback_spec, catalog_field_registry: FieldRegistry | None) -> float:
    if catalog_field_registry is not None and catalog_field_registry.contains(name):
        return float(catalog_field_registry.get(name).field_score or 0.0)
    return float(getattr(fallback_spec, "field_score", 0.0) or 0.0)


def _field_diagnostic_multipliers(
    *,
    repository: SQLiteRepository,
    config: AppConfig,
    run_id: str,
) -> dict[str, float]:
    try:
        rows = repository.connection.execute(
            """
            SELECT field_name, diagnostic_name, coverage_ratio, long_count, short_count
            FROM field_diagnostics
            WHERE run_id = ?
              AND status = 'completed'
              AND UPPER(region) = ?
              AND UPPER(universe) = ?
              AND delay = ?
            ORDER BY updated_at DESC
            """,
            (
                run_id,
                str(config.brain.region or "").strip().upper(),
                str(config.brain.universe or "").strip().upper(),
                int(config.brain.delay or 1),
            ),
        ).fetchall()
    except Exception:
        return {}
    coverage_by_field: dict[str, list[float]] = defaultdict(list)
    update_support_by_field: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        field_name = str(row["field_name"] or "").strip()
        coverage = _to_float(row["coverage_ratio"])
        if not field_name or coverage is None:
            continue
        diagnostic_name = str(row["diagnostic_name"] or "")
        if diagnostic_name in {"raw", "non_zero_coverage"}:
            coverage_by_field[field_name].append(max(0.0, min(1.0, coverage)))
        elif diagnostic_name == "update_frequency":
            update_support_by_field[field_name].append(max(0.0, min(1.0, coverage)))
    multipliers: dict[str, float] = {}
    for field_name, values in coverage_by_field.items():
        avg_coverage = sum(values) / max(1, len(values))
        multiplier = 1.0
        if avg_coverage < 0.05:
            multiplier *= 0.25
        elif avg_coverage < 0.15:
            multiplier *= 0.60
        elif avg_coverage >= 0.35:
            multiplier *= 1.10
        updates = update_support_by_field.get(field_name, [])
        if updates and max(updates) < 0.03:
            multiplier *= 0.70
        if abs(multiplier - 1.0) > 1e-9:
            multipliers[field_name] = round(multiplier, 6)
    return multipliers


def _recent_validation_field_counts(
    *,
    repository: SQLiteRepository,
    run_id: str,
    round_index: int,
    lookback_rounds: int,
) -> Counter[str]:
    rows = repository.list_recent_generation_stage_metrics(
        run_id,
        limit=lookback_rounds,
        before_round_index=round_index if round_index > 0 else None,
    )
    counts: Counter[str] = Counter()
    for row in rows:
        metrics = _decode_json(row.get("metrics_json"))
        field_counts = metrics.get("validation_disallowed_field_counts")
        if not isinstance(field_counts, dict):
            continue
        for field_name, count in field_counts.items():
            try:
                numeric_count = int(count)
            except (TypeError, ValueError):
                continue
            if numeric_count > 0:
                counts[str(field_name)] += numeric_count
    return counts


def _completed_result_penalties(
    *,
    repository: SQLiteRepository,
    run_id: str,
    round_index: int,
    lookback_rounds: int,
    min_support: int,
    check_penalty_min_support: int,
    sharpe_floor: float,
    fitness_floor: float,
) -> tuple[Counter[str], Counter[str], Counter[str], Counter[str], Counter[str], Counter[str], Counter[str]]:
    try:
        if round_index > 0:
            rows = repository.connection.execute(
                """
                SELECT r.sharpe, r.fitness, r.hard_fail_checks_json, r.blocking_warning_checks_json,
                       a.fields_used_json, a.operators_used_json, a.generation_metadata
                FROM brain_results r
                JOIN alphas a ON a.run_id = r.run_id AND a.alpha_id = r.candidate_id
                WHERE r.run_id = ?
                  AND r.status = 'completed'
                  AND r.round_index < ?
                  AND r.sharpe IS NOT NULL
                  AND r.fitness IS NOT NULL
                ORDER BY r.round_index DESC, r.created_at DESC
                LIMIT ?
                """,
                (run_id, int(round_index), max(50, int(lookback_rounds) * 20)),
            ).fetchall()
        else:
            rows = repository.connection.execute(
                """
                SELECT r.sharpe, r.fitness, r.hard_fail_checks_json, r.blocking_warning_checks_json,
                       a.fields_used_json, a.operators_used_json, a.generation_metadata
                FROM brain_results r
                JOIN alphas a ON a.run_id = r.run_id AND a.alpha_id = r.candidate_id
                WHERE r.run_id = ?
                  AND r.status = 'completed'
                  AND r.sharpe IS NOT NULL
                  AND r.fitness IS NOT NULL
                ORDER BY r.round_index DESC, r.created_at DESC
                LIMIT ?
                """,
                (run_id, max(50, int(lookback_rounds) * 20)),
            ).fetchall()
    except Exception:
        return Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    field_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    operator_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    field_hard_fail_values: Counter[str] = Counter()
    operator_hard_fail_values: Counter[str] = Counter()
    field_blocking_warning_values: Counter[str] = Counter()
    operator_blocking_warning_values: Counter[str] = Counter()
    field_robustness_values: Counter[str] = Counter()
    for row in rows:
        sharpe = _to_float(row["sharpe"])
        fitness = _to_float(row["fitness"])
        if sharpe is None or fitness is None:
            continue
        fields = _decode_text_list(row["fields_used_json"])
        operators = _decode_text_list(row["operators_used_json"])
        if not fields or not operators:
            metadata = _decode_json(row["generation_metadata"])
            fields = fields or [str(item) for item in metadata.get("fields_used", []) if str(item)]
            operators = operators or [str(item) for item in metadata.get("operators_used", []) if str(item)]
        for field_name in dict.fromkeys(fields):
            field_values[field_name].append((sharpe, fitness))
        for operator_name in dict.fromkeys(_policy_operator_names(operators)):
            operator_values[operator_name].append((sharpe, fitness))
        hard_fail_checks = parse_names_json(row["hard_fail_checks_json"])
        blocking_warning_checks = parse_names_json(row["blocking_warning_checks_json"])
        hard_structural_checks = structural_risk_check_names(hard_fail_checks)
        blocking_structural_checks = structural_risk_check_names(blocking_warning_checks)
        robustness_checks = tuple(
            dict.fromkeys(
                robustness_check_names(hard_fail_checks)
                + robustness_check_names(blocking_warning_checks)
            )
        )
        if hard_structural_checks:
            for field_name in dict.fromkeys(fields):
                field_hard_fail_values[field_name] += 1
        if blocking_structural_checks:
            for field_name in dict.fromkeys(fields):
                field_blocking_warning_values[field_name] += 1
        if robustness_checks:
            for field_name in dict.fromkeys(fields):
                field_robustness_values[field_name] += 1
    return (
        _underperforming_keys(field_values, min_support=min_support, sharpe_floor=sharpe_floor, fitness_floor=fitness_floor),
        Counter(),
        _counts_meeting_support(field_hard_fail_values, min_support=check_penalty_min_support),
        _counts_meeting_support(operator_hard_fail_values, min_support=check_penalty_min_support),
        _counts_meeting_support(field_blocking_warning_values, min_support=check_penalty_min_support),
        _counts_meeting_support(operator_blocking_warning_values, min_support=check_penalty_min_support),
        _counts_meeting_support(field_robustness_values, min_support=_ROBUSTNESS_CHECK_FIELD_MIN_SUPPORT),
    )


def _completed_result_priors(
    *,
    repository: SQLiteRepository,
    config: AppConfig,
    run_id: str,
    round_index: int,
) -> _WinnerPriorResult:
    filter_config = config.adaptive_generation.search_space_filter
    if not bool(filter_config.winner_prior_enabled):
        return _empty_winner_prior_result()
    cache_ttl_seconds = float(getattr(filter_config, "winner_prior_cache_ttl_seconds", 0.0))
    cache_key = _winner_prior_cache_key(
        repository=repository,
        filter_config=filter_config,
        run_id=run_id,
        round_index=round_index,
    )
    now = time.monotonic()
    if cache_ttl_seconds > 0.0:
        with _WINNER_PRIOR_CACHE_LOCK:
            cached = _WINNER_PRIOR_CACHE.get(cache_key)
            if cached is not None and now - cached[0] < cache_ttl_seconds:
                _log_winner_prior_stats(cached[1])
                return cached[1]

    result = _compute_completed_result_priors(
        repository=repository,
        config=config,
        run_id=run_id,
        round_index=round_index,
    )
    if cache_ttl_seconds > 0.0:
        with _WINNER_PRIOR_CACHE_LOCK:
            _WINNER_PRIOR_CACHE[cache_key] = (now, result)
    _log_winner_prior_stats(result)
    return result


def _compute_completed_result_priors(
    *,
    repository: SQLiteRepository,
    config: AppConfig,
    run_id: str,
    round_index: int,
) -> _WinnerPriorResult:
    filter_config = config.adaptive_generation.search_space_filter
    try:
        rows, round_window_available = _winner_prior_history_rows(
            repository=repository,
            run_id=run_id,
            round_index=round_index,
        )
    except Exception:
        return _empty_winner_prior_result()

    all_time_observation = _winner_prior_observation(rows, filter_config=filter_config)
    recent_rows = _winner_prior_recent_rows(
        rows,
        current_round=round_index,
        lookback_rounds=int(filter_config.winner_prior_lookback_rounds),
        round_window_available=round_window_available,
    )
    recent_observation = _winner_prior_observation(recent_rows, filter_config=filter_config)
    min_completed = int(filter_config.winner_prior_min_completed)
    if recent_observation.quality_count >= min_completed:
        return _winner_prior_result_from_observation(
            recent_observation,
            filter_config=filter_config,
            source="recent",
            dampen=1.0,
            recent_quality_count=recent_observation.quality_count,
            all_time_quality_count=all_time_observation.quality_count,
        )
    if all_time_observation.quality_count >= min_completed:
        return _winner_prior_result_from_observation(
            all_time_observation,
            filter_config=filter_config,
            source="all_time_dampened",
            dampen=float(filter_config.winner_prior_alltime_dampen),
            recent_quality_count=recent_observation.quality_count,
            all_time_quality_count=all_time_observation.quality_count,
        )
    return _winner_prior_result_from_observation(
        all_time_observation,
        filter_config=filter_config,
        source="insufficient",
        dampen=1.0,
        recent_quality_count=recent_observation.quality_count,
        all_time_quality_count=all_time_observation.quality_count,
        force_neutral=True,
    )


def _winner_prior_history_rows(
    *,
    repository: SQLiteRepository,
    run_id: str,
    round_index: int,
) -> tuple[list[Any], bool]:
    round_clause = "AND r.round_index < ?" if round_index > 0 else ""
    params: list[object] = [run_id, run_id]
    if round_index > 0:
        params.append(int(round_index))
    try:
        rows = repository.connection.execute(
            f"""
            WITH latest_runtime AS (
                SELECT *
                FROM service_runtime
                WHERE service_run_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
            )
            SELECT r.job_id, r.run_id, r.round_index, r.batch_id, r.candidate_id,
                   r.status, r.sharpe, r.fitness, r.rejection_reason,
                   r.hard_fail_checks_json, r.blocking_warning_checks_json,
                   s.error_message AS submission_error_message,
                   s.service_failure_reason AS submission_service_failure_reason,
                   sb.service_status_reason AS batch_service_status_reason,
                   COALESCE((
                       SELECT COUNT(*)
                       FROM brain_results peer
                       WHERE peer.run_id = r.run_id
                         AND peer.batch_id = r.batch_id
                         AND peer.status = 'timeout'
                         AND peer.job_id <> r.job_id
                   ), 0) AS batch_timeout_peer_count,
                   COALESCE((
                       SELECT COUNT(*)
                       FROM brain_results peer
                       WHERE peer.run_id = r.run_id
                         AND peer.batch_id = r.batch_id
                         AND peer.status = 'completed'
                   ), 0) AS batch_completed_count,
                   COALESCE((
                       SELECT COUNT(*)
                       FROM brain_results peer
                       WHERE peer.run_id = r.run_id
                         AND peer.batch_id = r.batch_id
                         AND peer.status = 'failed'
                   ), 0) AS batch_failed_count,
                   COALESCE((
                       SELECT COUNT(*)
                       FROM brain_results peer
                       WHERE peer.run_id = r.run_id
                         AND peer.batch_id = r.batch_id
                         AND peer.status = 'rejected'
                   ), 0) AS batch_rejected_count,
                   COALESCE((
                       SELECT COUNT(DISTINCT peer.batch_id)
                       FROM brain_results peer
                       WHERE peer.run_id = r.run_id
                         AND peer.candidate_id = r.candidate_id
                         AND peer.status = 'timeout'
                   ), 0) AS candidate_timeout_batch_count,
                   sr.status AS runtime_status,
                   sr.last_error AS runtime_last_error,
                   CASE
                       WHEN COALESCE(sr.cooldown_until, '') <> ''
                            AND (
                                sr.status IN ('cooldown', 'auth_throttled', 'auth_unavailable')
                                OR sr.cooldown_until > COALESCE(s.submitted_at, r.created_at)
                            )
                       THEN 1 ELSE 0
                   END AS auth_cooldown_active,
                   CASE
                       WHEN sr.status = 'waiting_persona_confirmation'
                            OR COALESCE(sr.persona_confirmation_nonce, '') <> ''
                       THEN 1 ELSE 0
                   END AS persona_confirmation_pending,
                   a.fields_used_json, a.operators_used_json, a.generation_metadata
            FROM brain_results r
            JOIN alphas a ON a.run_id = r.run_id AND a.alpha_id = r.candidate_id
            LEFT JOIN submissions s ON s.job_id = r.job_id
            LEFT JOIN submission_batches sb ON sb.batch_id = r.batch_id
            LEFT JOIN latest_runtime sr ON sr.service_run_id = r.run_id
            WHERE r.run_id = ?
              {round_clause}
            ORDER BY r.round_index DESC, r.created_at DESC
            """,
            tuple(params),
        ).fetchall()
        return list(rows), True
    except Exception:
        rows = repository.connection.execute(
            """
            WITH latest_runtime AS (
                SELECT *
                FROM service_runtime
                WHERE service_run_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
            )
            SELECT r.job_id, r.run_id, 0 AS round_index, r.batch_id, r.candidate_id,
                   r.status, r.sharpe, r.fitness, r.rejection_reason,
                   r.hard_fail_checks_json, r.blocking_warning_checks_json,
                   s.error_message AS submission_error_message,
                   s.service_failure_reason AS submission_service_failure_reason,
                   sb.service_status_reason AS batch_service_status_reason,
                   0 AS batch_timeout_peer_count,
                   0 AS batch_completed_count,
                   0 AS batch_failed_count,
                   0 AS batch_rejected_count,
                   0 AS candidate_timeout_batch_count,
                   sr.status AS runtime_status,
                   sr.last_error AS runtime_last_error,
                   0 AS auth_cooldown_active,
                   0 AS persona_confirmation_pending,
                   a.fields_used_json, a.operators_used_json, a.generation_metadata
            FROM brain_results r
            JOIN alphas a ON a.run_id = r.run_id AND a.alpha_id = r.candidate_id
            LEFT JOIN submissions s ON s.job_id = r.job_id
            LEFT JOIN submission_batches sb ON sb.batch_id = r.batch_id
            LEFT JOIN latest_runtime sr ON sr.service_run_id = r.run_id
            WHERE r.run_id = ?
            ORDER BY r.created_at DESC
            """,
            (run_id, run_id),
        ).fetchall()
        return list(rows), False


def _winner_prior_recent_rows(
    rows: list[Any],
    *,
    current_round: int,
    lookback_rounds: int,
    round_window_available: bool,
) -> list[Any]:
    if not round_window_available or current_round <= 0:
        return []
    min_round = max(0, int(current_round) - int(lookback_rounds))
    recent: list[Any] = []
    for row in rows:
        row_round = _to_int(_row_get(row, "round_index"))
        if row_round is not None and min_round <= row_round < int(current_round):
            recent.append(row)
    return recent


def _winner_prior_observation(rows: list[Any], *, filter_config: Any) -> _WinnerPriorObservation:
    field_labels: dict[str, Counter[str]] = defaultdict(Counter)
    operator_labels: dict[str, Counter[str]] = defaultdict(Counter)
    field_support: Counter[str] = Counter()
    operator_support: Counter[str] = Counter()
    timeout_cause_counts: Counter[str] = Counter()
    timeout_field_counts: Counter[str] = Counter()
    timeout_operator_counts: Counter[str] = Counter()
    quality_count = 0
    for row in rows:
        fields = _decode_text_list(_row_get(row, "fields_used_json"))
        operators = _decode_text_list(_row_get(row, "operators_used_json"))
        if not fields or not operators:
            metadata = _decode_json(_row_get(row, "generation_metadata"))
            fields = fields or [str(item) for item in metadata.get("fields_used", []) if str(item)]
            operators = operators or [str(item) for item in metadata.get("operators_used", []) if str(item)]
        if not fields and not operators:
            continue

        status = str(_row_get(row, "status", "") or "").strip().lower()
        if status == "timeout":
            timeout_cause = classify_timeout_cause(row)
            timeout_cause_counts[timeout_cause] += 1
            if timeout_cause == "operational":
                continue
            for field_name in dict.fromkeys(fields):
                timeout_field_counts[field_name] += 1
            for operator_name in dict.fromkeys(_policy_operator_names(operators)):
                timeout_operator_counts[operator_name] += 1
        else:
            rejection_reason = str(_row_get(row, "rejection_reason", "") or "")
            if _is_operational_rejection(rejection_reason):
                continue
            hard_fail_checks = parse_names_json(_row_get(row, "hard_fail_checks_json"))
            blocking_warning_checks = parse_names_json(_row_get(row, "blocking_warning_checks_json"))
            if has_structural_risk_blocker(hard_fail_checks, blocking_warning_checks):
                continue

        quality_count += 1
        label = "win" if is_winner_result(row, filter_config) else "loss"
        for field_name in dict.fromkeys(fields):
            field_support[field_name] += 1
            field_labels[field_name][label] += 1
        for operator_name in dict.fromkeys(_policy_operator_names(operators)):
            operator_support[operator_name] += 1
            operator_labels[operator_name][label] += 1
    return _WinnerPriorObservation(
        field_labels=dict(field_labels),
        operator_labels=dict(operator_labels),
        field_support=field_support,
        operator_support=operator_support,
        timeout_cause_counts=timeout_cause_counts,
        timeout_field_counts=timeout_field_counts,
        timeout_operator_counts=timeout_operator_counts,
        quality_count=quality_count,
    )


def _winner_prior_result_from_observation(
    observation: _WinnerPriorObservation,
    *,
    filter_config: Any,
    source: str,
    dampen: float,
    recent_quality_count: int,
    all_time_quality_count: int,
    force_neutral: bool = False,
) -> _WinnerPriorResult:
    if force_neutral:
        field_multipliers: dict[str, float] = {}
        operator_multipliers: dict[str, float] = {}
        promoted_fields: Counter[str] = Counter()
        demoted_fields: Counter[str] = Counter()
        promoted_operators: Counter[str] = Counter()
        demoted_operators: Counter[str] = Counter()
    else:
        field_multipliers, promoted_fields, demoted_fields = _winner_prior_multipliers(
            observation.field_labels,
            min_winners_for_boost=int(filter_config.winner_prior_min_winners_for_boost),
            min_losers_for_penalty=int(filter_config.winner_prior_min_losers_for_penalty),
            laplace_k=float(filter_config.winner_prior_laplace_k),
            multiplier_min=float(filter_config.winner_prior_multiplier_min),
            multiplier_max=float(filter_config.winner_prior_multiplier_max),
            dampen=float(dampen),
        )
        operator_multipliers, promoted_operators, demoted_operators = _winner_prior_multipliers(
            observation.operator_labels,
            min_winners_for_boost=int(filter_config.winner_prior_min_winners_for_boost),
            min_losers_for_penalty=int(filter_config.winner_prior_min_losers_for_penalty),
            laplace_k=float(filter_config.winner_prior_laplace_k),
            multiplier_min=float(filter_config.winner_prior_multiplier_min),
            multiplier_max=float(filter_config.winner_prior_multiplier_max),
            dampen=float(dampen),
        )
    boost_values = [value for value in field_multipliers.values() if value > 1.0]
    penalty_values = [value for value in field_multipliers.values() if value < 1.0]
    fields_with_prior = len(field_multipliers)
    fields_neutral = max(0, len(observation.field_support) - fields_with_prior)
    timeout_demoted_fields = Counter(
        {
            key: observation.timeout_field_counts[key]
            for key in demoted_fields
            if observation.timeout_field_counts.get(key, 0) > 0
        }
    )
    timeout_demoted_operators = Counter(
        {
            key: observation.timeout_operator_counts[key]
            for key in demoted_operators
            if observation.timeout_operator_counts.get(key, 0) > 0
        }
    )
    return _WinnerPriorResult(
        field_multipliers=field_multipliers,
        operator_multipliers=operator_multipliers,
        promoted_field_counts=promoted_fields,
        demoted_field_counts=demoted_fields,
        promoted_operator_counts=promoted_operators,
        demoted_operator_counts=demoted_operators,
        timeout_cause_counts=observation.timeout_cause_counts,
        timeout_demoted_field_counts=timeout_demoted_fields,
        timeout_demoted_operator_counts=timeout_demoted_operators,
        source=source,
        lookback_rounds=int(filter_config.winner_prior_lookback_rounds),
        recent_quality_count=int(recent_quality_count),
        all_time_quality_count=int(all_time_quality_count),
        fields_with_prior=fields_with_prior,
        fields_neutral=fields_neutral,
        avg_boost=sum(boost_values) / len(boost_values) if boost_values else 1.0,
        avg_penalty=sum(penalty_values) / len(penalty_values) if penalty_values else 1.0,
    )


def is_winner_result(result: Any, filter_config: Any) -> bool:
    normalized_status = str(_row_get(result, "status", "completed") or "completed").strip().lower()
    if normalized_status != "completed":
        return False
    if bool(_row_get(result, "is_rejected", False)):
        return False
    rejection_reason = str(_row_get(result, "rejection_reason", "") or "")
    if rejection_reason:
        return False
    hard_fail_checks = parse_names_json(_row_get(result, "hard_fail_checks_json"))
    blocking_warning_checks = parse_names_json(_row_get(result, "blocking_warning_checks_json"))
    if has_structural_risk_blocker(hard_fail_checks, blocking_warning_checks):
        return False
    sharpe = _to_float(_row_get(result, "sharpe"))
    if sharpe is None or sharpe < float(filter_config.winner_prior_min_sharpe):
        return False
    fitness = _to_float(_row_get(result, "fitness"))
    if fitness is not None and fitness < float(filter_config.winner_prior_min_fitness):
        return False
    return True


def _is_operational_rejection(rejection_reason: str) -> bool:
    normalized = str(rejection_reason or "").strip().lower()
    return bool(normalized) and any(marker in normalized for marker in _OPERATIONAL_REJECTION_MARKERS)


def _winner_prior_multipliers(
    labels: dict[str, Counter[str]],
    *,
    min_winners_for_boost: int,
    min_losers_for_penalty: int,
    laplace_k: float,
    multiplier_min: float,
    multiplier_max: float,
    dampen: float,
) -> tuple[dict[str, float], Counter[str], Counter[str]]:
    multipliers: dict[str, float] = {}
    promoted: Counter[str] = Counter()
    demoted: Counter[str] = Counter()
    for key, counts in labels.items():
        win_count = int(counts.get("win", 0))
        loss_count = int(counts.get("loss", 0))
        if win_count < int(min_winners_for_boost) and loss_count < int(min_losers_for_penalty):
            continue
        multiplier = _prior_multiplier_from_counts(
            win_count=win_count,
            loss_count=loss_count,
            laplace_k=laplace_k,
            multiplier_min=multiplier_min,
            multiplier_max=multiplier_max,
            dampen=dampen,
        )
        if multiplier > 1.0 and win_count >= int(min_winners_for_boost):
            multipliers[key] = round(multiplier, 6)
            promoted[key] = win_count
        elif multiplier < 1.0 and loss_count >= int(min_losers_for_penalty):
            multipliers[key] = round(multiplier, 6)
            demoted[key] = loss_count
    return multipliers, promoted, demoted


def _prior_multiplier_from_counts(
    *,
    win_count: int,
    loss_count: int,
    laplace_k: float,
    multiplier_min: float,
    multiplier_max: float,
    dampen: float = 1.0,
) -> float:
    total = max(0, int(win_count)) + max(0, int(loss_count))
    if total <= 0:
        return 1.0
    k = max(1e-9, float(laplace_k))
    win_rate = (max(0, int(win_count)) + k) / (float(total) + 2.0 * k)
    multiplier = 1.0 + (win_rate - 0.5) * 2.0 * (float(multiplier_max) - 1.0)
    multiplier = max(float(multiplier_min), min(float(multiplier_max), multiplier))
    dampened = 1.0 + (multiplier - 1.0) * max(0.0, min(1.0, float(dampen)))
    return max(float(multiplier_min), min(float(multiplier_max), dampened))


def _winner_prior_cache_key(
    *,
    repository: SQLiteRepository,
    filter_config: Any,
    run_id: str,
    round_index: int,
) -> tuple[Any, ...]:
    return (
        id(repository.connection),
        str(run_id),
        int(round_index),
        int(filter_config.winner_prior_lookback_rounds),
        int(filter_config.winner_prior_min_completed),
        int(filter_config.winner_prior_min_winners_for_boost),
        int(filter_config.winner_prior_min_losers_for_penalty),
        round(float(filter_config.winner_prior_laplace_k), 8),
        round(float(filter_config.winner_prior_multiplier_min), 8),
        round(float(filter_config.winner_prior_multiplier_max), 8),
        round(float(filter_config.winner_prior_alltime_dampen), 8),
        round(float(filter_config.winner_prior_min_sharpe), 8),
        round(float(filter_config.winner_prior_min_fitness), 8),
    )


def _log_winner_prior_stats(result: _WinnerPriorResult) -> None:
    logger.info(
        "Winner prior stats: fields_with_prior=%d fields_neutral=%d "
        "avg_boost=%.2f avg_penalty=%.2f "
        "based_on_recent_rounds=%d completed_in_window=%d",
        int(result.fields_with_prior),
        int(result.fields_neutral),
        float(result.avg_boost),
        float(result.avg_penalty),
        int(result.lookback_rounds),
        int(result.recent_quality_count),
    )


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except (IndexError, KeyError, TypeError):
        return getattr(row, key, default)


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _top_prior_items(
    multipliers: dict[str, float],
    counts: Counter[str],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    items = [
        {
            "name": key,
            "support": int(count),
            "multiplier": round(float(multipliers.get(key, 1.0)), 6),
        }
        for key, count in counts.items()
        if float(multipliers.get(key, 1.0)) > 1.0
    ]
    return sorted(items, key=lambda item: (-float(item["multiplier"]), -int(item["support"]), str(item["name"])))[: int(limit)]


def _top_multiplier_items(multipliers: dict[str, float], *, limit: int = 10) -> list[dict[str, Any]]:
    items = [
        {"name": key, "multiplier": round(float(value), 6)}
        for key, value in multipliers.items()
        if abs(float(value) - 1.0) > 1e-9
    ]
    return sorted(items, key=lambda item: (float(item["multiplier"]), str(item["name"])))[: int(limit)]


def _underperforming_keys(
    values: dict[str, list[tuple[float, float]]],
    *,
    min_support: int,
    sharpe_floor: float,
    fitness_floor: float,
) -> Counter[str]:
    penalties: Counter[str] = Counter()
    for key, samples in values.items():
        if len(samples) < int(min_support):
            continue
        avg_sharpe = sum(item[0] for item in samples) / len(samples)
        avg_fitness = sum(item[1] for item in samples) / len(samples)
        if avg_sharpe < float(sharpe_floor) or avg_fitness < float(fitness_floor):
            penalties[key] = len(samples)
    return penalties


def _counts_meeting_support(counts: Counter[str], *, min_support: int) -> Counter[str]:
    return Counter({key: count for key, count in counts.items() if int(count) >= int(min_support)})


def _penalized_template_weights(
    template_weights: dict[str, float],
    operator_multipliers: dict[str, float],
) -> dict[str, float]:
    updated = dict(template_weights)
    for template_name, operators in _TEMPLATE_OPERATORS.items():
        multiplier = min((operator_multipliers.get(operator, 1.0) for operator in operators), default=1.0)
        if multiplier >= 0.999999:
            continue
        base_weight = float(updated.get(template_name, 1.0))
        updated[template_name] = max(1e-6, base_weight * float(multiplier))
    return updated


def _decode_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        decoded = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _decode_text_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if str(item)]
    try:
        decoded = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if isinstance(decoded, list):
        return [str(item) for item in decoded if str(item)]
    return []


def _policy_operator_names(operators: list[str] | tuple[str, ...]) -> list[str]:
    return [
        normalized
        for item in operators
        if (normalized := str(item).strip())
        and not normalized.startswith(("binary:", "unary:"))
    ]


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
