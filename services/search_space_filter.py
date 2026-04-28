from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any

from alpha.ast_nodes import BinaryOpNode, FunctionCallNode, IdentifierNode, UnaryOpNode
from alpha.parser import parse_expression
from core.config import AppConfig, GenerationConfig
from data.field_registry import FieldRegistry
from storage.repository import SQLiteRepository


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


@dataclass(frozen=True, slots=True)
class _WinnerPriorResult:
    field_multipliers: dict[str, float]
    operator_multipliers: dict[str, float]
    promoted_field_counts: Counter[str]
    demoted_field_counts: Counter[str]
    promoted_operator_counts: Counter[str]
    demoted_operator_counts: Counter[str]


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
                "search_space_filter_promoted_field_count": 0,
                "search_space_filter_demoted_field_count": 0,
                "search_space_filter_promoted_operator_count": 0,
                "search_space_filter_demoted_operator_count": 0,
                "search_space_filter_top_promoted_fields": [],
                "search_space_filter_top_promoted_operators": [],
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
    field_result_penalties, operator_result_penalties = _completed_result_penalties(
        repository=repository,
        run_id=run_id,
        round_index=round_index,
        lookback_rounds=int(filter_config.completed_lookback_rounds),
        min_support=int(filter_config.min_completed_support),
        sharpe_floor=float(filter_config.sharpe_floor),
        fitness_floor=float(filter_config.fitness_floor),
    )
    winner_prior = _completed_result_priors(
        repository=repository,
        config=config,
        run_id=run_id,
        round_index=round_index,
    )
    updated_fields = {}
    soft_penalized_fields: set[str] = set()
    field_multipliers: dict[str, float] = {}
    for name, spec in field_registry.fields.items():
        if name in blocked_fields:
            continue
        multiplier = _profile_multiplier(spec, config=config)
        if validation_counts.get(name, 0) >= int(filter_config.validation_field_min_count):
            multiplier *= float(filter_config.validation_field_multiplier)
        if name in field_result_penalties:
            multiplier *= float(filter_config.field_result_multiplier)
        if name in winner_prior.field_multipliers:
            multiplier *= float(winner_prior.field_multipliers[name])
        multiplier = max(1e-6, float(multiplier))
        if multiplier < 0.999999:
            soft_penalized_fields.add(name)
        if abs(multiplier - 1.0) > 1e-9:
            field_multipliers[name] = round(multiplier, 6)
        updated_fields[name] = replace(spec, field_score=max(1e-6, float(spec.field_score) * multiplier))

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
    for operator, multiplier in winner_prior.operator_multipliers.items():
        operator_multipliers[operator] = round(float(operator_multipliers.get(operator, 1.0)) * float(multiplier), 6)
    lane_field_pool_counts = {
        lane: len(pool) for lane, pool in sorted(lane_field_pools.items())
    }
    operator_penalty_counts = dict(sorted(operator_result_penalties.items()))
    field_result_penalty_counts = dict(sorted(field_result_penalties.items()))
    validation_field_penalty_counts = dict(sorted(validation_counts.items()))
    top_promoted_fields = _top_prior_items(winner_prior.field_multipliers, winner_prior.promoted_field_counts)
    top_promoted_operators = _top_prior_items(winner_prior.operator_multipliers, winner_prior.promoted_operator_counts)
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
            "search_space_filter_validation_field_penalty_counts": validation_field_penalty_counts,
            "search_space_filter_promoted_field_count": len(winner_prior.promoted_field_counts),
            "search_space_filter_demoted_field_count": len(winner_prior.demoted_field_counts),
            "search_space_filter_promoted_operator_count": len(winner_prior.promoted_operator_counts),
            "search_space_filter_demoted_operator_count": len(winner_prior.demoted_operator_counts),
            "search_space_filter_top_promoted_fields": top_promoted_fields,
            "search_space_filter_top_promoted_operators": top_promoted_operators,
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
        pools[lane] = {spec.name for spec in numeric[:target_count]}
    return pools, before_counts


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
    sharpe_floor: float,
    fitness_floor: float,
) -> tuple[Counter[str], Counter[str]]:
    try:
        if round_index > 0:
            rows = repository.connection.execute(
                """
                SELECT r.sharpe, r.fitness, a.fields_used_json, a.operators_used_json, a.generation_metadata
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
                SELECT r.sharpe, r.fitness, a.fields_used_json, a.operators_used_json, a.generation_metadata
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
        return Counter(), Counter()
    field_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    operator_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
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
    return (
        _underperforming_keys(field_values, min_support=min_support, sharpe_floor=sharpe_floor, fitness_floor=fitness_floor),
        _underperforming_keys(operator_values, min_support=min_support, sharpe_floor=sharpe_floor, fitness_floor=fitness_floor),
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
        return _WinnerPriorResult({}, {}, Counter(), Counter(), Counter(), Counter())
    try:
        if round_index > 0:
            rows = repository.connection.execute(
                """
                SELECT r.status, r.sharpe, r.fitness, r.rejection_reason,
                       a.fields_used_json, a.operators_used_json, a.generation_metadata
                FROM brain_results r
                JOIN alphas a ON a.run_id = r.run_id AND a.alpha_id = r.candidate_id
                WHERE r.run_id = ?
                  AND r.round_index < ?
                ORDER BY r.round_index DESC, r.created_at DESC
                LIMIT ?
                """,
                (
                    run_id,
                    int(round_index),
                    max(50, int(filter_config.winner_prior_lookback_rounds) * 20),
                ),
            ).fetchall()
        else:
            rows = repository.connection.execute(
                """
                SELECT r.status, r.sharpe, r.fitness, r.rejection_reason,
                       a.fields_used_json, a.operators_used_json, a.generation_metadata
                FROM brain_results r
                JOIN alphas a ON a.run_id = r.run_id AND a.alpha_id = r.candidate_id
                WHERE r.run_id = ?
                ORDER BY r.round_index DESC, r.created_at DESC
                LIMIT ?
                """,
                (run_id, max(50, int(filter_config.winner_prior_lookback_rounds) * 20)),
            ).fetchall()
    except Exception:
        return _WinnerPriorResult({}, {}, Counter(), Counter(), Counter(), Counter())

    field_labels: dict[str, Counter[str]] = defaultdict(Counter)
    operator_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        label = _winner_prior_label(
            status=str(row["status"] or ""),
            sharpe=_to_float(row["sharpe"]),
            fitness=_to_float(row["fitness"]),
            rejection_reason=str(row["rejection_reason"] or ""),
            config=config,
        )
        if not label:
            continue
        fields = _decode_text_list(row["fields_used_json"])
        operators = _decode_text_list(row["operators_used_json"])
        if not fields or not operators:
            metadata = _decode_json(row["generation_metadata"])
            fields = fields or [str(item) for item in metadata.get("fields_used", []) if str(item)]
            operators = operators or [str(item) for item in metadata.get("operators_used", []) if str(item)]
        for field_name in dict.fromkeys(fields):
            field_labels[field_name][label] += 1
        for operator_name in dict.fromkeys(_policy_operator_names(operators)):
            operator_labels[operator_name][label] += 1

    field_multipliers, promoted_fields, demoted_fields = _winner_prior_multipliers(
        field_labels,
        min_support=int(filter_config.winner_prior_min_support),
        winner_multiplier=float(filter_config.winner_field_multiplier),
        strong_winner_multiplier=float(filter_config.strong_winner_field_multiplier),
        weak_multiplier=float(filter_config.weak_field_multiplier),
    )
    operator_multipliers, promoted_operators, demoted_operators = _winner_prior_multipliers(
        operator_labels,
        min_support=int(filter_config.winner_prior_min_support),
        winner_multiplier=float(filter_config.winner_operator_multiplier),
        strong_winner_multiplier=float(filter_config.strong_winner_operator_multiplier),
        weak_multiplier=float(filter_config.weak_operator_multiplier),
    )
    return _WinnerPriorResult(
        field_multipliers=field_multipliers,
        operator_multipliers=operator_multipliers,
        promoted_field_counts=promoted_fields,
        demoted_field_counts=demoted_fields,
        promoted_operator_counts=promoted_operators,
        demoted_operator_counts=demoted_operators,
    )


def _winner_prior_label(
    *,
    status: str,
    sharpe: float | None,
    fitness: float | None,
    rejection_reason: str,
    config: AppConfig,
) -> str:
    normalized_status = str(status or "").strip().lower()
    normalized_rejection = str(rejection_reason or "").strip().lower()
    if normalized_status != "completed":
        return ""
    if _is_operational_rejection(normalized_rejection):
        return ""
    if normalized_rejection:
        return "weak"
    if sharpe is None or fitness is None:
        return ""
    filter_config = config.adaptive_generation.search_space_filter
    if (
        sharpe >= float(filter_config.winner_prior_strong_sharpe_floor)
        and fitness >= float(filter_config.winner_prior_strong_fitness_floor)
    ):
        return "strong"
    if (
        sharpe >= float(filter_config.winner_prior_sharpe_floor)
        and fitness >= float(filter_config.winner_prior_fitness_floor)
    ):
        return "pass"
    return "weak"


def _is_operational_rejection(rejection_reason: str) -> bool:
    normalized = str(rejection_reason or "").strip().lower()
    return bool(normalized) and any(marker in normalized for marker in _OPERATIONAL_REJECTION_MARKERS)


def _winner_prior_multipliers(
    labels: dict[str, Counter[str]],
    *,
    min_support: int,
    winner_multiplier: float,
    strong_winner_multiplier: float,
    weak_multiplier: float,
) -> tuple[dict[str, float], Counter[str], Counter[str]]:
    multipliers: dict[str, float] = {}
    promoted: Counter[str] = Counter()
    demoted: Counter[str] = Counter()
    for key, counts in labels.items():
        strong_count = int(counts.get("strong", 0))
        pass_count = int(counts.get("pass", 0)) + strong_count
        weak_count = int(counts.get("weak", 0))
        if strong_count >= int(min_support):
            multipliers[key] = float(strong_winner_multiplier)
            promoted[key] = strong_count
        elif pass_count >= int(min_support):
            multipliers[key] = float(winner_multiplier)
            promoted[key] = pass_count
        elif weak_count >= int(min_support) and pass_count <= 0:
            multipliers[key] = float(weak_multiplier)
            demoted[key] = weak_count
    return multipliers, promoted, demoted


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
