from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from data.field_registry import FieldRegistry, FieldSpec
from features.registry import OperatorRegistry


@dataclass(frozen=True, slots=True)
class FieldSlot:
    name: str
    operator_type: str = "matrix"
    preferred_categories: tuple[str, ...] = ()
    distinct_from: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    name: str
    field_slots: tuple[FieldSlot, ...]
    build_expression: Callable[[dict[str, str], dict[str, Any]], str]
    parameter_builder: Callable[[list[int], random.Random], dict[str, Any]]
    base_weight: float = 1.0
    max_depth: int = 4
    turnover_bias: float = 0.0


@dataclass(frozen=True, slots=True)
class TemplateInstance:
    template_name: str
    expression: str
    fields_used: tuple[str, ...]
    parameters: dict[str, Any] = field(default_factory=dict)


def build_default_template_library(
    lookbacks: list[int],
    template_weights: dict[str, float] | None = None,
) -> list[TemplateSpec]:
    weights = template_weights or {}

    def unary_window_parameters(choices: list[int], randomizer: random.Random) -> dict[str, Any]:
        return {"window": randomizer.choice(choices)}

    def fast_slow_parameters(choices: list[int], randomizer: random.Random) -> dict[str, Any]:
        ordered = sorted(set(choices))
        fast = randomizer.choice(ordered[:-1] or ordered)
        slower_pool = [value for value in ordered if value > fast] or ordered[-1:]
        return {"fast_window": fast, "slow_window": randomizer.choice(slower_pool)}

    def pair_window_parameters(choices: list[int], randomizer: random.Random) -> dict[str, Any]:
        return {"window": randomizer.choice(choices), "pair_operator": randomizer.choice(["ts_corr", "ts_covariance"])}

    def no_params(_: list[int], __: random.Random) -> dict[str, Any]:
        return {}

    def weight_for(name: str, fallback: float) -> float:
        return float(weights.get(name, fallback))

    return [
        TemplateSpec(
            name="momentum",
            field_slots=(FieldSlot("x", preferred_categories=("price", "volume", "fundamental", "analyst")),),
            build_expression=lambda fields, params: f"rank(ts_mean({fields['x']}, {params['window']}))",
            parameter_builder=unary_window_parameters,
            base_weight=weight_for("momentum", 1.00),
            turnover_bias=-0.10,
        ),
        TemplateSpec(
            name="mean_reversion",
            field_slots=(FieldSlot("x", preferred_categories=("price", "volume", "fundamental", "analyst")),),
            build_expression=lambda fields, params: f"rank(({fields['x']}-ts_mean({fields['x']}, {params['window']})))",
            parameter_builder=unary_window_parameters,
            base_weight=weight_for("mean_reversion", 0.95),
            turnover_bias=0.10,
        ),
        TemplateSpec(
            name="volatility",
            field_slots=(FieldSlot("x", preferred_categories=("price", "volume", "risk", "model")),),
            build_expression=lambda fields, params: f"rank(ts_std_dev({fields['x']}, {params['window']}))",
            parameter_builder=unary_window_parameters,
            base_weight=weight_for("volatility", 0.85),
            turnover_bias=-0.05,
        ),
        TemplateSpec(
            name="ratio",
            field_slots=(FieldSlot("x", preferred_categories=("price", "fundamental", "analyst", "model")),),
            build_expression=lambda fields, params: f"rank(({fields['x']}/ts_mean({fields['x']}, {params['window']})))",
            parameter_builder=unary_window_parameters,
            base_weight=weight_for("ratio", 0.90),
            turnover_bias=-0.10,
        ),
        TemplateSpec(
            name="delta",
            field_slots=(FieldSlot("x", preferred_categories=("price", "volume", "fundamental", "analyst")),),
            build_expression=lambda fields, params: f"rank(ts_delta({fields['x']}, {params['window']}))",
            parameter_builder=unary_window_parameters,
            base_weight=weight_for("delta", 1.00),
            turnover_bias=0.20,
        ),
        TemplateSpec(
            name="cross_field_spread",
            field_slots=(
                FieldSlot("x", preferred_categories=("price", "fundamental", "analyst", "model")),
                FieldSlot("y", preferred_categories=("price", "fundamental", "analyst", "model"), distinct_from=("x",)),
            ),
            build_expression=lambda fields, params: f"rank(({fields['x']}-{fields['y']}))",
            parameter_builder=no_params,
            base_weight=weight_for("cross_field_spread", 0.80),
            turnover_bias=0.00,
        ),
        TemplateSpec(
            name="correlation_pair",
            field_slots=(
                FieldSlot("x", preferred_categories=("price", "volume", "fundamental", "analyst")),
                FieldSlot("y", preferred_categories=("price", "volume", "fundamental", "analyst"), distinct_from=("x",)),
            ),
            build_expression=lambda fields, params: f"rank({params['pair_operator']}({fields['x']}, {fields['y']}, {params['window']}))",
            parameter_builder=pair_window_parameters,
            base_weight=weight_for("correlation_pair", 0.75),
            turnover_bias=-0.15,
        ),
        TemplateSpec(
            name="group_relative",
            field_slots=(
                FieldSlot("x", preferred_categories=("price", "fundamental", "analyst", "model")),
                FieldSlot("group", operator_type="group"),
            ),
            build_expression=lambda fields, params: f"rank(group_neutralize({fields['x']}, {fields['group']}))",
            parameter_builder=no_params,
            base_weight=weight_for("group_relative", 0.80),
            turnover_bias=-0.05,
        ),
        TemplateSpec(
            name="smoothed_momentum",
            field_slots=(FieldSlot("x", preferred_categories=("price", "volume", "fundamental", "analyst")),),
            build_expression=lambda fields, params: (
                f"rank(ts_decay_linear(ts_delta({fields['x']}, {params['fast_window']}), {params['slow_window']}))"
            ),
            parameter_builder=fast_slow_parameters,
            base_weight=weight_for("smoothed_momentum", 0.95),
            turnover_bias=-0.25,
        ),
        TemplateSpec(
            name="fundamental_vs_price_confirmation",
            field_slots=(
                FieldSlot("x", preferred_categories=("fundamental", "analyst", "model")),
                FieldSlot("y", preferred_categories=("price", "volume"), distinct_from=("x",)),
            ),
            build_expression=lambda fields, params: (
                f"rank((ts_mean({fields['x']}, {params['slow_window']})-ts_delta({fields['y']}, {params['fast_window']})))"
            ),
            parameter_builder=fast_slow_parameters,
            base_weight=weight_for("fundamental_vs_price_confirmation", 0.85),
            turnover_bias=-0.10,
        ),
    ]


def instantiate_template(
    template: TemplateSpec,
    *,
    field_registry: FieldRegistry,
    allowed_fields: set[str],
    lookbacks: list[int],
    randomizer: random.Random,
    template_memory: dict[str, float] | None = None,
    field_memory: dict[str, float] | None = None,
) -> TemplateInstance | None:
    selected_fields: dict[str, str] = {}
    for slot in template.field_slots:
        candidate = _select_field_for_slot(
            slot=slot,
            template=template,
            field_registry=field_registry,
            allowed_fields=allowed_fields,
            randomizer=randomizer,
            selected_fields=selected_fields,
            field_memory=field_memory or {},
        )
        if candidate is None:
            return None
        selected_fields[slot.name] = candidate.name

    parameters = template.parameter_builder(lookbacks, randomizer)
    expression = template.build_expression(selected_fields, parameters)
    return TemplateInstance(
        template_name=template.name,
        expression=expression,
        fields_used=tuple(selected_fields.values()),
        parameters={
            **parameters,
            "template_weight": float(template_memory.get(template.name, 0.0)) if template_memory else 0.0,
        },
    )


def select_template_specs(
    library: list[TemplateSpec],
    *,
    count: int,
    randomizer: random.Random,
    template_memory: dict[str, float] | None = None,
    max_turnover_bias: float = 0.35,
) -> list[TemplateSpec]:
    if count <= 0 or not library:
        return []
    memory = template_memory or {}
    weights = [
        max(
            1e-6,
            spec.base_weight
            * (1.0 + memory.get(spec.name, 0.0))
            * (1.0 - min(abs(spec.turnover_bias), max_turnover_bias)),
        )
        for spec in library
    ]
    return randomizer.choices(library, weights=weights, k=count)


def generate_template_instances(
    *,
    field_registry: FieldRegistry,
    allowed_fields: set[str],
    lookbacks: list[int],
    template_weights: dict[str, float],
    template_pool_size: int,
    max_turnover_bias: float,
    randomizer: random.Random,
    registry: OperatorRegistry,
    field_memory: dict[str, float] | None = None,
    template_memory: dict[str, float] | None = None,
    allowed_templates: set[str] | None = None,
) -> list[TemplateInstance]:
    library = [
        spec
        for spec in build_default_template_library(lookbacks, template_weights=template_weights)
        if allowed_templates is None or spec.name in allowed_templates
    ]
    specs = select_template_specs(
        library,
        count=max(template_pool_size, len(library)),
        randomizer=randomizer,
        template_memory=template_memory,
        max_turnover_bias=max_turnover_bias,
    )
    instances: list[TemplateInstance] = []
    seen: set[str] = set()
    for spec in specs:
        instance = instantiate_template(
            spec,
            field_registry=field_registry,
            allowed_fields=allowed_fields,
            lookbacks=lookbacks,
            randomizer=randomizer,
            template_memory=template_memory,
            field_memory=field_memory,
        )
        if instance is None or instance.expression in seen:
            continue
        seen.add(instance.expression)
        instances.append(instance)
    return instances


def generate_template_expressions(
    *,
    field_registry: FieldRegistry,
    allowed_fields: set[str],
    lookbacks: list[int],
    template_weights: dict[str, float],
    template_pool_size: int,
    max_turnover_bias: float,
    seed: int,
    registry: OperatorRegistry,
    field_memory: dict[str, float] | None = None,
    template_memory: dict[str, float] | None = None,
    allowed_templates: set[str] | None = None,
) -> list[TemplateInstance]:
    randomizer = random.Random(seed)
    return generate_template_instances(
        field_registry=field_registry,
        allowed_fields=allowed_fields,
        lookbacks=lookbacks,
        template_weights=template_weights,
        template_pool_size=template_pool_size,
        max_turnover_bias=max_turnover_bias,
        randomizer=randomizer,
        registry=registry,
        field_memory=field_memory,
        template_memory=template_memory,
        allowed_templates=allowed_templates,
    )


def _select_field_for_slot(
    *,
    slot: FieldSlot,
    template: TemplateSpec,
    field_registry: FieldRegistry,
    allowed_fields: set[str],
    randomizer: random.Random,
    selected_fields: dict[str, str],
    field_memory: dict[str, float],
) -> FieldSpec | None:
    pool = (
        field_registry.runtime_group_fields(allowed=allowed_fields)
        if slot.operator_type == "group"
        else field_registry.runtime_numeric_fields(allowed=allowed_fields)
    )
    candidates: list[FieldSpec] = []
    weights: list[float] = []
    blocked = {selected_fields[name] for name in slot.distinct_from if name in selected_fields}
    for spec in pool:
        if spec.name in blocked:
            continue
        weight = max(spec.field_score, 1e-6)
        if slot.preferred_categories:
            if spec.category in slot.preferred_categories or any(spec.category.startswith(prefix) for prefix in slot.preferred_categories):
                weight *= 1.25
            else:
                weight *= 0.75
        weight *= 1.0 + field_memory.get(spec.name, 0.0)
        weight *= 1.0 - min(abs(template.turnover_bias), 0.35)
        candidates.append(spec)
        weights.append(max(weight, 1e-6))
    if not candidates:
        return None
    return randomizer.choices(candidates, weights=weights, k=1)[0]
