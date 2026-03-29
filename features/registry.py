from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from features import operators


OperatorType = str


BRAIN_OPERATOR_ALIASES: dict[str, str] = {
    "delay": "ts_delay",
    "delta": "ts_delta",
    "rolling_mean": "ts_mean",
    "rolling_std": "ts_std_dev",
    "rolling_min": "ts_min",
    "rolling_max": "ts_max",
    "correlation": "ts_corr",
    "covariance": "ts_covariance",
    "decay_linear": "ts_decay_linear",
    "ts_std": "ts_std_dev",
}

BRAIN_DEFAULT_OPERATORS: tuple[str, ...] = (
    "ts_delay",
    "ts_delta",
    "ts_mean",
    "ts_std_dev",
    "ts_min",
    "ts_max",
    "rank",
    "zscore",
    "ts_corr",
    "ts_covariance",
    "ts_decay_linear",
    "ts_rank",
    "ts_sum",
    "sign",
    "abs",
    "log",
    "group_rank",
    "group_zscore",
    "group_neutralize",
)

WINDOWED_OPERATORS = {
    "delay",
    "delta",
    "returns",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "correlation",
    "covariance",
    "decay_linear",
    "ts_delay",
    "ts_delta",
    "ts_mean",
    "ts_std",
    "ts_std_dev",
    "ts_min",
    "ts_max",
    "ts_corr",
    "ts_covariance",
    "ts_decay_linear",
    "ts_rank",
    "ts_sum",
}

GROUP_OPERATORS = {"group_rank", "group_zscore", "group_neutralize"}
NORMALIZATION_OPERATORS = {"rank", "zscore", "sign", "abs"}
IDEMPOTENT_WRAPPERS = {"rank", "zscore", "sign", "abs"}


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    function: Callable
    min_args: int
    max_args: int
    description: str
    input_type_signatures: tuple[tuple[OperatorType, ...], ...] = ()
    output_type: OperatorType = "matrix"
    family: str = "other"
    parameter_requirements: dict[str, str] = field(default_factory=dict)
    turnover_hint: float = 0.0

    def supports_arg_count(self, count: int) -> bool:
        return self.min_args <= count <= self.max_args

    def compatible_signatures(self, arg_types: tuple[OperatorType, ...]) -> list[tuple[OperatorType, ...]]:
        if not self.input_type_signatures:
            return []
        return [signature for signature in self.input_type_signatures if _signature_matches(signature, arg_types)]

    def resolve_output_type(self, arg_types: tuple[OperatorType, ...]) -> OperatorType:
        if self.output_type == "same_as_first":
            return arg_types[0]
        return self.output_type


class OperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[str, OperatorSpec] = {}

    def register(
        self,
        name: str,
        function: Callable,
        min_args: int,
        max_args: int | None = None,
        description: str = "",
        input_type_signatures: tuple[tuple[OperatorType, ...], ...] = (),
        output_type: OperatorType = "matrix",
        family: str = "other",
        parameter_requirements: dict[str, str] | None = None,
        turnover_hint: float = 0.0,
    ) -> None:
        self._operators[name] = OperatorSpec(
            name=name,
            function=function,
            min_args=min_args,
            max_args=max_args if max_args is not None else min_args,
            description=description,
            input_type_signatures=input_type_signatures,
            output_type=output_type,
            family=family,
            parameter_requirements=parameter_requirements or {},
            turnover_hint=turnover_hint,
        )

    def register_alias(self, alias: str, canonical: str) -> None:
        spec = self._operators[canonical]
        self.register(
            alias,
            spec.function,
            spec.min_args,
            max_args=spec.max_args,
            description=f"{spec.description} Alias of '{canonical}'.",
            input_type_signatures=spec.input_type_signatures,
            output_type=spec.output_type,
            family=spec.family,
            parameter_requirements=spec.parameter_requirements,
            turnover_hint=spec.turnover_hint,
        )

    def get(self, name: str) -> OperatorSpec:
        return self._operators[name]

    def contains(self, name: str) -> bool:
        return name in self._operators

    def names(self) -> set[str]:
        return set(self._operators)

    def items(self):
        return self._operators.items()

    def family_for(self, name: str) -> str:
        return self._operators[name].family if name in self._operators else "other"


def build_default_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    _register_brain_canonical_specs(registry)
    _register_legacy_aliases(registry)
    _register_local_only_specs(registry)
    return registry


def build_registry(allowed_operators: list[str] | None = None) -> OperatorRegistry:
    default_registry = build_default_registry()
    if not allowed_operators:
        return default_registry

    configured = OperatorRegistry()
    for requested_name in allowed_operators:
        name = BRAIN_OPERATOR_ALIASES.get(requested_name, requested_name)
        spec = default_registry.get(name)
        configured.register(
            name=spec.name,
            function=spec.function,
            min_args=spec.min_args,
            max_args=spec.max_args,
            description=spec.description,
            input_type_signatures=spec.input_type_signatures,
            output_type=spec.output_type,
            family=spec.family,
            parameter_requirements=spec.parameter_requirements,
            turnover_hint=spec.turnover_hint,
        )
        if requested_name != name and default_registry.contains(requested_name):
            alias_spec = default_registry.get(requested_name)
            configured.register(
                name=alias_spec.name,
                function=alias_spec.function,
                min_args=alias_spec.min_args,
                max_args=alias_spec.max_args,
                description=alias_spec.description,
                input_type_signatures=alias_spec.input_type_signatures,
                output_type=alias_spec.output_type,
                family=alias_spec.family,
                parameter_requirements=alias_spec.parameter_requirements,
                turnover_hint=alias_spec.turnover_hint,
            )
    return configured


def _register_brain_canonical_specs(registry: OperatorRegistry) -> None:
    registry.register(
        "ts_delay",
        operators.delay,
        2,
        description="Returns the value of x from d days ago.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="same_as_first",
        family="time_series",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=0.30,
    )
    registry.register(
        "ts_delta",
        operators.delta,
        2,
        description="Difference between the current value and its delayed version over d days.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="same_as_first",
        family="time_series",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=0.90,
    )
    registry.register(
        "ts_mean",
        operators.ts_mean,
        2,
        description="Rolling mean over the last d days.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="smoothing",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.30,
    )
    registry.register(
        "ts_std_dev",
        operators.ts_std,
        2,
        description="Rolling standard deviation over the last d days.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="volatility",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.20,
    )
    registry.register(
        "ts_min",
        operators.rolling_min,
        2,
        description="Rolling minimum over the last d days.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="time_series",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.15,
    )
    registry.register(
        "ts_max",
        operators.rolling_max,
        2,
        description="Rolling maximum over the last d days.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="time_series",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.15,
    )
    registry.register(
        "rank",
        operators.rank,
        1,
        description="Cross-sectional rank by timestamp.",
        input_type_signatures=(("matrix",),),
        output_type="matrix",
        family="normalization",
        turnover_hint=0.10,
    )
    registry.register(
        "zscore",
        operators.zscore,
        1,
        description="Cross-sectional z-score by timestamp.",
        input_type_signatures=(("matrix",),),
        output_type="matrix",
        family="normalization",
        turnover_hint=0.08,
    )
    registry.register(
        "ts_corr",
        operators.correlation,
        3,
        description="Rolling correlation over the last d days.",
        input_type_signatures=(("matrix", "matrix", "scalar"),),
        output_type="matrix",
        family="cross_field",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.10,
    )
    registry.register(
        "ts_covariance",
        operators.covariance,
        3,
        description="Rolling covariance over the last d days.",
        input_type_signatures=(("matrix", "matrix", "scalar"),),
        output_type="matrix",
        family="cross_field",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.12,
    )
    registry.register(
        "ts_decay_linear",
        operators.decay_linear,
        2,
        description="Linearly weighted moving average over the last d days.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="smoothing",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.30,
    )
    registry.register(
        "ts_rank",
        operators.ts_rank,
        2,
        description="Time-series rank of the latest observation in the last d days.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="time_series",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.05,
    )
    registry.register(
        "ts_sum",
        operators.ts_sum,
        2,
        description="Rolling sum over the last d days.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="time_series",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.18,
    )
    registry.register(
        "sign",
        operators.sign,
        1,
        description="Elementwise sign.",
        input_type_signatures=(("matrix",), ("scalar",)),
        output_type="same_as_first",
        family="normalization",
        turnover_hint=0.20,
    )
    registry.register(
        "abs",
        operators.abs_value,
        1,
        description="Elementwise absolute value.",
        input_type_signatures=(("matrix",), ("scalar",)),
        output_type="same_as_first",
        family="transformation",
        turnover_hint=0.00,
    )
    registry.register(
        "log",
        operators.log,
        1,
        description="Natural logarithm for positive values.",
        input_type_signatures=(("matrix",), ("scalar",)),
        output_type="same_as_first",
        family="transformation",
        turnover_hint=0.00,
    )
    registry.register(
        "group_rank",
        operators.group_rank,
        2,
        description="Rank values within each group by timestamp.",
        input_type_signatures=(("matrix", "group"),),
        output_type="matrix",
        family="group",
        turnover_hint=0.05,
    )
    registry.register(
        "group_zscore",
        operators.group_zscore,
        2,
        description="Z-score values within each group by timestamp.",
        input_type_signatures=(("matrix", "group"),),
        output_type="matrix",
        family="group",
        turnover_hint=0.03,
    )
    registry.register(
        "group_neutralize",
        operators.group_neutralize,
        2,
        description="Demean values within each group by timestamp.",
        input_type_signatures=(("matrix", "group"),),
        output_type="matrix",
        family="group",
        turnover_hint=-0.02,
    )


def _register_legacy_aliases(registry: OperatorRegistry) -> None:
    for alias, canonical in BRAIN_OPERATOR_ALIASES.items():
        registry.register_alias(alias, canonical)


def _register_local_only_specs(registry: OperatorRegistry) -> None:
    registry.register(
        "returns",
        operators.returns,
        1,
        2,
        description="Percentage return over a time window. Local helper; not enabled by default for BRAIN submission.",
        input_type_signatures=(("matrix",), ("matrix", "scalar")),
        output_type="matrix",
        family="time_series",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=0.95,
    )
    registry.register(
        "clip",
        operators.clip,
        3,
        description="Elementwise clipping. Local-only helper; not a canonical BRAIN operator.",
        input_type_signatures=(("matrix", "scalar", "scalar"), ("scalar", "scalar", "scalar")),
        output_type="same_as_first",
        family="transformation",
        parameter_requirements={"lower": "numeric_literal", "upper": "numeric_literal"},
        turnover_hint=-0.05,
    )


def _signature_matches(expected: tuple[OperatorType, ...], actual: tuple[OperatorType, ...]) -> bool:
    if len(expected) != len(actual):
        return False
    return all(left == right for left, right in zip(expected, actual, strict=True))
