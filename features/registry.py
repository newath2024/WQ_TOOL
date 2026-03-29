from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from features import operators


OperatorType = str


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
    "ts_rank",
    "ts_sum",
    "ts_mean",
    "ts_std",
}

GROUP_OPERATORS = {"group_rank", "group_zscore", "group_neutralize"}
NORMALIZATION_OPERATORS = {"rank", "zscore", "sign", "abs"}
IDEMPOTENT_WRAPPERS = {"rank", "zscore", "sign", "abs"}


def build_default_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    registry.register(
        "delay",
        operators.delay,
        2,
        description="Shift a field by N periods.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="same_as_first",
        family="time_series",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=0.30,
    )
    registry.register(
        "delta",
        operators.delta,
        2,
        description="Difference from N periods ago.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="same_as_first",
        family="time_series",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=0.90,
    )
    registry.register(
        "returns",
        operators.returns,
        1,
        2,
        description="Percentage return over N periods.",
        input_type_signatures=(("matrix",), ("matrix", "scalar")),
        output_type="matrix",
        family="time_series",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=0.95,
    )
    registry.register(
        "rolling_mean",
        operators.rolling_mean,
        2,
        description="Rolling arithmetic mean.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="smoothing",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.35,
    )
    registry.register(
        "rolling_std",
        operators.rolling_std,
        2,
        description="Rolling standard deviation.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="volatility",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.20,
    )
    registry.register(
        "rolling_min",
        operators.rolling_min,
        2,
        description="Rolling minimum.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="time_series",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.15,
    )
    registry.register(
        "rolling_max",
        operators.rolling_max,
        2,
        description="Rolling maximum.",
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
        "correlation",
        operators.correlation,
        3,
        description="Rolling correlation.",
        input_type_signatures=(("matrix", "matrix", "scalar"),),
        output_type="matrix",
        family="cross_field",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.10,
    )
    registry.register(
        "covariance",
        operators.covariance,
        3,
        description="Rolling covariance.",
        input_type_signatures=(("matrix", "matrix", "scalar"),),
        output_type="matrix",
        family="cross_field",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.12,
    )
    registry.register(
        "decay_linear",
        operators.decay_linear,
        2,
        description="Linearly weighted moving average.",
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
        description="Time-series rank of the latest observation.",
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
        description="Rolling sum.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="time_series",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.18,
    )
    registry.register(
        "ts_mean",
        operators.ts_mean,
        2,
        description="Rolling mean.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="smoothing",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.30,
    )
    registry.register(
        "ts_std",
        operators.ts_std,
        2,
        description="Rolling standard deviation.",
        input_type_signatures=(("matrix", "scalar"),),
        output_type="matrix",
        family="volatility",
        parameter_requirements={"window": "positive_int"},
        turnover_hint=-0.20,
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
        "clip",
        operators.clip,
        3,
        description="Elementwise clipping.",
        input_type_signatures=(("matrix", "scalar", "scalar"), ("scalar", "scalar", "scalar")),
        output_type="same_as_first",
        family="transformation",
        parameter_requirements={"lower": "numeric_literal", "upper": "numeric_literal"},
        turnover_hint=-0.05,
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
    return registry


def build_registry(allowed_operators: list[str] | None = None) -> OperatorRegistry:
    default_registry = build_default_registry()
    if not allowed_operators:
        return default_registry

    configured = OperatorRegistry()
    for name in allowed_operators:
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
    return configured


def _signature_matches(expected: tuple[OperatorType, ...], actual: tuple[OperatorType, ...]) -> bool:
    if len(expected) != len(actual):
        return False
    return all(left == right for left, right in zip(expected, actual, strict=True))
