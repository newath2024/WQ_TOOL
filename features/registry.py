from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from features import operators


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    function: Callable
    min_args: int
    max_args: int
    description: str


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
    ) -> None:
        self._operators[name] = OperatorSpec(
            name=name,
            function=function,
            min_args=min_args,
            max_args=max_args if max_args is not None else min_args,
            description=description,
        )

    def get(self, name: str) -> OperatorSpec:
        return self._operators[name]

    def contains(self, name: str) -> bool:
        return name in self._operators

    def names(self) -> set[str]:
        return set(self._operators)

    def items(self):
        return self._operators.items()


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


def build_default_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    registry.register("delay", operators.delay, 2, description="Shift a field by N periods.")
    registry.register("delta", operators.delta, 2, description="Difference from N periods ago.")
    registry.register("returns", operators.returns, 1, 2, description="Percentage return over N periods.")
    registry.register("rolling_mean", operators.rolling_mean, 2, description="Rolling arithmetic mean.")
    registry.register("rolling_std", operators.rolling_std, 2, description="Rolling standard deviation.")
    registry.register("rolling_min", operators.rolling_min, 2, description="Rolling minimum.")
    registry.register("rolling_max", operators.rolling_max, 2, description="Rolling maximum.")
    registry.register("rank", operators.rank, 1, description="Cross-sectional rank by timestamp.")
    registry.register("zscore", operators.zscore, 1, description="Cross-sectional z-score by timestamp.")
    registry.register("correlation", operators.correlation, 3, description="Rolling correlation.")
    registry.register("covariance", operators.covariance, 3, description="Rolling covariance.")
    registry.register("decay_linear", operators.decay_linear, 2, description="Linearly weighted moving average.")
    registry.register("ts_rank", operators.ts_rank, 2, description="Time-series rank of the latest observation.")
    registry.register("ts_sum", operators.ts_sum, 2, description="Rolling sum.")
    registry.register("ts_mean", operators.ts_mean, 2, description="Rolling mean.")
    registry.register("ts_std", operators.ts_std, 2, description="Rolling standard deviation.")
    registry.register("sign", operators.sign, 1, description="Elementwise sign.")
    registry.register("abs", operators.abs_value, 1, description="Elementwise absolute value.")
    registry.register("log", operators.log, 1, description="Natural logarithm for positive values.")
    registry.register("clip", operators.clip, 3, description="Elementwise clipping.")
    registry.register("group_rank", operators.group_rank, 2, description="Rank values within each group by timestamp.")
    registry.register("group_zscore", operators.group_zscore, 2, description="Z-score values within each group by timestamp.")
    registry.register("group_neutralize", operators.group_neutralize, 2, description="Demean values within each group by timestamp.")
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
        )
    return configured
