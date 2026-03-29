from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def bucket_horizon(window: int | None) -> str:
    if window is None or window <= 0:
        return "unknown"
    if window <= 3:
        return "very_short"
    if window <= 10:
        return "short"
    if window <= 20:
        return "medium"
    return "long"


def bucket_complexity(complexity: int) -> str:
    if complexity <= 5:
        return "simple"
    if complexity <= 10:
        return "moderate"
    if complexity <= 16:
        return "layered"
    return "complex"


def bucket_turnover_hint(turnover_hint: float) -> str:
    if turnover_hint <= -0.15:
        return "low"
    if turnover_hint <= 0.15:
        return "balanced"
    if turnover_hint <= 0.50:
        return "active"
    return "very_active"


@dataclass(frozen=True, slots=True)
class FeatureGene:
    primary_field: str
    primary_family: str
    secondary_field: str = ""
    secondary_family: str = ""
    auxiliary_field: str = ""
    auxiliary_family: str = ""
    group_field: str = ""
    liquidity_field: str = ""

    def field_names(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name in (
                self.primary_field,
                self.secondary_field,
                self.auxiliary_field,
                self.group_field,
                self.liquidity_field,
            )
            if field_name
        )

    def field_families(self) -> tuple[str, ...]:
        return tuple(
            family
            for family in (
                self.primary_family,
                self.secondary_family,
                self.auxiliary_family,
                "group" if self.group_field else "",
                "liquidity" if self.liquidity_field else "",
            )
            if family
        )


@dataclass(frozen=True, slots=True)
class TransformGene:
    motif: str
    primitive_transform: str
    secondary_transform: str = ""
    pair_operator: str = ""
    arithmetic_operator: str = "-"
    residualization_mode: str = "difference"

    def operator_path(self) -> tuple[str, ...]:
        return tuple(value for value in (self.primitive_transform, self.secondary_transform, self.pair_operator) if value)


@dataclass(frozen=True, slots=True)
class HorizonGene:
    fast_window: int
    slow_window: int = 0
    context_window: int = 0

    @property
    def primary_window(self) -> int:
        return self.slow_window or self.fast_window

    @property
    def horizon_bucket(self) -> str:
        return bucket_horizon(self.primary_window)


@dataclass(frozen=True, slots=True)
class WrapperGene:
    pre_wrappers: tuple[str, ...] = ()
    post_wrappers: tuple[str, ...] = ()

    def all_wrappers(self) -> tuple[str, ...]:
        return tuple(self.pre_wrappers) + tuple(self.post_wrappers)


@dataclass(frozen=True, slots=True)
class RegimeGene:
    conditioning_mode: str = "none"
    conditioning_field: str = ""
    invert_condition: bool = False


@dataclass(frozen=True, slots=True)
class TurnoverGene:
    smoothing_operator: str = ""
    smoothing_window: int = 0
    turnover_hint: float = 0.0

    @property
    def turnover_bucket(self) -> str:
        return bucket_turnover_hint(self.turnover_hint)


@dataclass(frozen=True, slots=True)
class ComplexityGene:
    target_depth: int = 3
    binary_branching: int = 1
    wrapper_budget: int = 1


@dataclass(frozen=True, slots=True)
class Genome:
    feature_gene: FeatureGene
    transform_gene: TransformGene
    horizon_gene: HorizonGene
    wrapper_gene: WrapperGene = field(default_factory=WrapperGene)
    regime_gene: RegimeGene = field(default_factory=RegimeGene)
    turnover_gene: TurnoverGene = field(default_factory=TurnoverGene)
    complexity_gene: ComplexityGene = field(default_factory=ComplexityGene)
    source_mode: str = "random"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Genome":
        if not payload:
            raise ValueError("Genome payload must not be empty.")
        return cls(
            feature_gene=FeatureGene(**payload["feature_gene"]),
            transform_gene=TransformGene(**payload["transform_gene"]),
            horizon_gene=HorizonGene(**payload["horizon_gene"]),
            wrapper_gene=WrapperGene(**payload.get("wrapper_gene", {})),
            regime_gene=RegimeGene(**payload.get("regime_gene", {})),
            turnover_gene=TurnoverGene(**payload.get("turnover_gene", {})),
            complexity_gene=ComplexityGene(**payload.get("complexity_gene", {})),
            source_mode=str(payload.get("source_mode") or "random"),
        )

    @property
    def motif(self) -> str:
        return self.transform_gene.motif

    @property
    def operator_path(self) -> tuple[str, ...]:
        path = list(self.transform_gene.operator_path())
        if self.turnover_gene.smoothing_operator:
            path.append(self.turnover_gene.smoothing_operator)
        return tuple(path)

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.feature_gene.field_names()

    @property
    def field_families(self) -> tuple[str, ...]:
        return self.feature_gene.field_families()

    @property
    def horizon_bucket(self) -> str:
        return self.horizon_gene.horizon_bucket

    @property
    def turnover_bucket(self) -> str:
        return self.turnover_gene.turnover_bucket

    @property
    def complexity_bucket(self) -> str:
        estimated_complexity = max(
            3,
            self.complexity_gene.target_depth + self.complexity_gene.binary_branching + len(self.wrapper_gene.all_wrappers()),
        )
        return bucket_complexity(estimated_complexity)

    @property
    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def family_signature_payload(self) -> dict[str, Any]:
        return {
            "motif": self.motif,
            "field_families": tuple(sorted(set(self.field_families))),
            "operator_path": self.operator_path,
            "horizon_bucket": self.horizon_bucket,
            "turnover_bucket": self.turnover_bucket,
            "complexity_bucket": self.complexity_bucket,
            "conditioning_mode": self.regime_gene.conditioning_mode,
        }


@dataclass(frozen=True, slots=True)
class GenomeRenderResult:
    genome: Genome
    expression: str
    normalized_expression: str
    family_signature: str
    operator_path: tuple[str, ...]
    field_names: tuple[str, ...]
    field_families: tuple[str, ...]
    wrappers: tuple[str, ...]
    horizon_bucket: str
    turnover_bucket: str
    complexity_bucket: str
    repair_actions: tuple[str, ...] = ()

