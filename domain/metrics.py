from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from domain.candidate import AlphaCandidate


@dataclass(frozen=True, slots=True)
class ObjectiveVector:
    fitness: float = 0.0
    sharpe: float = 0.0
    eligibility: float = 0.0
    robustness: float = 0.0
    novelty: float = 0.0
    diversity: float = 0.0
    turnover_cost: float = 0.0
    complexity_cost: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ObjectiveVector:
        return cls(
            fitness=float(payload.get("fitness", 0.0) or 0.0),
            sharpe=float(payload.get("sharpe", 0.0) or 0.0),
            eligibility=float(payload.get("eligibility", 0.0) or 0.0),
            robustness=float(payload.get("robustness", 0.0) or 0.0),
            novelty=float(payload.get("novelty", 0.0) or 0.0),
            diversity=float(payload.get("diversity", 0.0) or 0.0),
            turnover_cost=float(payload.get("turnover_cost", 0.0) or 0.0),
            complexity_cost=float(payload.get("complexity_cost", 0.0) or 0.0),
        )


@dataclass(frozen=True, slots=True)
class StructuralSignature:
    operators: tuple[str, ...]
    operator_families: tuple[str, ...]
    operator_path: tuple[str, ...]
    fields: tuple[str, ...]
    field_families: tuple[str, ...]
    lookbacks: tuple[int, ...]
    wrappers: tuple[str, ...]
    depth: int
    complexity: int
    complexity_bucket: str
    horizon_bucket: str
    turnover_bucket: str
    motif: str
    family_signature: str
    subexpressions: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    sharpe: float
    max_drawdown: float
    win_rate: float
    average_return: float
    turnover: float
    observation_count: int
    cumulative_return: float
    fitness: float


@dataclass(frozen=True, slots=True)
class TestResult:
    name: str
    passed: bool
    details: dict[str, Any]


@dataclass(slots=True, frozen=True)
class CandidateScore:
    candidate: AlphaCandidate
    objective_vector: ObjectiveVector
    local_heuristic_score: float
    novelty_score: float
    family_score: float
    structural_signature: StructuralSignature
    diversity_score: float = 0.0
    duplicate_risk: float = 0.0
    crowding_penalty: float = 0.0
    regime_fit: float = 0.0
    composite_score: float | None = None
    archive_reason: str | None = None
    reason_codes: tuple[str, ...] = ()
    ranking_rationale: dict[str, object] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        if self.composite_score is not None:
            return float(self.composite_score)
        return (
            self.local_heuristic_score
            + 0.15 * self.novelty_score
            + 0.20 * self.family_score
            + 0.20 * self.diversity_score
        )
