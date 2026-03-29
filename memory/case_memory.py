from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from generator.genome import Genome
from memory.pattern_memory import StructuralSignature


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
    def from_dict(cls, payload: dict[str, Any]) -> "ObjectiveVector":
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
class CaseMemoryRecord:
    run_id: str
    alpha_id: str
    regime_key: str
    metric_source: str
    family_signature: str
    structural_signature: StructuralSignature
    genome_hash: str
    genome: Genome | None
    motif: str
    field_families: tuple[str, ...]
    operator_path: tuple[str, ...]
    complexity_bucket: str
    turnover_bucket: str
    horizon_bucket: str
    mutation_mode: str
    parent_family_signatures: tuple[str, ...]
    fail_tags: tuple[str, ...]
    success_tags: tuple[str, ...]
    objective_vector: ObjectiveVector
    outcome_score: float
    created_at: str


@dataclass(frozen=True, slots=True)
class CaseAggregate:
    support: int
    avg_outcome: float
    avg_fitness: float
    avg_sharpe: float
    avg_eligibility: float
    avg_robustness: float
    avg_novelty: float
    avg_turnover_cost: float
    avg_complexity_cost: float
    success_rate: float
    failure_rate: float


@dataclass(frozen=True, slots=True)
class CaseMemorySnapshot:
    regime_key: str
    cases: tuple[CaseMemoryRecord, ...] = ()
    family_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    motif_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    mutation_mode_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    failure_combo_counts: dict[str, int] = field(default_factory=dict)

    def repeated_failure_count(self, *parts: str) -> int:
        key = "|".join(part for part in parts if part)
        return int(self.failure_combo_counts.get(key, 0))


class CaseMemoryService:
    def genome_from_metadata(self, metadata: dict[str, Any] | None) -> Genome | None:
        payload = (metadata or {}).get("genome")
        if not isinstance(payload, dict):
            return None
        try:
            return Genome.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            return None

    def record_from_persisted_payload(
        self,
        *,
        row: dict[str, Any],
        structural_signature: StructuralSignature,
    ) -> CaseMemoryRecord:
        genome_payload = json.loads(row["genome_json"] or "{}")
        return CaseMemoryRecord(
            run_id=row["run_id"],
            alpha_id=row["alpha_id"],
            regime_key=row["regime_key"],
            metric_source=row["metric_source"],
            family_signature=row["family_signature"],
            structural_signature=structural_signature,
            genome_hash=row["genome_hash"],
            genome=Genome.from_dict(genome_payload) if genome_payload else None,
            motif=row["motif"],
            field_families=tuple(json.loads(row["field_families_json"] or "[]")),
            operator_path=tuple(json.loads(row["operator_path_json"] or "[]")),
            complexity_bucket=row["complexity_bucket"],
            turnover_bucket=row["turnover_bucket"],
            horizon_bucket=row["horizon_bucket"],
            mutation_mode=row["mutation_mode"],
            parent_family_signatures=tuple(json.loads(row["parent_family_signatures_json"] or "[]")),
            fail_tags=tuple(json.loads(row["fail_tags_json"] or "[]")),
            success_tags=tuple(json.loads(row["success_tags_json"] or "[]")),
            objective_vector=ObjectiveVector.from_dict(json.loads(row["objective_vector_json"] or "{}")),
            outcome_score=float(row["outcome_score"]),
            created_at=row["created_at"],
        )

    def build_snapshot(self, regime_key: str, records: Iterable[CaseMemoryRecord]) -> CaseMemorySnapshot:
        cases = tuple(sorted(records, key=lambda item: (item.created_at, item.alpha_id)))
        return CaseMemorySnapshot(
            regime_key=regime_key,
            cases=cases,
            family_stats=self._aggregate(cases, key_fn=lambda case: case.family_signature),
            motif_stats=self._aggregate(cases, key_fn=lambda case: case.motif),
            mutation_mode_stats=self._aggregate(cases, key_fn=lambda case: case.mutation_mode),
            failure_combo_counts=self._failure_combo_counts(cases),
        )

    def predict_objectives(
        self,
        *,
        generation_metadata: dict[str, Any],
        signature: StructuralSignature,
        snapshot: CaseMemorySnapshot | None,
        novelty_score: float,
        diversity_score: float,
    ) -> ObjectiveVector:
        if snapshot is None or not snapshot.cases:
            complexity_cost = min(1.0, signature.complexity / 20.0)
            turnover_cost = self._turnover_cost(signature.turnover_bucket)
            return ObjectiveVector(
                fitness=0.0,
                sharpe=0.0,
                eligibility=0.0,
                robustness=0.0,
                novelty=novelty_score,
                diversity=diversity_score,
                turnover_cost=turnover_cost,
                complexity_cost=complexity_cost,
            )

        motif = str(generation_metadata.get("motif") or signature.motif or "")
        mutation_mode = str(generation_metadata.get("mutation_mode") or "")
        family_stats = snapshot.family_stats.get(signature.family_signature)
        motif_stats = snapshot.motif_stats.get(motif)
        mutation_stats = snapshot.mutation_mode_stats.get(mutation_mode)
        aggregates = [item for item in (family_stats, motif_stats, mutation_stats) if item is not None]
        if not aggregates:
            aggregates = list(snapshot.family_stats.values())[:1]
        avg_fitness = sum(item.avg_fitness for item in aggregates) / len(aggregates)
        avg_sharpe = sum(item.avg_sharpe for item in aggregates) / len(aggregates)
        avg_eligibility = sum(item.avg_eligibility for item in aggregates) / len(aggregates)
        avg_robustness = sum(item.avg_robustness for item in aggregates) / len(aggregates)
        avg_turnover_cost = sum(item.avg_turnover_cost for item in aggregates) / len(aggregates)
        avg_complexity_cost = sum(item.avg_complexity_cost for item in aggregates) / len(aggregates)
        return ObjectiveVector(
            fitness=avg_fitness,
            sharpe=avg_sharpe,
            eligibility=avg_eligibility,
            robustness=avg_robustness,
            novelty=novelty_score,
            diversity=diversity_score,
            turnover_cost=max(avg_turnover_cost, self._turnover_cost(signature.turnover_bucket)),
            complexity_cost=max(avg_complexity_cost, min(1.0, signature.complexity / 20.0)),
        )

    def mutation_mode_preferences(
        self,
        *,
        family_signature: str,
        fail_tags: Iterable[str],
        snapshot: CaseMemorySnapshot | None,
    ) -> dict[str, float]:
        base = {
            "exploit_local": 1.0,
            "structural": 1.0,
            "crossover": 1.0,
            "novelty": 1.0,
            "repair": 1.0,
        }
        if snapshot is None or not snapshot.cases:
            return base
        family_cases = [case for case in snapshot.cases if case.family_signature == family_signature]
        for case in family_cases:
            base[case.mutation_mode or "exploit_local"] = base.get(case.mutation_mode or "exploit_local", 1.0) + max(
                -0.5,
                min(1.5, case.outcome_score),
            )
        fail_tag_set = set(fail_tags)
        if {"high_turnover", "excessive_complexity", "brain_rejected"} & fail_tag_set:
            base["repair"] += 1.5
        if {"duplicate_family_no_improvement", "poor_fitness"} & fail_tag_set:
            base["novelty"] += 1.0
            base["structural"] += 0.75
        return base

    def repeated_failure_count(
        self,
        *,
        snapshot: CaseMemorySnapshot | None,
        family_signature: str,
        motif: str,
        mutation_mode: str,
    ) -> int:
        if snapshot is None:
            return 0
        return snapshot.repeated_failure_count(family_signature, motif, mutation_mode)

    def _aggregate(
        self,
        cases: tuple[CaseMemoryRecord, ...],
        *,
        key_fn,
    ) -> dict[str, CaseAggregate]:
        grouped: dict[str, list[CaseMemoryRecord]] = {}
        for case in cases:
            key = str(key_fn(case) or "")
            if not key:
                continue
            grouped.setdefault(key, []).append(case)
        aggregates: dict[str, CaseAggregate] = {}
        for key, rows in grouped.items():
            total = len(rows)
            aggregates[key] = CaseAggregate(
                support=total,
                avg_outcome=sum(item.outcome_score for item in rows) / total,
                avg_fitness=sum(item.objective_vector.fitness for item in rows) / total,
                avg_sharpe=sum(item.objective_vector.sharpe for item in rows) / total,
                avg_eligibility=sum(item.objective_vector.eligibility for item in rows) / total,
                avg_robustness=sum(item.objective_vector.robustness for item in rows) / total,
                avg_novelty=sum(item.objective_vector.novelty for item in rows) / total,
                avg_turnover_cost=sum(item.objective_vector.turnover_cost for item in rows) / total,
                avg_complexity_cost=sum(item.objective_vector.complexity_cost for item in rows) / total,
                success_rate=sum(1 for item in rows if item.outcome_score > 0) / total,
                failure_rate=sum(1 for item in rows if item.outcome_score <= 0) / total,
            )
        return aggregates

    def _failure_combo_counts(self, cases: tuple[CaseMemoryRecord, ...]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in cases:
            if case.outcome_score > 0:
                continue
            key = "|".join(
                part
                for part in (
                    case.family_signature,
                    case.motif,
                    case.mutation_mode,
                )
                if part
            )
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _turnover_cost(self, turnover_bucket: str) -> float:
        return {
            "low": 0.15,
            "balanced": 0.35,
            "active": 0.65,
            "very_active": 0.90,
        }.get(turnover_bucket, 0.50)

