from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from generator.genome import Genome
from memory.pattern_memory import BlendDiagnostics, StructuralSignature


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
    region: str
    regime_key: str
    global_regime_key: str
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
    global_regime_key: str = ""
    region: str = ""
    cases: tuple[CaseMemoryRecord, ...] = ()
    family_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    motif_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    mutation_mode_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    operator_path_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    horizon_bucket_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    turnover_bucket_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    complexity_bucket_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    failure_combo_counts: dict[str, int] = field(default_factory=dict)
    sample_count: int = 0
    global_cases: tuple[CaseMemoryRecord, ...] = ()
    global_family_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    global_motif_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    global_mutation_mode_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    global_operator_path_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    global_horizon_bucket_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    global_turnover_bucket_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    global_complexity_bucket_stats: dict[str, CaseAggregate] = field(default_factory=dict)
    global_failure_combo_counts: dict[str, int] = field(default_factory=dict)
    global_sample_count: int = 0
    blend: BlendDiagnostics | None = None

    def repeated_failure_count(self, *parts: str, scope: str = "blended") -> int:
        key = "|".join(part for part in parts if part)
        normalized_scope = str(scope or "blended").strip().lower()
        if normalized_scope == "local":
            return int(self.failure_combo_counts.get(key, 0))
        if normalized_scope == "global":
            return int(self.global_failure_combo_counts.get(key, 0))
        return int(
            round(
                _weighted_value(
                    float(self.failure_combo_counts.get(key, 0)),
                    float(self.global_failure_combo_counts.get(key, 0)),
                    local_available=key in self.failure_combo_counts,
                    global_available=key in self.global_failure_combo_counts,
                    blend=self.blend,
                )
            )
        )

    def stats_for_scope(self, category: str, scope: str = "blended") -> dict[str, CaseAggregate]:
        local_map = self._stats_map(category, "local")
        global_map = self._stats_map(category, "global")
        normalized_scope = str(scope or "blended").strip().lower()
        if normalized_scope == "local":
            return dict(local_map)
        if normalized_scope == "global":
            return dict(global_map)
        keys = set(local_map) | set(global_map)
        return {
            key: _merge_case_aggregate(local_map.get(key), global_map.get(key), self.blend)
            for key in keys
            if _merge_case_aggregate(local_map.get(key), global_map.get(key), self.blend) is not None
        }

    def aggregate_for(self, category: str, key: str, scope: str = "blended") -> CaseAggregate | None:
        return self.stats_for_scope(category, scope).get(key)

    def _stats_map(self, category: str, scope: str) -> dict[str, CaseAggregate]:
        normalized_scope = str(scope or "local").strip().lower()
        prefix = "global_" if normalized_scope == "global" else ""
        mapping = {
            "family": f"{prefix}family_stats",
            "motif": f"{prefix}motif_stats",
            "mutation_mode": f"{prefix}mutation_mode_stats",
            "operator_path": f"{prefix}operator_path_stats",
            "horizon_bucket": f"{prefix}horizon_bucket_stats",
            "turnover_bucket": f"{prefix}turnover_bucket_stats",
            "complexity_bucket": f"{prefix}complexity_bucket_stats",
        }
        return dict(getattr(self, mapping.get(category, "family_stats")))


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
            region=str(row.get("region") or ""),
            regime_key=row["regime_key"],
            global_regime_key=str(row.get("global_regime_key") or ""),
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

    def build_snapshot(
        self,
        regime_key: str,
        records: Iterable[CaseMemoryRecord],
        *,
        region: str = "",
        global_regime_key: str = "",
        global_records: Iterable[CaseMemoryRecord] | None = None,
        blend: BlendDiagnostics | None = None,
    ) -> CaseMemorySnapshot:
        cases = tuple(sorted(records, key=lambda item: (item.created_at, item.alpha_id)))
        fallback_cases = tuple(sorted(global_records or (), key=lambda item: (item.created_at, item.alpha_id)))
        return CaseMemorySnapshot(
            regime_key=regime_key,
            global_regime_key=global_regime_key,
            region=region,
            cases=cases,
            family_stats=self._aggregate(cases, key_fn=lambda case: case.family_signature),
            motif_stats=self._aggregate(cases, key_fn=lambda case: case.motif),
            mutation_mode_stats=self._aggregate(cases, key_fn=lambda case: case.mutation_mode),
            operator_path_stats=self._aggregate(cases, key_fn=lambda case: ">".join(case.operator_path[:4])),
            horizon_bucket_stats=self._aggregate(cases, key_fn=lambda case: case.horizon_bucket),
            turnover_bucket_stats=self._aggregate(cases, key_fn=lambda case: case.turnover_bucket),
            complexity_bucket_stats=self._aggregate(cases, key_fn=lambda case: case.complexity_bucket),
            failure_combo_counts=self._failure_combo_counts(cases),
            sample_count=len(cases),
            global_cases=fallback_cases,
            global_family_stats=self._aggregate(fallback_cases, key_fn=lambda case: case.family_signature),
            global_motif_stats=self._aggregate(fallback_cases, key_fn=lambda case: case.motif),
            global_mutation_mode_stats=self._aggregate(fallback_cases, key_fn=lambda case: case.mutation_mode),
            global_operator_path_stats=self._aggregate(fallback_cases, key_fn=lambda case: ">".join(case.operator_path[:4])),
            global_horizon_bucket_stats=self._aggregate(fallback_cases, key_fn=lambda case: case.horizon_bucket),
            global_turnover_bucket_stats=self._aggregate(fallback_cases, key_fn=lambda case: case.turnover_bucket),
            global_complexity_bucket_stats=self._aggregate(fallback_cases, key_fn=lambda case: case.complexity_bucket),
            global_failure_combo_counts=self._failure_combo_counts(fallback_cases),
            global_sample_count=len(fallback_cases),
            blend=blend,
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
        if snapshot is None or (not snapshot.cases and not snapshot.global_cases):
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
        operator_path_key = ">".join(signature.operator_path[:4])
        aggregates = [
            snapshot.aggregate_for("family", signature.family_signature, scope="blended"),
            snapshot.aggregate_for("motif", motif, scope="blended"),
            snapshot.aggregate_for("mutation_mode", mutation_mode, scope="blended"),
            snapshot.aggregate_for("operator_path", operator_path_key, scope="blended"),
            snapshot.aggregate_for("horizon_bucket", signature.horizon_bucket, scope="blended"),
            snapshot.aggregate_for("turnover_bucket", signature.turnover_bucket, scope="blended"),
            snapshot.aggregate_for("complexity_bucket", signature.complexity_bucket, scope="blended"),
        ]
        aggregates = [item for item in aggregates if item is not None]
        if not aggregates:
            aggregates = list(snapshot.stats_for_scope("family", scope="blended").values())[:1]
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
        if snapshot is None or (not snapshot.cases and not snapshot.global_cases):
            return base
        local_weight = float(snapshot.blend.local_weight) if snapshot.blend is not None else 1.0
        global_weight = float(snapshot.blend.global_weight) if snapshot.blend is not None else 0.0
        fail_tag_set = set(fail_tags)
        for case in [case for case in snapshot.cases if case.family_signature == family_signature]:
            mode = case.mutation_mode or "exploit_local"
            base[mode] = base.get(mode, 1.0) + local_weight * self._mutation_case_bonus(case=case, fail_tag_set=fail_tag_set)
        for case in [case for case in snapshot.global_cases if case.family_signature == family_signature]:
            mode = case.mutation_mode or "exploit_local"
            base[mode] = base.get(mode, 1.0) + global_weight * self._mutation_case_bonus(case=case, fail_tag_set=fail_tag_set)
        if {"high_turnover", "excessive_complexity", "brain_rejected"} & fail_tag_set:
            base["repair"] += 2.0
            base["exploit_local"] *= 0.80
            base["crossover"] *= 0.90
        if {"duplicate_family_no_improvement", "poor_fitness"} & fail_tag_set:
            base["novelty"] += 1.0
            base["structural"] += 0.75
            base["exploit_local"] *= 0.85
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
        return snapshot.repeated_failure_count(family_signature, motif, mutation_mode, scope="blended")

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

    def _mutation_case_bonus(self, *, case: CaseMemoryRecord, fail_tag_set: set[str]) -> float:
        outcome_component = max(-0.5, min(2.0, float(case.outcome_score)))
        success_component = 0.5 if case.outcome_score > 0 else -0.25
        fail_overlap = len(fail_tag_set & set(case.fail_tags))
        fail_overlap_bonus = min(1.5, 0.75 * fail_overlap)
        return max(-0.75, outcome_component + success_component + fail_overlap_bonus)


def _merge_case_aggregate(
    local_aggregate: CaseAggregate | None,
    global_aggregate: CaseAggregate | None,
    blend: BlendDiagnostics | None,
) -> CaseAggregate | None:
    if local_aggregate is None and global_aggregate is None:
        return None
    template = local_aggregate or global_aggregate
    assert template is not None
    return CaseAggregate(
        support=int(
            round(
                _weighted_value(
                    float(local_aggregate.support) if local_aggregate else 0.0,
                    float(global_aggregate.support) if global_aggregate else 0.0,
                    local_available=local_aggregate is not None,
                    global_available=global_aggregate is not None,
                    blend=blend,
                )
            )
        ),
        avg_outcome=_weighted_value(
            float(local_aggregate.avg_outcome) if local_aggregate else 0.0,
            float(global_aggregate.avg_outcome) if global_aggregate else 0.0,
            local_available=local_aggregate is not None,
            global_available=global_aggregate is not None,
            blend=blend,
        ),
        avg_fitness=_weighted_value(
            float(local_aggregate.avg_fitness) if local_aggregate else 0.0,
            float(global_aggregate.avg_fitness) if global_aggregate else 0.0,
            local_available=local_aggregate is not None,
            global_available=global_aggregate is not None,
            blend=blend,
        ),
        avg_sharpe=_weighted_value(
            float(local_aggregate.avg_sharpe) if local_aggregate else 0.0,
            float(global_aggregate.avg_sharpe) if global_aggregate else 0.0,
            local_available=local_aggregate is not None,
            global_available=global_aggregate is not None,
            blend=blend,
        ),
        avg_eligibility=_weighted_value(
            float(local_aggregate.avg_eligibility) if local_aggregate else 0.0,
            float(global_aggregate.avg_eligibility) if global_aggregate else 0.0,
            local_available=local_aggregate is not None,
            global_available=global_aggregate is not None,
            blend=blend,
        ),
        avg_robustness=_weighted_value(
            float(local_aggregate.avg_robustness) if local_aggregate else 0.0,
            float(global_aggregate.avg_robustness) if global_aggregate else 0.0,
            local_available=local_aggregate is not None,
            global_available=global_aggregate is not None,
            blend=blend,
        ),
        avg_novelty=_weighted_value(
            float(local_aggregate.avg_novelty) if local_aggregate else 0.0,
            float(global_aggregate.avg_novelty) if global_aggregate else 0.0,
            local_available=local_aggregate is not None,
            global_available=global_aggregate is not None,
            blend=blend,
        ),
        avg_turnover_cost=_weighted_value(
            float(local_aggregate.avg_turnover_cost) if local_aggregate else 0.0,
            float(global_aggregate.avg_turnover_cost) if global_aggregate else 0.0,
            local_available=local_aggregate is not None,
            global_available=global_aggregate is not None,
            blend=blend,
        ),
        avg_complexity_cost=_weighted_value(
            float(local_aggregate.avg_complexity_cost) if local_aggregate else 0.0,
            float(global_aggregate.avg_complexity_cost) if global_aggregate else 0.0,
            local_available=local_aggregate is not None,
            global_available=global_aggregate is not None,
            blend=blend,
        ),
        success_rate=_weighted_value(
            float(local_aggregate.success_rate) if local_aggregate else 0.0,
            float(global_aggregate.success_rate) if global_aggregate else 0.0,
            local_available=local_aggregate is not None,
            global_available=global_aggregate is not None,
            blend=blend,
        ),
        failure_rate=_weighted_value(
            float(local_aggregate.failure_rate) if local_aggregate else 0.0,
            float(global_aggregate.failure_rate) if global_aggregate else 0.0,
            local_available=local_aggregate is not None,
            global_available=global_aggregate is not None,
            blend=blend,
        ),
    )


def _weighted_value(
    local_value: float,
    global_value: float,
    *,
    local_available: bool,
    global_available: bool,
    blend: BlendDiagnostics | None,
) -> float:
    if local_available and not global_available:
        return float(local_value)
    if global_available and not local_available:
        return float(global_value)
    if not local_available and not global_available:
        return 0.0
    local_weight = float(blend.local_weight) if blend is not None else 1.0
    global_weight = float(blend.global_weight) if blend is not None else 0.0
    total = local_weight + global_weight
    if total <= 0:
        return float(local_value)
    return float((local_weight * local_value + global_weight * global_value) / total)

