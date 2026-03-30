from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


def operator_path_key(operator_path: Iterable[str] | None) -> str:
    values = [str(item) for item in (operator_path or ()) if str(item)]
    return ">".join(values[:4]) if values else "none"


@dataclass(slots=True)
class GenerationDiversityTracker:
    seen_genome_hashes: set[str] = field(default_factory=set)
    seen_structural_keys: set[str] = field(default_factory=set)
    accepted_motifs: Counter[str] = field(default_factory=Counter)
    accepted_field_families: Counter[str] = field(default_factory=Counter)
    accepted_operators: Counter[str] = field(default_factory=Counter)
    accepted_operator_paths: Counter[str] = field(default_factory=Counter)
    duplicate_by_mutation_mode: Counter[str] = field(default_factory=Counter)
    duplicate_by_motif: Counter[str] = field(default_factory=Counter)
    duplicate_by_operator_path: Counter[str] = field(default_factory=Counter)
    duplicate_by_operator: Counter[str] = field(default_factory=Counter)
    duplicate_by_lineage: Counter[str] = field(default_factory=Counter)

    def check_pre_dedup(
        self,
        *,
        genome_hash: str,
        structural_key: str,
        normalized_expression: str,
        existing_normalized: set[str],
    ) -> str | None:
        if genome_hash and genome_hash in self.seen_genome_hashes:
            return "structural_duplicate_expression"
        if structural_key and structural_key in self.seen_structural_keys:
            return "structural_duplicate_expression"
        if genome_hash:
            self.seen_genome_hashes.add(genome_hash)
        if structural_key:
            self.seen_structural_keys.add(structural_key)
        if normalized_expression and normalized_expression in existing_normalized:
            return "duplicate_normalized_expression"
        return None

    def record_candidate(
        self,
        *,
        motif: str,
        operator_path: tuple[str, ...],
        field_families: tuple[str, ...],
    ) -> None:
        if motif:
            self.accepted_motifs[motif] += 1
        path_key = operator_path_key(operator_path)
        self.accepted_operator_paths[path_key] += 1
        for operator in operator_path:
            if operator:
                self.accepted_operators[operator] += 1
        for family in field_families:
            if family:
                self.accepted_field_families[family] += 1

    def record_duplicate(
        self,
        *,
        mutation_mode: str,
        motif: str,
        operator_path: tuple[str, ...],
        lineage_key: str,
    ) -> None:
        if mutation_mode:
            self.duplicate_by_mutation_mode[mutation_mode] += 1
        if motif:
            self.duplicate_by_motif[motif] += 1
        path_key = operator_path_key(operator_path)
        self.duplicate_by_operator_path[path_key] += 1
        for operator in operator_path:
            if operator:
                self.duplicate_by_operator[operator] += 1
        if lineage_key:
            self.duplicate_by_lineage[lineage_key] += 1

    def motif_weight(self, motif: str) -> float:
        pressure = (0.10 * self.accepted_motifs.get(motif, 0)) + (0.75 * self.duplicate_by_motif.get(motif, 0))
        return max(0.35, 1.0 / (1.0 + pressure))

    def field_family_weight(self, family: str) -> float:
        pressure = 0.20 * self.accepted_field_families.get(family, 0)
        return max(0.40, 1.0 / (1.0 + pressure))

    def operator_weight(self, operator: str) -> float:
        pressure = (0.20 * self.accepted_operators.get(operator, 0)) + (0.75 * self.duplicate_by_operator.get(operator, 0))
        return max(0.35, 1.0 / (1.0 + pressure))

    def mutation_mode_weight(self, mode: str) -> float:
        pressure = self.duplicate_by_mutation_mode.get(mode, 0)
        return max(0.10, 1.0 / (1.0 + (0.85 * pressure)))

    def lineage_weight(self, lineage_key: str) -> float:
        pressure = self.duplicate_by_lineage.get(lineage_key, 0)
        return max(0.10, 1.0 / (1.0 + pressure))
