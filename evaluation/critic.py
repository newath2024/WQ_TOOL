from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

from core.config import AdaptiveGenerationConfig, EvaluationConfig, GenerationConfig


@dataclass(frozen=True, slots=True)
class MutationHint:
    hint: str
    reason: str


@dataclass(slots=True)
class AlphaDiagnosis:
    fail_tags: list[str] = field(default_factory=list)
    success_tags: list[str] = field(default_factory=list)
    mutation_hints: list[MutationHint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fail_tags": list(self.fail_tags),
            "success_tags": list(self.success_tags),
            "mutation_hints": [asdict(hint) for hint in self.mutation_hints],
            "notes": list(self.notes),
        }


class AlphaCritic:
    def __init__(
        self,
        adaptive_config: AdaptiveGenerationConfig,
        evaluation_config: EvaluationConfig,
        generation_config: GenerationConfig,
    ) -> None:
        self.adaptive_config = adaptive_config
        self.evaluation_config = evaluation_config
        self.generation_config = generation_config

    def diagnose_pre_round(self, evaluations: list) -> dict[str, AlphaDiagnosis]:
        threshold = self.adaptive_config.critic_thresholds
        validation_fitnesses = [evaluation.split_metrics["validation"].fitness for evaluation in evaluations]
        validation_median = statistics.median(validation_fitnesses) if validation_fitnesses else 0.0

        diagnoses: dict[str, AlphaDiagnosis] = {}
        for evaluation in evaluations:
            diagnosis = AlphaDiagnosis()
            train = evaluation.split_metrics["train"]
            validation = evaluation.split_metrics["validation"]
            complexity = evaluation.candidate.complexity
            lookbacks = list(getattr(evaluation.structural_signature, "lookbacks", ()))
            median_lookback = statistics.median(lookbacks) if lookbacks else None

            if validation.turnover > self.evaluation_config.max_turnover * threshold.turnover_warning_fraction:
                self._add_fail_tag(diagnosis, "high_turnover", "Validation turnover is near or above the warning band.")
                diagnosis.mutation_hints.append(
                    MutationHint(hint="smoothen_and_slow_down", reason="Reduce turnover with slower and smoother signals.")
                )

            train_fitness_abs = max(abs(train.fitness), 1e-6)
            relative_gap = abs(train.fitness - validation.fitness) / train_fitness_abs
            if relative_gap > threshold.overfit_gap_threshold or (train.sharpe > 0 > validation.sharpe) or (
                train.sharpe < 0 < validation.sharpe
            ):
                self._add_fail_tag(
                    diagnosis,
                    "overfit_train_validation_gap",
                    "Train/validation performance gap is too large or changes sign.",
                )
                diagnosis.mutation_hints.append(
                    MutationHint(hint="simplify_and_stabilize", reason="Reduce overfitting by simplifying the structure.")
                )

            if evaluation.stability_score < self.evaluation_config.min_stability:
                self._add_fail_tag(diagnosis, "low_stability", "Train/validation stability is below the minimum.")
                diagnosis.mutation_hints.append(
                    MutationHint(hint="prefer_stable_genes", reason="Favor historically stable genes and longer windows.")
                )

            if complexity > self.generation_config.complexity_limit * threshold.complexity_warning_fraction:
                self._add_fail_tag(diagnosis, "excessive_complexity", "Expression complexity is too close to the limit.")
                diagnosis.mutation_hints.append(
                    MutationHint(hint="reduce_complexity", reason="Reduce wrappers, nesting depth, or binary branching.")
                )

            if median_lookback is not None and median_lookback <= threshold.noisy_short_horizon_max_lookback and (
                validation.turnover > self.evaluation_config.max_turnover * threshold.turnover_warning_fraction
                or evaluation.stability_score < self.evaluation_config.min_stability
            ):
                self._add_fail_tag(
                    diagnosis,
                    "noisy_short_horizon",
                    "Short-horizon lookbacks combined with turnover/stability issues.",
                )
                diagnosis.mutation_hints.append(
                    MutationHint(hint="lengthen_lookbacks", reason="Move away from short-horizon windows.")
                )

            if validation.fitness <= 0 or (validation.fitness < validation_median and train.fitness > 0):
                self._add_fail_tag(diagnosis, "weak_validation", "Validation fitness is weak relative to the round.")
                diagnosis.mutation_hints.append(
                    MutationHint(hint="favor_robust_windows", reason="Favor more robust windows and simpler transforms.")
                )

            diagnoses[evaluation.candidate.alpha_id] = diagnosis
        return diagnoses

    def diagnose_post_round(
        self,
        evaluations: list,
        diagnoses: dict[str, AlphaDiagnosis],
        selected_ids: set[str],
        duplicate_map: dict[str, dict],
    ) -> dict[str, AlphaDiagnosis]:
        novelty_threshold = self.adaptive_config.critic_thresholds.novelty_success_threshold
        for evaluation in evaluations:
            diagnosis = diagnoses.setdefault(evaluation.candidate.alpha_id, AlphaDiagnosis())
            if evaluation.passed_filters:
                self._add_success_tag(diagnosis, "passed_validation_filters")
            if evaluation.submission_tests and all(result.passed for result in evaluation.submission_tests.values()):
                self._add_success_tag(diagnosis, "robust_submission")
            if evaluation.behavioral_novelty_score >= novelty_threshold:
                self._add_success_tag(diagnosis, "behaviorally_novel")
            if evaluation.candidate.alpha_id in selected_ids:
                self._add_success_tag(diagnosis, "selected_top_alpha")

            duplicate = duplicate_map.get(evaluation.candidate.alpha_id)
            if duplicate:
                self._add_fail_tag(
                    diagnosis,
                    "high_correlation_with_existing",
                    f"Correlated with incumbent {duplicate['incumbent_alpha_id']} and removed during dedup.",
                )
                diagnosis.mutation_hints.append(
                    MutationHint(
                        hint="diversify_feature_family",
                        reason="Shift to a different feature family or operator family to reduce correlation.",
                    )
                )
        return diagnoses

    def _add_fail_tag(self, diagnosis: AlphaDiagnosis, tag: str, note: str) -> None:
        if tag not in diagnosis.fail_tags:
            diagnosis.fail_tags.append(tag)
        diagnosis.notes.append(note)

    def _add_success_tag(self, diagnosis: AlphaDiagnosis, tag: str) -> None:
        if tag not in diagnosis.success_tags:
            diagnosis.success_tags.append(tag)
