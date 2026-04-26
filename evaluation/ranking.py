from __future__ import annotations

from domain.metrics import ObjectiveVector
from evaluation.multi_objective_selection import MultiObjectiveSelectionService, RankedItem


def rank_evaluations(evaluations):
    selector = MultiObjectiveSelectionService()
    ranked_items: list[RankedItem] = []
    for evaluation in evaluations:
        validation = evaluation.split_metrics["validation"]
        ranked_items.append(
            RankedItem(
                item=evaluation,
                objective_vector=ObjectiveVector(
                    fitness=float(validation.fitness),
                    sharpe=float(validation.sharpe),
                    eligibility=float(evaluation.submission_passes),
                    robustness=float(evaluation.stability_score),
                    novelty=float(evaluation.behavioral_novelty_score),
                    diversity=float(evaluation.behavioral_novelty_score),
                    turnover_cost=min(1.0, max(0.0, float(validation.turnover) / 3.0)),
                    complexity_cost=min(1.0, max(0.0, float(evaluation.candidate.complexity) / 20.0)),
                ),
                family_signature=str(
                    getattr(evaluation.structural_signature, "family_signature", evaluation.candidate.normalized_expression)
                ),
                primary_field_category=str(
                    (evaluation.candidate.generation_metadata.get("field_families") or ["other"])[0]
                ),
                horizon_bucket=str(getattr(evaluation.structural_signature, "horizon_bucket", "unknown")),
                operator_path_key=">".join(getattr(evaluation.structural_signature, "operator_path", ())[:4]) or "none",
                diversity_score=float(evaluation.behavioral_novelty_score),
                exploration_candidate="novelty" in str(evaluation.candidate.generation_mode),
            )
        )
    return [item.item for item in selector.order(ranked_items)]
