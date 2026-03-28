from __future__ import annotations

import pandas as pd

from backtest.metrics import PerformanceMetrics
from core.config import (
    AdaptiveGenerationConfig,
    CriticThresholdConfig,
    EvaluationConfig,
    GenerationConfig,
)
from evaluation.critic import AlphaCritic
from evaluation.filtering import build_evaluated_alpha
from evaluation.submission import TestResult as SubmissionTestResult
from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemoryService


def make_metrics(
    *,
    sharpe: float,
    turnover: float,
    fitness: float,
    observation_count: int = 10,
    max_drawdown: float = -0.10,
    win_rate: float = 0.55,
    average_return: float = 0.01,
    cumulative_return: float = 0.10,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        average_return=average_return,
        turnover=turnover,
        observation_count=observation_count,
        cumulative_return=cumulative_return,
        fitness=fitness,
    )


def make_candidate(expression: str, complexity: int) -> AlphaCandidate:
    return AlphaCandidate(
        alpha_id=f"alpha-{abs(hash((expression, complexity))) % 10_000}",
        expression=expression,
        normalized_expression=expression,
        generation_mode="guided_mutation",
        parent_ids=(),
        complexity=complexity,
        created_at="2026-01-01T00:00:00+00:00",
    )


def build_critic() -> AlphaCritic:
    return AlphaCritic(
        adaptive_config=AdaptiveGenerationConfig(
            critic_thresholds=CriticThresholdConfig(
                turnover_warning_fraction=0.85,
                overfit_gap_threshold=0.35,
                complexity_warning_fraction=0.80,
                noisy_short_horizon_max_lookback=3,
                novelty_success_threshold=0.70,
                score_prior_weight=3.0,
            )
        ),
        evaluation_config=EvaluationConfig(
            min_sharpe=-1.0,
            max_turnover=10.0,
            min_observations=3,
            max_drawdown=1.0,
            min_stability=0.70,
            signal_correlation_threshold=0.95,
            returns_correlation_threshold=0.95,
            top_k=5,
        ),
        generation_config=GenerationConfig(
            allowed_fields=["open", "high", "low", "close", "volume", "returns"],
            allowed_operators=["rank", "delta", "zscore", "ts_mean", "decay_linear"],
            lookbacks=[2, 3, 5, 10],
            max_depth=5,
            complexity_limit=20,
            template_count=8,
            grammar_count=8,
            mutation_count=4,
            normalization_wrappers=["rank", "zscore", "sign"],
            random_seed=7,
        ),
    )


def test_alpha_critic_assigns_fail_tags_and_hints() -> None:
    critic = build_critic()
    service = PatternMemoryService()
    evaluation = build_evaluated_alpha(
        candidate=make_candidate("rank(delta(close, 2))", complexity=18),
        split_metrics={
            "train": make_metrics(sharpe=1.60, turnover=2.0, fitness=1.40),
            "validation": make_metrics(sharpe=-0.20, turnover=9.10, fitness=-0.40),
            "test": make_metrics(sharpe=3.50, turnover=0.10, fitness=3.30),
        },
        validation_signal=pd.DataFrame({"AAA": [1.0, -1.0]}),
        validation_returns=pd.Series([0.01, -0.02]),
        structural_signature=service.extract_signature("rank(delta(close, 2))"),
    )

    diagnoses = critic.diagnose_pre_round([evaluation])
    diagnosis = diagnoses[evaluation.candidate.alpha_id]

    assert set(diagnosis.fail_tags) >= {
        "high_turnover",
        "overfit_train_validation_gap",
        "low_stability",
        "excessive_complexity",
        "noisy_short_horizon",
        "weak_validation",
    }
    assert {hint.hint for hint in diagnosis.mutation_hints} >= {
        "smoothen_and_slow_down",
        "simplify_and_stabilize",
        "prefer_stable_genes",
        "reduce_complexity",
        "lengthen_lookbacks",
        "favor_robust_windows",
    }


def test_alpha_critic_post_round_adds_success_and_correlation_tags() -> None:
    critic = build_critic()
    evaluation = build_evaluated_alpha(
        candidate=make_candidate("zscore(ts_mean(volume, 5))", complexity=6),
        split_metrics={
            "train": make_metrics(sharpe=1.2, turnover=1.0, fitness=1.0),
            "validation": make_metrics(sharpe=1.1, turnover=1.1, fitness=0.9),
            "test": make_metrics(sharpe=-4.0, turnover=9.9, fitness=-5.0),
        },
        validation_signal=pd.DataFrame({"AAA": [1.0, 2.0]}),
        validation_returns=pd.Series([0.01, 0.02]),
        submission_tests={
            "subuniverse": SubmissionTestResult(name="subuniverse", passed=True, details={}),
            "ladder": SubmissionTestResult(name="ladder", passed=True, details={}),
        },
    )
    evaluation.passed_filters = True
    evaluation.behavioral_novelty_score = 0.82

    diagnoses = {evaluation.candidate.alpha_id: critic.diagnose_pre_round([evaluation])[evaluation.candidate.alpha_id]}
    diagnoses = critic.diagnose_post_round(
        evaluations=[evaluation],
        diagnoses=diagnoses,
        selected_ids={evaluation.candidate.alpha_id},
        duplicate_map={
            evaluation.candidate.alpha_id: {
                "incumbent_alpha_id": "incumbent-1",
                "signal_correlation": 0.98,
                "returns_correlation": 0.97,
            }
        },
    )
    diagnosis = diagnoses[evaluation.candidate.alpha_id]

    assert set(diagnosis.success_tags) >= {
        "passed_validation_filters",
        "robust_submission",
        "behaviorally_novel",
        "selected_top_alpha",
    }
    assert "high_correlation_with_existing" in diagnosis.fail_tags
    assert "diversify_feature_family" in {hint.hint for hint in diagnosis.mutation_hints}
