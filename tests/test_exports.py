from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from backtest.metrics import PerformanceMetrics
from core.run_context import RunContext
from evaluation.critic import AlphaDiagnosis, MutationHint
from evaluation.filtering import build_evaluated_alpha
from generator.engine import AlphaCandidate
from services.export_service import export_evaluated_alphas, export_generated_alphas
from services.models import CommandEnvironment, EvaluationServiceResult
from storage.models import SelectionRecord
from storage.repository import SQLiteRepository


def _make_environment(run_id: str, command_name: str) -> CommandEnvironment:
    return CommandEnvironment(
        config_path="config/dev.yaml",
        command_name=command_name,
        context=RunContext.create(seed=42, config_path="config/dev.yaml", run_id=run_id),
    )


def _make_candidate() -> AlphaCandidate:
    return AlphaCandidate(
        alpha_id="alpha123",
        expression="rank(delta(close, 5))",
        normalized_expression="rank(delta(close,5))",
        generation_mode="template",
        parent_ids=(),
        complexity=4,
        created_at=datetime.now(timezone.utc).isoformat(),
        generation_metadata={"source": "test"},
    )


def _make_metrics(fitness: float, sharpe: float) -> PerformanceMetrics:
    return PerformanceMetrics(
        sharpe=sharpe,
        max_drawdown=-0.12,
        win_rate=0.55,
        average_return=0.01,
        turnover=0.8,
        observation_count=20,
        cumulative_return=0.15,
        fitness=fitness,
    )


def test_export_generated_alphas_writes_readable_csv(tmp_path) -> None:
    repository = SQLiteRepository(":memory:")
    run_id = "run_generated"
    environment = _make_environment(run_id, "generate")
    repository.upsert_run(
        run_id=run_id,
        seed=42,
        config_path="config/dev.yaml",
        config_snapshot="{}",
        status="generating",
        started_at=environment.context.started_at,
    )
    repository.save_alpha_candidates(run_id, [_make_candidate()])

    export_paths = export_generated_alphas(repository, environment, output_dir=tmp_path)
    generated_frame = pd.read_csv(export_paths["generated_alphas_latest_csv"])

    assert generated_frame.shape[0] == 1
    assert generated_frame.loc[0, "alpha_id"] == "alpha123"
    assert generated_frame.loc[0, "expression"] == "rank(delta(close, 5))"
    assert generated_frame.loc[0, "generation_metadata_json"] == '{"source": "test"}'
    assert pd.notna(generated_frame.loc[0, "parent_count"])

    repository.close()


def test_export_evaluated_alphas_writes_detailed_and_selected_csv(tmp_path) -> None:
    candidate = _make_candidate()
    evaluation = build_evaluated_alpha(
        candidate=candidate,
        split_metrics={
            "train": _make_metrics(fitness=1.2, sharpe=1.4),
            "validation": _make_metrics(fitness=0.8, sharpe=1.1),
            "test": _make_metrics(fitness=0.3, sharpe=0.5),
        },
        validation_signal=pd.DataFrame({"AAA": [0.1, 0.2]}, index=pd.to_datetime(["2021-01-01", "2021-01-02"])),
        validation_returns=pd.Series([0.01, -0.02], index=pd.to_datetime(["2021-01-01", "2021-01-02"])),
        simulation_signature="sig-123",
        regime_key="regime-abc",
        simulation_profile={"delay_mode": "fast_d1", "neutralization": "sector"},
    )
    evaluation.passed_filters = True
    evaluation.submission_passes = 3
    evaluation.behavioral_novelty_score = 0.77
    evaluation.gene_ids = ["gene_a", "gene_b"]
    evaluation.diagnosis = AlphaDiagnosis(
        fail_tags=["high_turnover"],
        success_tags=["selected_top_alpha"],
        mutation_hints=[MutationHint(hint="smoothen_and_slow_down", reason="test")],
    )
    selection_record = SelectionRecord(
        run_id="run_evaluated",
        alpha_id=candidate.alpha_id,
        rank=1,
        selected_at=datetime.now(timezone.utc).isoformat(),
        validation_fitness=0.8,
        reason="selected_after_filtering_and_dedup",
        ranking_rationale_json='{"validation_fitness": 0.8}',
    )
    result = EvaluationServiceResult(
        evaluations=[evaluation],
        metric_records=[],
        selection_records=[selection_record],
        regime_key="regime-abc",
        evaluation_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    environment = _make_environment("run_evaluated", "evaluate")

    export_paths = export_evaluated_alphas(result, environment, output_dir=tmp_path)
    evaluated_frame = pd.read_csv(export_paths["evaluated_alphas_latest_csv"])
    selected_frame = pd.read_csv(export_paths["selected_alphas_latest_csv"])

    assert evaluated_frame.shape[0] == 1
    assert evaluated_frame.loc[0, "alpha_id"] == "alpha123"
    assert evaluated_frame.loc[0, "selected"] == True
    assert evaluated_frame.loc[0, "delay_mode"] == "fast_d1"
    assert evaluated_frame.loc[0, "neutralization"] == "sector"
    assert evaluated_frame.loc[0, "fail_tags"] == "high_turnover"
    assert evaluated_frame.loc[0, "mutation_hints"] == "smoothen_and_slow_down"

    assert selected_frame.shape[0] == 1
    assert selected_frame.loc[0, "rank"] == 1
    assert selected_frame.loc[0, "alpha_id"] == "alpha123"
