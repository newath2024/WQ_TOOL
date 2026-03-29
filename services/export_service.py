from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from services.models import CommandEnvironment, EvaluationServiceResult
from storage.repository import SQLiteRepository

DEFAULT_OUTPUT_DIR = Path("outputs")

_GENERATED_COLUMNS = [
    "run_id",
    "alpha_id",
    "region",
    "regime_key",
    "global_regime_key",
    "pattern_local_weight",
    "case_local_weight",
    "expression",
    "normalized_expression",
    "generation_mode",
    "template",
    "fields_used",
    "operators_used",
    "depth",
    "complexity",
    "status",
    "created_at",
    "parent_count",
    "parent_alpha_ids",
    "parent_run_ids",
    "generation_metadata_json",
]

_EVALUATED_COLUMNS = [
    "run_id",
    "alpha_id",
    "region",
    "regime_key",
    "global_regime_key",
    "pattern_local_weight",
    "case_local_weight",
    "expression",
    "normalized_expression",
    "generation_mode",
    "template",
    "fields_used",
    "operators_used",
    "depth",
    "complexity",
    "passed_filters",
    "selected",
    "selection_rank",
    "train_fitness",
    "train_sharpe",
    "validation_fitness",
    "validation_sharpe",
    "validation_turnover",
    "validation_max_drawdown",
    "validation_observation_count",
    "test_fitness",
    "test_sharpe",
    "stability_score",
    "submission_pass_count",
    "behavioral_novelty_score",
    "cache_hit",
    "delay_mode",
    "neutralization",
    "simulation_signature",
    "fail_tags",
    "success_tags",
    "mutation_hints",
    "rejection_reasons",
    "gene_ids",
    "evaluated_at",
]

_SELECTED_COLUMNS = [
    "run_id",
    "rank",
    "alpha_id",
    "region",
    "regime_key",
    "global_regime_key",
    "pattern_local_weight",
    "case_local_weight",
    "expression",
    "generation_mode",
    "template",
    "fields_used",
    "operators_used",
    "depth",
    "complexity",
    "validation_fitness",
    "validation_sharpe",
    "submission_pass_count",
    "behavioral_novelty_score",
    "delay_mode",
    "neutralization",
    "reason",
    "ranking_rationale_json",
    "selected_at",
]


def export_generated_alphas(
    repository: SQLiteRepository,
    environment: CommandEnvironment,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Export the generated alpha table for the active run to readable CSV files."""
    records = repository.list_alpha_records(environment.context.run_id)
    parent_refs = repository.get_parent_refs(environment.context.run_id)
    rows: list[dict[str, Any]] = []
    for record in records:
        parents = parent_refs.get(record.alpha_id, [])
        metadata = _parse_json(record.generation_metadata)
        memory_context = metadata.get("memory_context", {}) if isinstance(metadata, dict) else {}
        pattern_blend = memory_context.get("pattern_blend", {}) if isinstance(memory_context, dict) else {}
        case_blend = memory_context.get("case_blend", {}) if isinstance(memory_context, dict) else {}
        rows.append(
            {
                "run_id": record.run_id,
                "alpha_id": record.alpha_id,
                "region": str(metadata.get("region") or ""),
                "regime_key": str(metadata.get("regime_key") or ""),
                "global_regime_key": str(metadata.get("global_regime_key") or ""),
                "pattern_local_weight": pattern_blend.get("local_weight", ""),
                "case_local_weight": case_blend.get("local_weight", ""),
                "expression": record.expression,
                "normalized_expression": record.normalized_expression,
                "generation_mode": record.generation_mode,
                "template": record.template_name,
                "fields_used": _json_or_pipe(record.fields_used_json),
                "operators_used": _json_or_pipe(record.operators_used_json),
                "depth": record.depth,
                "complexity": record.complexity,
                "status": record.status,
                "created_at": record.created_at,
                "parent_count": len(parents),
                "parent_alpha_ids": _pipe_join(parent["alpha_id"] for parent in parents),
                "parent_run_ids": _pipe_join(parent["run_id"] for parent in parents),
                "generation_metadata_json": _stable_json(record.generation_metadata),
            }
        )
    return _write_export_rows(
        rows=rows,
        columns=_GENERATED_COLUMNS,
        base_name="generated_alphas",
        run_id=environment.context.run_id,
        output_dir=output_dir,
    )


def export_evaluated_alphas(
    result: EvaluationServiceResult,
    environment: CommandEnvironment,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Export detailed evaluation results and selected alphas to CSV files."""
    selection_by_alpha_id = {record.alpha_id: record for record in result.selection_records}
    evaluated_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for evaluation in result.evaluations:
        train = evaluation.split_metrics["train"]
        validation = evaluation.split_metrics["validation"]
        test = evaluation.split_metrics["test"]
        selection = selection_by_alpha_id.get(evaluation.candidate.alpha_id)
        delay_mode = str(evaluation.simulation_profile.get("delay_mode", ""))
        neutralization = str(evaluation.simulation_profile.get("neutralization", ""))
        mutation_hints = _pipe_join(hint.hint for hint in evaluation.diagnosis.mutation_hints)
        memory_context = evaluation.candidate.generation_metadata.get("memory_context", {})
        pattern_blend = memory_context.get("pattern_blend", {}) if isinstance(memory_context, dict) else {}
        case_blend = memory_context.get("case_blend", {}) if isinstance(memory_context, dict) else {}

        evaluated_rows.append(
            {
                "run_id": environment.context.run_id,
                "alpha_id": evaluation.candidate.alpha_id,
                "region": result.region,
                "regime_key": result.regime_key,
                "global_regime_key": result.global_regime_key,
                "pattern_local_weight": pattern_blend.get("local_weight", ""),
                "case_local_weight": case_blend.get("local_weight", ""),
                "expression": evaluation.candidate.expression,
                "normalized_expression": evaluation.candidate.normalized_expression,
                "generation_mode": evaluation.candidate.generation_mode,
                "template": evaluation.candidate.template_name,
                "fields_used": _pipe_join(evaluation.candidate.fields_used),
                "operators_used": _pipe_join(evaluation.candidate.operators_used),
                "depth": evaluation.candidate.depth,
                "complexity": evaluation.candidate.complexity,
                "passed_filters": evaluation.passed_filters,
                "selected": selection is not None,
                "selection_rank": selection.rank if selection is not None else "",
                "train_fitness": train.fitness,
                "train_sharpe": train.sharpe,
                "validation_fitness": validation.fitness,
                "validation_sharpe": validation.sharpe,
                "validation_turnover": validation.turnover,
                "validation_max_drawdown": validation.max_drawdown,
                "validation_observation_count": validation.observation_count,
                "test_fitness": test.fitness,
                "test_sharpe": test.sharpe,
                "stability_score": evaluation.stability_score,
                "submission_pass_count": evaluation.submission_passes,
                "behavioral_novelty_score": evaluation.behavioral_novelty_score,
                "cache_hit": evaluation.cache_hit,
                "delay_mode": delay_mode,
                "neutralization": neutralization,
                "simulation_signature": evaluation.simulation_signature,
                "fail_tags": _pipe_join(evaluation.diagnosis.fail_tags),
                "success_tags": _pipe_join(evaluation.diagnosis.success_tags),
                "mutation_hints": mutation_hints,
                "rejection_reasons": " | ".join(evaluation.rejection_reasons),
                "gene_ids": _pipe_join(evaluation.gene_ids),
                "evaluated_at": result.evaluation_timestamp,
            }
        )

        if selection is not None:
            selected_rows.append(
                {
                    "run_id": environment.context.run_id,
                    "rank": selection.rank,
                    "alpha_id": evaluation.candidate.alpha_id,
                    "region": result.region,
                    "regime_key": result.regime_key,
                    "global_regime_key": result.global_regime_key,
                    "pattern_local_weight": pattern_blend.get("local_weight", ""),
                    "case_local_weight": case_blend.get("local_weight", ""),
                    "expression": evaluation.candidate.expression,
                    "generation_mode": evaluation.candidate.generation_mode,
                    "template": evaluation.candidate.template_name,
                    "fields_used": _pipe_join(evaluation.candidate.fields_used),
                    "operators_used": _pipe_join(evaluation.candidate.operators_used),
                    "depth": evaluation.candidate.depth,
                    "complexity": evaluation.candidate.complexity,
                    "validation_fitness": validation.fitness,
                    "validation_sharpe": validation.sharpe,
                    "submission_pass_count": evaluation.submission_passes,
                    "behavioral_novelty_score": evaluation.behavioral_novelty_score,
                    "delay_mode": delay_mode,
                    "neutralization": neutralization,
                    "reason": selection.reason,
                    "ranking_rationale_json": selection.ranking_rationale_json,
                    "selected_at": selection.selected_at,
                }
            )

    export_paths = {}
    export_paths.update(
        _write_export_rows(
            rows=evaluated_rows,
            columns=_EVALUATED_COLUMNS,
            base_name="evaluated_alphas",
            run_id=environment.context.run_id,
            output_dir=output_dir,
        )
    )
    export_paths.update(
        _write_export_rows(
            rows=selected_rows,
            columns=_SELECTED_COLUMNS,
            base_name="selected_alphas",
            run_id=environment.context.run_id,
            output_dir=output_dir,
        )
    )
    return export_paths


def _write_export_rows(
    rows: list[dict[str, Any]],
    columns: list[str],
    base_name: str,
    run_id: str,
    output_dir: str | Path,
) -> dict[str, str]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    latest_path = output_root / f"{base_name}.csv"
    run_path = output_root / f"{base_name}_{run_id}.csv"
    frame.to_csv(latest_path, index=False)
    frame.to_csv(run_path, index=False)
    return {
        f"{base_name}_latest_csv": str(latest_path),
        f"{base_name}_run_csv": str(run_path),
    }


def _pipe_join(values: Any) -> str:
    normalized = [str(value) for value in values if str(value)]
    return "|".join(normalized)


def _stable_json(value: Any) -> str:
    if not value:
        return "{}"
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return json.dumps(parsed, sort_keys=True)
    return json.dumps(value, sort_keys=True)


def _json_or_pipe(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, list):
            return _pipe_join(parsed)
        return json.dumps(parsed, sort_keys=True)
    if isinstance(value, list):
        return _pipe_join(value)
    return str(value)


def _parse_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
