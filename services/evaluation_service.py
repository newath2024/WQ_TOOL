from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd
import yaml

from alpha.evaluator import evaluate_expression
from alpha.parser import parse_expression
from alpha.validator import validate_expression
from backtest.engine import run_backtest
from backtest.metrics import PerformanceMetrics, compute_performance_metrics
from backtest.simulation import build_subuniverse_masks
from core.config import AppConfig
from core.logging import get_logger
from core.signatures import build_simulation_signature
from evaluation.critic import AlphaCritic
from evaluation.dedup import deduplicate_with_diagnostics, returns_correlation, signal_correlation
from evaluation.filtering import EvaluatedAlpha, apply_quality_filters, build_evaluated_alpha
from evaluation.ranking import rank_evaluations
from evaluation.submission import TestResult, build_submission_tests
from features.registry import build_registry
from generator.engine import AlphaCandidate
from services.data_service import combine_series_for_periods, load_research_context, resolve_field_registry, slice_frame_by_period, slice_series_by_period
from services.export_service import export_evaluated_alphas
from services.models import CommandEnvironment, EvaluationServiceResult
from storage.models import MetricRecord, SelectionRecord, SimulationCacheRecord, SubmissionTestRecord
from storage.repository import SQLiteRepository


def alpha_candidate_from_record(record, parent_refs: list[dict[str, str]] | None = None) -> AlphaCandidate:
    """Rebuild an alpha candidate from persistent storage."""
    generation_metadata = json.loads(record.generation_metadata) if getattr(record, "generation_metadata", None) else {}
    if parent_refs:
        generation_metadata["parent_refs"] = parent_refs
    fields_used = []
    operators_used = []
    if getattr(record, "fields_used_json", ""):
        try:
            fields_used = json.loads(record.fields_used_json)
        except json.JSONDecodeError:
            fields_used = []
    if getattr(record, "operators_used_json", ""):
        try:
            operators_used = json.loads(record.operators_used_json)
        except json.JSONDecodeError:
            operators_used = []
    if (
        "canonical_structural_signature" not in generation_metadata
        and getattr(record, "structural_signature_json", "")
    ):
        try:
            structural_signature = json.loads(record.structural_signature_json or "{}")
        except json.JSONDecodeError:
            structural_signature = {}
        if isinstance(structural_signature, dict) and structural_signature:
            generation_metadata["canonical_structural_signature"] = structural_signature
    return AlphaCandidate(
        alpha_id=record.alpha_id,
        expression=record.expression,
        normalized_expression=record.normalized_expression,
        generation_mode=record.generation_mode,
        parent_ids=tuple(parent["alpha_id"] for parent in parent_refs or []),
        complexity=record.complexity,
        created_at=record.created_at,
        template_name=getattr(record, "template_name", ""),
        fields_used=tuple(fields_used),
        operators_used=tuple(operators_used),
        depth=int(getattr(record, "depth", 0) or 0),
        generation_metadata=generation_metadata,
    )


def build_split_metrics(
    net_returns: pd.Series,
    turnover: pd.Series,
    config: AppConfig,
) -> dict[str, PerformanceMetrics]:
    """Compute train, validation, and test performance metrics."""
    return {
        "train": compute_performance_metrics(
            daily_returns=slice_series_by_period(net_returns, config.splits.train),
            turnover=slice_series_by_period(turnover, config.splits.train),
            annualization_factor=config.backtest.annualization_factor,
            turnover_penalty=config.backtest.turnover_penalty,
            drawdown_penalty=config.backtest.drawdown_penalty,
        ),
        "validation": compute_performance_metrics(
            daily_returns=slice_series_by_period(net_returns, config.splits.validation),
            turnover=slice_series_by_period(turnover, config.splits.validation),
            annualization_factor=config.backtest.annualization_factor,
            turnover_penalty=config.backtest.turnover_penalty,
            drawdown_penalty=config.backtest.drawdown_penalty,
        ),
        "test": compute_performance_metrics(
            daily_returns=slice_series_by_period(net_returns, config.splits.test),
            turnover=slice_series_by_period(turnover, config.splits.test),
            annualization_factor=config.backtest.annualization_factor,
            turnover_penalty=config.backtest.turnover_penalty,
            drawdown_penalty=config.backtest.drawdown_penalty,
        ),
    }


def compute_behavioral_novelty_score(
    evaluation: EvaluatedAlpha,
    references: list[dict],
    signal_memory_service,
) -> float:
    """Compute validation-only novelty from signal and return correlations."""
    if not references:
        return 0.5
    max_signal_corr = 0.0
    max_returns_corr = 0.0
    for reference in references:
        signal_corr = signal_correlation(evaluation.validation_signal, reference["validation_signal"])
        returns_corr = returns_correlation(evaluation.validation_returns, reference["validation_returns"])
        max_signal_corr = max(max_signal_corr, abs(signal_corr))
        max_returns_corr = max(max_returns_corr, abs(returns_corr))
    return signal_memory_service.behavioral_novelty(max_signal_corr, max_returns_corr)


def build_alpha_simulation_signature(candidate: AlphaCandidate, bundle, config: AppConfig) -> str:
    """Build a stable simulation signature for caching evaluation results."""
    return build_simulation_signature(
        {
            "expression": candidate.normalized_expression,
            "timeframe": config.backtest.timeframe,
            "simulation": asdict(config.simulation),
            "backtest": asdict(config.backtest),
            "submission_tests": asdict(config.submission_tests),
            "universe": config.data.universe,
            "dataset_fingerprint": bundle.fingerprint,
            "subuniverses": [asdict(item) for item in config.simulation.subuniverses],
        }
    )


def build_simulation_cache_record(
    evaluation: EvaluatedAlpha,
    simulation_snapshot: str,
    created_at: str,
) -> SimulationCacheRecord:
    """Convert evaluation results into a cache record."""
    return SimulationCacheRecord(
        simulation_signature=evaluation.simulation_signature,
        normalized_expression=evaluation.candidate.normalized_expression,
        simulation_config_snapshot=simulation_snapshot,
        delay_mode=str(evaluation.simulation_profile.get("delay_mode", "d1")),
        neutralization=str(evaluation.simulation_profile.get("neutralization", "none")),
        neutralization_profile=json.dumps(evaluation.simulation_profile, sort_keys=True),
        split_metrics_json=json.dumps(serialize_split_metrics(evaluation.split_metrics), sort_keys=True),
        submission_tests_json=json.dumps(serialize_submission_tests(evaluation.submission_tests), sort_keys=True),
        subuniverse_metrics_json=json.dumps(serialize_subuniverse_metrics(evaluation.subuniverse_metrics), sort_keys=True),
        validation_signal_json=SQLiteRepository.dataframe_to_json(evaluation.validation_signal),
        validation_returns_json=SQLiteRepository.series_to_json(evaluation.validation_returns),
        created_at=created_at,
    )


def rebuild_evaluation_from_cache(candidate: AlphaCandidate, cached: SimulationCacheRecord) -> EvaluatedAlpha:
    """Rehydrate an evaluation object from cached simulation payloads."""
    validation_signal = SQLiteRepository.dataframe_from_json(cached.validation_signal_json)
    validation_signal.index = pd.to_datetime(validation_signal.index)
    validation_returns = SQLiteRepository.series_from_json(cached.validation_returns_json)
    validation_returns.index = pd.to_datetime(validation_returns.index)
    return build_evaluated_alpha(
        candidate=candidate,
        split_metrics=deserialize_split_metrics(json.loads(cached.split_metrics_json)),
        validation_signal=validation_signal,
        validation_returns=validation_returns,
        simulation_signature=cached.simulation_signature,
        submission_tests=deserialize_submission_tests(json.loads(cached.submission_tests_json)),
        subuniverse_metrics=deserialize_subuniverse_metrics(json.loads(cached.subuniverse_metrics_json)),
        cache_hit=True,
        simulation_profile=json.loads(cached.neutralization_profile),
    )


def build_submission_test_records(
    run_id: str,
    alpha_id: str,
    tests: dict[str, TestResult],
    created_at: str,
) -> list[SubmissionTestRecord]:
    """Convert submission test results into persistence records."""
    return [
        SubmissionTestRecord(
            run_id=run_id,
            alpha_id=alpha_id,
            test_name=name,
            passed=result.passed,
            details_json=json.dumps({"name": result.name, "passed": result.passed, "details": result.details}, sort_keys=True),
            created_at=created_at,
        )
        for name, result in tests.items()
    ]


def serialize_split_metrics(split_metrics: dict[str, PerformanceMetrics]) -> dict[str, dict]:
    return {name: asdict(metrics) for name, metrics in split_metrics.items()}


def deserialize_split_metrics(payload: dict[str, dict]) -> dict[str, PerformanceMetrics]:
    return {name: PerformanceMetrics(**values) for name, values in payload.items()}


def serialize_submission_tests(tests: dict[str, TestResult]) -> dict[str, dict]:
    return {
        name: {"name": result.name, "passed": result.passed, "details": result.details}
        for name, result in tests.items()
    }


def deserialize_submission_tests(payload: dict[str, dict]) -> dict[str, TestResult]:
    return {
        name: TestResult(name=values["name"], passed=bool(values["passed"]), details=values["details"])
        for name, values in payload.items()
    }


def serialize_subuniverse_metrics(metrics: dict[str, PerformanceMetrics]) -> dict[str, dict]:
    return {name: asdict(metric) for name, metric in metrics.items()}


def deserialize_subuniverse_metrics(payload: dict[str, dict]) -> dict[str, PerformanceMetrics]:
    return {name: PerformanceMetrics(**values) for name, values in payload.items()}


def evaluate_run(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
) -> EvaluationServiceResult:
    """Evaluate stored alpha candidates for the active run."""
    logger = get_logger(__name__, run_id=environment.context.run_id, stage="evaluate")
    registry = build_registry(
        config.generation.allowed_operators,
        operator_catalog_paths=config.generation.operator_catalog_paths,
    )
    research_context = load_research_context(config, environment, stage="evaluate-data")
    bundle = research_context.bundle
    matrices = research_context.matrices
    regime_key = research_context.regime_key
    memory_service = research_context.memory_service
    field_registry = resolve_field_registry(config, research_context)
    alpha_records = repository.list_alpha_records(environment.context.run_id)
    if not alpha_records:
        logger.warning("No alpha candidates found for run %s.", environment.context.run_id)
        return EvaluationServiceResult(
            evaluations=[],
            metric_records=[],
            selection_records=[],
            region=research_context.region,
            regime_key=regime_key,
            global_regime_key=research_context.global_regime_key,
            evaluation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    evaluations: list[EvaluatedAlpha] = []
    metrics: list[MetricRecord] = []
    allowed_fields = field_registry.allowed_runtime_fields(config.generation.allowed_fields) | {
        spec.name for spec in field_registry.runtime_group_fields()
    }
    group_fields = {spec.name for spec in field_registry.runtime_group_fields()}
    field_types = field_registry.field_types(allowed=allowed_fields)
    field_categories = {name: spec.category for name, spec in field_registry.fields.items()}
    fail_fast = config.runtime.fail_fast
    timestamp = datetime.now(timezone.utc).isoformat()
    parent_refs_map = repository.get_parent_refs(environment.context.run_id)
    novelty_references = repository.alpha_history.get_novelty_references(
        regime_key,
        config.adaptive_generation.novelty_reference_top_k,
    )
    simulation_snapshot = yaml.safe_dump(
        {
            "simulation": asdict(config.simulation),
            "backtest": asdict(config.backtest),
            "submission_tests": asdict(config.submission_tests),
            "evaluation": asdict(config.evaluation),
        },
        sort_keys=True,
    )
    subuniverse_masks = build_subuniverse_masks(matrices, config.simulation.subuniverses)

    for index, record in enumerate(alpha_records, start=1):
        candidate = alpha_candidate_from_record(record, parent_refs=parent_refs_map.get(record.alpha_id))
        signature = build_alpha_simulation_signature(candidate, bundle, config)
        structural_signature = memory_service.extract_signature(
            record.expression,
            generation_metadata=candidate.generation_metadata,
            field_categories=field_categories,
        )
        observations = memory_service.build_observations(
            structural_signature,
            generation_metadata=candidate.generation_metadata,
        )
        gene_ids = [observation.pattern_id for observation in observations if observation.pattern_kind == "subexpression"]
        logger.info("Evaluating alpha %s/%s: %s", index, len(alpha_records), record.alpha_id)
        try:
            cached = repository.get_cached_simulation(signature) if config.simulation.cache_enabled else None
            if cached:
                evaluation = rebuild_evaluation_from_cache(candidate, cached)
                logger.info("Cache hit for alpha %s with signature %s", record.alpha_id, signature)
            else:
                node = parse_expression(record.expression)
                validation = validate_expression(
                    node=node,
                    registry=registry,
                    allowed_fields=allowed_fields,
                    max_depth=config.generation.max_depth,
                    group_fields=group_fields,
                    field_types=field_types,
                    complexity_limit=config.generation.complexity_limit,
                )
                if not validation.is_valid:
                    raise ValueError("; ".join(validation.errors))

                signal = evaluate_expression(
                    node=node,
                    fields=matrices.numeric_fields,
                    registry=registry,
                    group_fields=matrices.group_fields,
                )
                if not isinstance(signal, pd.DataFrame):
                    raise TypeError("Alpha evaluation did not produce a matrix-valued signal.")

                backtest = run_backtest(
                    signal=signal,
                    matrices=matrices,
                    backtest_config=config.backtest,
                    simulation_config=config.simulation,
                )
                split_metrics = build_split_metrics(backtest.net_returns, backtest.turnover, config)
                subuniverse_metrics: dict[str, PerformanceMetrics] = {}
                subuniverse_returns: dict[str, pd.Series] = {}
                for name, mask in subuniverse_masks.items():
                    sub_backtest = run_backtest(
                        signal=signal,
                        matrices=matrices,
                        backtest_config=config.backtest,
                        simulation_config=config.simulation,
                        universe_mask=mask,
                    )
                    subuniverse_returns[name] = sub_backtest.net_returns
                    subuniverse_metrics[name] = compute_performance_metrics(
                        daily_returns=slice_series_by_period(sub_backtest.net_returns, config.splits.validation),
                        turnover=slice_series_by_period(sub_backtest.turnover, config.splits.validation),
                        annualization_factor=config.backtest.annualization_factor,
                        turnover_penalty=config.backtest.turnover_penalty,
                        drawdown_penalty=config.backtest.drawdown_penalty,
                    )

                backtest.subuniverse_returns = subuniverse_returns
                combined_returns = combine_series_for_periods(
                    backtest.net_returns,
                    config.splits.train,
                    config.splits.validation,
                )
                combined_turnover = combine_series_for_periods(
                    backtest.turnover,
                    config.splits.train,
                    config.splits.validation,
                )
                submission_tests = build_submission_tests(
                    validation_metrics=split_metrics["validation"],
                    combined_returns=combined_returns,
                    combined_turnover=combined_turnover,
                    subuniverse_metrics=subuniverse_metrics,
                    config=config,
                )
                evaluation = build_evaluated_alpha(
                    candidate=candidate,
                    split_metrics=split_metrics,
                    validation_signal=slice_frame_by_period(backtest.neutralized_signal, config.splits.validation),
                    validation_returns=slice_series_by_period(backtest.net_returns, config.splits.validation),
                    simulation_signature=signature,
                    regime_key=regime_key,
                    submission_tests=submission_tests,
                    subuniverse_metrics=subuniverse_metrics,
                    cache_hit=False,
                    simulation_profile=backtest.simulation_profile,
                    structural_signature=structural_signature,
                    gene_ids=gene_ids,
                )
                if config.simulation.cache_enabled:
                    repository.save_simulation_cache(
                        build_simulation_cache_record(
                            evaluation=evaluation,
                            simulation_snapshot=simulation_snapshot,
                            created_at=timestamp,
                        )
                    )

            evaluation.regime_key = regime_key
            evaluation.structural_signature = structural_signature
            evaluation.gene_ids = gene_ids
            evaluation.behavioral_novelty_score = compute_behavioral_novelty_score(
                evaluation=evaluation,
                references=novelty_references,
                signal_memory_service=memory_service,
            )
            repository.replace_submission_tests(
                environment.context.run_id,
                candidate.alpha_id,
                build_submission_test_records(
                    environment.context.run_id,
                    candidate.alpha_id,
                    evaluation.submission_tests,
                    timestamp,
                ),
            )
            evaluations.append(evaluation)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Alpha %s failed during evaluation: %s", record.alpha_id, exc)
            if fail_fast:
                raise

    critic = AlphaCritic(config.adaptive_generation, config.evaluation, config.generation)
    diagnoses = critic.diagnose_pre_round(evaluations)
    for evaluation in evaluations:
        evaluation.diagnosis = diagnoses.get(evaluation.candidate.alpha_id, evaluation.diagnosis)

    passed, rejected = apply_quality_filters(evaluations, config.evaluation)
    deduped, duplicate_map = deduplicate_with_diagnostics(
        evaluations=passed,
        signal_threshold=config.evaluation.diversity.signal_correlation_threshold,
        returns_threshold=config.evaluation.diversity.returns_correlation_threshold,
    )
    ranked = rank_evaluations(deduped)[: config.evaluation.ranking.top_k]

    selected_ids = {evaluation.candidate.alpha_id for evaluation in ranked}
    diagnoses = critic.diagnose_post_round(
        evaluations=evaluations,
        diagnoses=diagnoses,
        selected_ids=selected_ids,
        duplicate_map=duplicate_map,
    )
    for evaluation in evaluations:
        selected = evaluation.candidate.alpha_id in selected_ids
        evaluation.diagnosis = diagnoses.get(evaluation.candidate.alpha_id, evaluation.diagnosis)
        evaluation.outcome_score = memory_service.compute_outcome_score(
            validation_fitness=evaluation.split_metrics["validation"].fitness,
            passed_filters=evaluation.passed_filters,
            selected_top_alpha=selected,
            behavioral_novelty_score=evaluation.behavioral_novelty_score,
            fail_tags=evaluation.diagnosis.fail_tags,
        )
        for split_name, performance in evaluation.split_metrics.items():
            metrics.append(
                MetricRecord(
                    run_id=environment.context.run_id,
                    alpha_id=evaluation.candidate.alpha_id,
                    split=split_name,
                    sharpe=performance.sharpe,
                    max_drawdown=performance.max_drawdown,
                    win_rate=performance.win_rate,
                    average_return=performance.average_return,
                    turnover=performance.turnover,
                    observation_count=performance.observation_count,
                    cumulative_return=performance.cumulative_return,
                    fitness=performance.fitness,
                    stability_score=evaluation.stability_score,
                    passed_filters=evaluation.passed_filters,
                    simulation_signature=evaluation.simulation_signature,
                    simulation_config_snapshot=simulation_snapshot,
                    delay_mode=str(evaluation.simulation_profile.get("delay_mode", config.simulation.delay_mode)),
                    neutralization=str(
                        evaluation.simulation_profile.get("neutralization", config.simulation.neutralization)
                    ),
                    neutralization_profile=json.dumps(evaluation.simulation_profile, sort_keys=True),
                    submission_pass_count=evaluation.submission_passes,
                    cache_hit=evaluation.cache_hit,
                    created_at=timestamp,
                )
            )

    selection_records = [
        SelectionRecord(
            run_id=environment.context.run_id,
            alpha_id=evaluation.candidate.alpha_id,
            rank=rank,
            selected_at=timestamp,
            validation_fitness=evaluation.split_metrics["validation"].fitness,
            reason=f"selected_after_filtering_and_dedup;submission_passes={evaluation.submission_passes}",
            ranking_rationale_json=json.dumps(
                {
                    "validation_fitness": evaluation.split_metrics["validation"].fitness,
                    "submission_passes": evaluation.submission_passes,
                    "validation_sharpe": evaluation.split_metrics["validation"].sharpe,
                    "behavioral_novelty_score": evaluation.behavioral_novelty_score,
                    "complexity": evaluation.candidate.complexity,
                    "selection_objectives": {
                        "fitness": evaluation.split_metrics["validation"].fitness,
                        "sharpe": evaluation.split_metrics["validation"].sharpe,
                        "eligibility": 1.0 if evaluation.submission_passes > 0 else 0.0,
                        "robustness": evaluation.stability_score,
                        "novelty": evaluation.behavioral_novelty_score,
                        "diversity": evaluation.behavioral_novelty_score,
                        "turnover_cost": min(1.0, max(0.0, evaluation.split_metrics["validation"].turnover / 3.0)),
                        "complexity_cost": min(1.0, max(0.0, evaluation.candidate.complexity / 20.0)),
                    },
                },
                sort_keys=True,
            ),
        )
        for rank, evaluation in enumerate(ranked, start=1)
    ]

    logger.info(
        "Evaluation complete. total=%s passed=%s rejected=%s selected=%s",
        len(evaluations),
        len(passed),
        len(rejected),
        len(selection_records),
    )
    return EvaluationServiceResult(
        evaluations=evaluations,
        metric_records=metrics,
        selection_records=selection_records,
        region=research_context.region,
        regime_key=regime_key,
        global_regime_key=research_context.global_regime_key,
        evaluation_timestamp=timestamp,
    )


def persist_evaluation_result(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    result: EvaluationServiceResult,
    status: str = "evaluated",
    finished: bool = False,
) -> None:
    """Persist metrics, selections, history, and final run status."""
    repository.save_metrics(result.metric_records)
    repository.replace_selections(environment.context.run_id, result.selection_records)
    if result.evaluations:
        repository.alpha_history.persist_evaluations(
            run_id=environment.context.run_id,
            regime_key=result.regime_key,
            region=result.region,
            global_regime_key=result.global_regime_key,
            evaluations=result.evaluations,
            pattern_decay=config.adaptive_generation.pattern_decay,
            prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
            created_at=result.evaluation_timestamp,
        )
    repository.update_run_status(environment.context.run_id, status, finished=finished)


def evaluate_and_persist(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    status: str = "evaluated",
    finished: bool = False,
) -> EvaluationServiceResult:
    """Run evaluation and persist all resulting artifacts."""
    result = evaluate_run(repository, config, environment)
    persist_evaluation_result(repository, config, environment, result, status=status, finished=finished)
    export_paths = export_evaluated_alphas(result, environment)
    result.export_paths.update(export_paths)
    logger = get_logger(__name__, run_id=environment.context.run_id, stage="evaluate")
    logger.info("Exported evaluated alpha CSV to %s", export_paths["evaluated_alphas_latest_csv"])
    logger.info("Exported selected alpha CSV to %s", export_paths["selected_alphas_latest_csv"])
    return result
