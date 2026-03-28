from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from alpha.evaluator import evaluate_expression
from alpha.parser import parse_expression
from alpha.validator import validate_expression
from backtest.engine import run_backtest
from backtest.metrics import PerformanceMetrics, compute_performance_metrics
from backtest.simulation import build_subuniverse_masks
from core.config import AppConfig, PeriodConfig, load_config
from core.logging import configure_logging, get_logger
from core.run_context import RunContext
from core.signatures import build_simulation_signature
from data.loader import load_market_data
from evaluation.critic import AlphaCritic
from evaluation.dedup import deduplicate_with_diagnostics, returns_correlation, signal_correlation
from evaluation.filtering import EvaluatedAlpha, apply_quality_filters, build_evaluated_alpha
from evaluation.ranking import rank_evaluations
from evaluation.submission import TestResult, build_submission_tests
from features.registry import build_registry
from features.transforms import ResearchMatrices, build_research_matrices
from generator.engine import AlphaCandidate, AlphaGenerationEngine
from generator.guided_generator import GuidedGenerator
from memory.pattern_memory import PatternMemoryService
from storage.models import MetricRecord, SelectionRecord, SimulationCacheRecord, SubmissionTestRecord
from storage.repository import SQLiteRepository


def build_parser() -> argparse.ArgumentParser:
    def add_common_arguments(parser: argparse.ArgumentParser, suppress_defaults: bool) -> None:
        default = argparse.SUPPRESS if suppress_defaults else "config/default.yaml"
        parser.add_argument("--config", default=default, help="Path to the YAML config file.")
        parser.add_argument(
            "--run-id",
            default=argparse.SUPPRESS if suppress_defaults else None,
            help="Run identifier to reuse or inspect.",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            default=argparse.SUPPRESS if suppress_defaults else False,
            help="Reuse the latest run when no run-id is provided.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=argparse.SUPPRESS if suppress_defaults else None,
            help="Override the generation seed.",
        )
        parser.add_argument(
            "--log-level",
            default=argparse.SUPPRESS if suppress_defaults else None,
            help="Override the configured log level.",
        )

    common = argparse.ArgumentParser(add_help=False)
    add_common_arguments(common, suppress_defaults=True)

    parser = argparse.ArgumentParser(description="Automated alpha generation research platform")
    add_common_arguments(parser, suppress_defaults=False)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("load-data", help="Load, validate, and summarize the configured dataset.", parents=[common])

    generate_parser = subparsers.add_parser("generate", help="Generate alpha candidates.", parents=[common])
    generate_parser.add_argument("--count", type=int, default=None, help="Total number of candidates to generate.")

    subparsers.add_parser("evaluate", help="Backtest and evaluate stored alpha candidates.", parents=[common])

    top_parser = subparsers.add_parser("top", help="Display selected top alphas.", parents=[common])
    top_parser.add_argument("--limit", type=int, default=20, help="Number of rows to display.")

    report_parser = subparsers.add_parser("report", help="Display a summary report for the run.", parents=[common])
    report_parser.add_argument("--limit", type=int, default=10, help="Number of top alphas to include.")

    memory_patterns_parser = subparsers.add_parser(
        "memory-top-patterns",
        help="Display the highest scoring structural patterns for the current regime.",
        parents=[common],
    )
    memory_patterns_parser.add_argument("--limit", type=int, default=10, help="Number of patterns to display.")
    memory_patterns_parser.add_argument("--kind", default=None, help="Optional pattern kind filter.")

    memory_failed_parser = subparsers.add_parser(
        "memory-failed-patterns",
        help="Display the most failure-prone patterns for the current regime.",
        parents=[common],
    )
    memory_failed_parser.add_argument("--limit", type=int, default=10, help="Number of patterns to display.")

    memory_genes_parser = subparsers.add_parser(
        "memory-top-genes",
        help="Display the strongest reusable subexpression genes for the current regime.",
        parents=[common],
    )
    memory_genes_parser.add_argument("--limit", type=int, default=10, help="Number of genes to display.")

    lineage_parser = subparsers.add_parser("lineage", help="Display lineage for one alpha.", parents=[common])
    lineage_parser.add_argument("--alpha-id", required=True, help="Alpha identifier to inspect.")

    mutate_parser = subparsers.add_parser("mutate", help="Create mutated variants from top-ranked alphas.", parents=[common])
    mutate_parser.add_argument("--from-top", type=int, default=20, help="Number of top parents to use.")
    mutate_parser.add_argument("--count", type=int, default=200, help="Number of mutations to attempt.")

    pipeline_parser = subparsers.add_parser(
        "run-full-pipeline",
        help="Run load, generate, evaluate, and select.",
        parents=[common],
    )
    pipeline_parser.add_argument("--count", type=int, default=None, help="Override the initial generation count.")
    return parser


def resolve_run_context(
    repository: SQLiteRepository,
    config_path: str,
    seed: int,
    run_id: str | None,
    resume: bool,
    command: str,
) -> RunContext:
    if run_id:
        existing = repository.get_run(run_id)
        if existing:
            return RunContext(
                run_id=existing.run_id,
                seed=existing.seed,
                started_at=existing.started_at,
                config_path=existing.config_path,
            )
        return RunContext.create(seed=seed, config_path=config_path, run_id=run_id)

    if resume or command in {
        "generate",
        "evaluate",
        "mutate",
        "top",
        "report",
        "memory-top-patterns",
        "memory-failed-patterns",
        "memory-top-genes",
        "lineage",
    }:
        existing = repository.get_latest_run()
        if existing:
            return RunContext(
                run_id=existing.run_id,
                seed=existing.seed,
                started_at=existing.started_at,
                config_path=existing.config_path,
            )

    return RunContext.create(seed=seed, config_path=config_path)


def init_run(repository: SQLiteRepository, config: AppConfig, context: RunContext, status: str) -> None:
    repository.upsert_run(
        run_id=context.run_id,
        seed=context.seed,
        config_path=context.config_path,
        config_snapshot=yaml.safe_dump(config.to_dict(), sort_keys=False),
        status=status,
        started_at=context.started_at,
    )


def load_dataset(config: AppConfig, context: RunContext, stage: str):
    logger = get_logger(__name__, run_id=context.run_id, stage=stage)
    bundle = load_market_data(config.data, logger, aux_config=config.aux_data)
    logger.info("Loaded market data summary: %s", bundle.summary())
    return bundle


def load_research_context(config: AppConfig, context: RunContext, stage: str) -> tuple:
    bundle = load_dataset(config, context, stage=stage)
    matrices = build_research_matrices(bundle.get_timeframe_data(config.backtest.timeframe))
    memory_service = PatternMemoryService()
    regime_key = memory_service.build_regime_key(bundle.fingerprint, config)
    return bundle, matrices, regime_key, memory_service


def slice_frame_by_period(frame: pd.DataFrame, period: PeriodConfig) -> pd.DataFrame:
    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end)
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def slice_series_by_period(series: pd.Series, period: PeriodConfig) -> pd.Series:
    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end)
    return series.loc[(series.index >= start) & (series.index <= end)]


def combine_series_for_periods(series: pd.Series, *periods: PeriodConfig) -> pd.Series:
    pieces = [slice_series_by_period(series, period) for period in periods]
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


def alpha_candidate_from_record(record, parent_refs: list[dict[str, str]] | None = None) -> AlphaCandidate:
    generation_metadata = json.loads(record.generation_metadata) if getattr(record, "generation_metadata", None) else {}
    if parent_refs:
        generation_metadata["parent_refs"] = parent_refs
    return AlphaCandidate(
        alpha_id=record.alpha_id,
        expression=record.expression,
        normalized_expression=record.normalized_expression,
        generation_mode=record.generation_mode,
        parent_ids=tuple(parent["alpha_id"] for parent in parent_refs or []),
        complexity=record.complexity,
        created_at=record.created_at,
        generation_metadata=generation_metadata,
    )


def build_split_metrics(
    net_returns: pd.Series,
    turnover: pd.Series,
    config: AppConfig,
) -> dict[str, PerformanceMetrics]:
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
    memory_service: PatternMemoryService,
) -> float:
    if not references:
        return 0.5
    max_signal_corr = 0.0
    max_returns_corr = 0.0
    for reference in references:
        signal_corr = signal_correlation(evaluation.validation_signal, reference["validation_signal"])
        returns_corr = returns_correlation(evaluation.validation_returns, reference["validation_returns"])
        max_signal_corr = max(max_signal_corr, abs(signal_corr))
        max_returns_corr = max(max_returns_corr, abs(returns_corr))
    return memory_service.behavioral_novelty(max_signal_corr, max_returns_corr)


def evaluate_run(
    repository: SQLiteRepository,
    config: AppConfig,
    context: RunContext,
) -> tuple[list[EvaluatedAlpha], list[MetricRecord], list[SelectionRecord], str, str]:
    logger = get_logger(__name__, run_id=context.run_id, stage="evaluate")
    registry = build_registry(config.generation.allowed_operators)
    bundle, matrices, regime_key, memory_service = load_research_context(config, context, stage="evaluate-data")
    alpha_records = repository.list_alpha_records(context.run_id)
    if not alpha_records:
        logger.warning("No alpha candidates found for run %s.", context.run_id)
        return [], [], [], regime_key, datetime.now(timezone.utc).isoformat()

    evaluations: list[EvaluatedAlpha] = []
    metrics: list[MetricRecord] = []
    allowed_fields = set(config.generation.allowed_fields)
    group_fields = set(matrices.group_fields)
    fail_fast = config.runtime.fail_fast
    timestamp = datetime.now(timezone.utc).isoformat()
    parent_refs_map = repository.get_parent_refs(context.run_id)
    novelty_references = repository.alpha_history.get_novelty_references(
        regime_key,
        config.adaptive_generation.novelty_reference_top_k,
    )
    simulation_snapshot = yaml.safe_dump(
        {
            "simulation": asdict(config.simulation),
            "backtest": asdict(config.backtest),
            "submission_tests": asdict(config.submission_tests),
        },
        sort_keys=True,
    )
    subuniverse_masks = build_subuniverse_masks(matrices, config.simulation.subuniverses)

    for index, record in enumerate(alpha_records, start=1):
        candidate = alpha_candidate_from_record(record, parent_refs=parent_refs_map.get(record.alpha_id))
        signature = build_alpha_simulation_signature(candidate, bundle, config)
        structural_signature = memory_service.extract_signature(record.expression)
        observations = memory_service.build_observations(structural_signature)
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
                memory_service=memory_service,
            )
            repository.replace_submission_tests(
                context.run_id,
                candidate.alpha_id,
                build_submission_test_records(context.run_id, candidate.alpha_id, evaluation.submission_tests, timestamp),
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
        signal_threshold=config.evaluation.signal_correlation_threshold,
        returns_threshold=config.evaluation.returns_correlation_threshold,
    )
    ranked = rank_evaluations(deduped)[: config.evaluation.top_k]

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
                    run_id=context.run_id,
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
            run_id=context.run_id,
            alpha_id=evaluation.candidate.alpha_id,
            rank=rank,
            selected_at=timestamp,
            validation_fitness=evaluation.split_metrics["validation"].fitness,
            reason=f"selected_after_filtering_and_dedup;submission_passes={evaluation.submission_passes}",
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
    return evaluations, metrics, selection_records, regime_key, timestamp


def build_alpha_simulation_signature(candidate: AlphaCandidate, bundle, config: AppConfig) -> str:
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


def cmd_load_data(repository: SQLiteRepository, config: AppConfig, context: RunContext) -> int:
    init_run(repository, config, context, status="loading_data")
    bundle = load_dataset(config, context, stage="load-data")
    repository.save_dataset_summary(context.run_id, bundle.summary())
    repository.update_run_status(context.run_id, "data_loaded")
    return 0


def cmd_generate(
    repository: SQLiteRepository,
    config: AppConfig,
    context: RunContext,
    count: int | None,
) -> int:
    init_run(repository, config, context, status="generating")
    logger = get_logger(__name__, run_id=context.run_id, stage="generate")
    registry = build_registry(config.generation.allowed_operators)
    existing = repository.list_existing_normalized_expressions(context.run_id)
    total_count = count or (config.generation.template_count + config.generation.grammar_count)
    if config.adaptive_generation.enabled:
        bundle, _, regime_key, _ = load_research_context(config, context, stage="generate-data")
        snapshot = repository.alpha_history.load_snapshot(
            regime_key=regime_key,
            parent_pool_size=config.adaptive_generation.parent_pool_size,
        )
        engine = GuidedGenerator(
            generation_config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            memory_service=PatternMemoryService(),
        )
        candidates = engine.generate(count=total_count, snapshot=snapshot, existing_normalized=existing)
        logger.info("Adaptive generation used regime %s with %s learned patterns.", regime_key[:12], len(snapshot.patterns))
        repository.save_dataset_summary(context.run_id, bundle.summary())
    else:
        engine = AlphaGenerationEngine(config=config.generation, registry=registry)
        candidates = engine.generate(count=total_count, existing_normalized=existing)
    inserted = repository.save_alpha_candidates(context.run_id, candidates)
    repository.update_run_status(context.run_id, "generated")
    logger.info("Generated %s candidates and inserted %s new rows.", len(candidates), inserted)
    return 0


def cmd_mutate(
    repository: SQLiteRepository,
    config: AppConfig,
    context: RunContext,
    from_top: int,
    count: int,
) -> int:
    init_run(repository, config, context, status="mutating")
    logger = get_logger(__name__, run_id=context.run_id, stage="mutate")
    parent_records = repository.get_top_alpha_records(context.run_id, limit=from_top)
    if not parent_records:
        logger.warning("No top alphas available for mutation in run %s.", context.run_id)
        return 1

    existing = repository.list_existing_normalized_expressions(context.run_id)
    registry = build_registry(config.generation.allowed_operators)
    if config.adaptive_generation.enabled:
        _, _, regime_key, _ = load_research_context(config, context, stage="mutate-data")
        snapshot = repository.alpha_history.load_snapshot(
            regime_key=regime_key,
            parent_pool_size=max(config.adaptive_generation.parent_pool_size, from_top * 2),
        )
        selected_parent_ids = {record.alpha_id for record in parent_records}
        parent_pool = [
            parent
            for parent in snapshot.top_parents
            if parent.alpha_id in selected_parent_ids and parent.run_id == context.run_id
        ]
        if not parent_pool:
            parent_pool = list(snapshot.top_parents[:from_top])
        engine = GuidedGenerator(
            generation_config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            memory_service=PatternMemoryService(),
        )
        candidates = engine.generate_mutations(
            count=count,
            snapshot=snapshot,
            parent_pool=parent_pool,
            existing_normalized=existing,
        )
    else:
        engine = AlphaGenerationEngine(config=config.generation, registry=registry)
        parents = [alpha_candidate_from_record(record) for record in parent_records]
        candidates = engine.generate_mutations(parents=parents, count=count, existing_normalized=existing)
    inserted = repository.save_alpha_candidates(context.run_id, candidates)
    repository.update_run_status(context.run_id, "mutated")
    logger.info("Mutated %s candidates and inserted %s new rows.", len(candidates), inserted)
    return 0


def cmd_evaluate(repository: SQLiteRepository, config: AppConfig, context: RunContext) -> int:
    init_run(repository, config, context, status="evaluating")
    evaluations, metric_records, selection_records, regime_key, evaluation_timestamp = evaluate_run(repository, config, context)
    repository.save_metrics(metric_records)
    repository.replace_selections(context.run_id, selection_records)
    if evaluations:
        repository.alpha_history.persist_evaluations(
            run_id=context.run_id,
            regime_key=regime_key,
            evaluations=evaluations,
            pattern_decay=config.adaptive_generation.pattern_decay,
            prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
            created_at=evaluation_timestamp,
        )
    repository.update_run_status(context.run_id, "evaluated")
    return 0


def cmd_top(repository: SQLiteRepository, context: RunContext, limit: int) -> int:
    logger = get_logger(__name__, run_id=context.run_id, stage="top")
    rows = repository.get_top_selections(context.run_id, limit=limit)
    if not rows:
        logger.warning("No selections found for run %s.", context.run_id)
        return 1
    for row in rows:
        print(
            f"rank={row['rank']:>2} alpha_id={row['alpha_id']} "
            f"fitness={row['validation_fitness']:.4f} mode={row['generation_mode']:<8} "
            f"delay={row.get('delay_mode', '-'):<7} neutral={row.get('neutralization', '-'):<16} "
            f"submission={row.get('submission_pass_count', 0):<2} cache={int(row.get('cache_hit') or 0)} "
            f"complexity={row['complexity']:<2} expr={row['expression']}"
        )
    return 0


def cmd_report(repository: SQLiteRepository, config: AppConfig, context: RunContext, limit: int) -> int:
    logger = get_logger(__name__, run_id=context.run_id, stage="report")
    run = repository.get_run(context.run_id)
    if run is None:
        logger.warning("Run %s was not found.", context.run_id)
        return 1
    bundle, _, regime_key, _ = load_research_context(config, context, stage="report-data")
    top_rows = repository.get_top_selections(context.run_id, limit=limit)
    cache_stats = repository.get_cache_stats(context.run_id)
    submission_rows = repository.get_submission_tests_for_run(context.run_id)
    top_gene_rows = repository.alpha_history.get_top_genes(regime_key, limit=1)
    fail_tag_rows = repository.alpha_history.get_run_fail_tag_summary(context.run_id)
    generation_mix = repository.get_generation_mix(context.run_id)
    print(f"run_id={run.run_id} status={run.status} started_at={run.started_at}")
    if cache_stats["validation_rows"] > 0:
        cache_rate = cache_stats["cache_hits"] / cache_stats["validation_rows"]
    else:
        cache_rate = 0.0
    print(
        f"cache_hits={cache_stats['cache_hits']} validation_rows={cache_stats['validation_rows']} "
        f"cache_hit_rate={cache_rate:.2%}"
    )
    if top_rows:
        print("top_alphas:")
        for row in top_rows:
            print(
                f"  rank={row['rank']} alpha_id={row['alpha_id']} fitness={row['validation_fitness']:.4f} "
                f"delay={row.get('delay_mode', '-')} neutral={row.get('neutralization', '-')} "
                f"submission={row.get('submission_pass_count', 0)} expr={row['expression']}"
            )
    if submission_rows:
        summary: dict[str, dict[str, int]] = {}
        for row in submission_rows:
            test_name = row["test_name"]
            bucket = summary.setdefault(test_name, {"passed": 0, "total": 0})
            bucket["passed"] += int(row["passed"])
            bucket["total"] += 1
        print("submission_tests:")
        for test_name, bucket in summary.items():
            print(f"  {test_name}: {bucket['passed']}/{bucket['total']} passed")
    if top_gene_rows:
        gene = top_gene_rows[0]
        print(f"top_gene: kind={gene['pattern_kind']} score={gene['pattern_score']:.4f} value={gene['pattern_value']}")
    if fail_tag_rows:
        print("common_fail_tags:")
        for row in fail_tag_rows[:limit]:
            print(f"  {row['tag']}: {row['total_count']}")
    if generation_mix:
        print("generation_mix:")
        for row in generation_mix:
            print(f"  {row['generation_mode']}: {row['alpha_count']}")
    del bundle
    return 0


def cmd_memory_top_patterns(repository: SQLiteRepository, config: AppConfig, context: RunContext, limit: int, kind: str | None) -> int:
    _, _, regime_key, _ = load_research_context(config, context, stage="memory-top-patterns-data")
    rows = repository.alpha_history.get_top_patterns(regime_key=regime_key, limit=limit, pattern_kind=kind)
    if not rows:
        return 1
    for row in rows:
        print(
            f"kind={row['pattern_kind']:<13} score={row['pattern_score']:.4f} support={row['support']:<3} "
            f"success={row['success_count']:<3} failure={row['failure_count']:<3} value={row['pattern_value']}"
        )
    return 0


def cmd_memory_failed_patterns(repository: SQLiteRepository, config: AppConfig, context: RunContext, limit: int) -> int:
    _, _, regime_key, _ = load_research_context(config, context, stage="memory-failed-patterns-data")
    rows = repository.alpha_history.get_failed_patterns(regime_key=regime_key, limit=limit)
    if not rows:
        return 1
    for row in rows:
        print(
            f"kind={row['pattern_kind']:<13} score={row['pattern_score']:.4f} support={row['support']:<3} "
            f"failure={row['failure_count']:<3} value={row['pattern_value']}"
        )
    return 0


def cmd_memory_top_genes(repository: SQLiteRepository, config: AppConfig, context: RunContext, limit: int) -> int:
    _, _, regime_key, _ = load_research_context(config, context, stage="memory-top-genes-data")
    rows = repository.alpha_history.get_top_genes(regime_key=regime_key, limit=limit)
    if not rows:
        return 1
    for row in rows:
        print(
            f"score={row['pattern_score']:.4f} support={row['support']:<3} success={row['success_count']:<3} "
            f"value={row['pattern_value']}"
        )
    return 0


def cmd_lineage(repository: SQLiteRepository, context: RunContext, alpha_id: str) -> int:
    rows = repository.alpha_history.get_lineage(run_id=context.run_id, alpha_id=alpha_id)
    if not rows:
        return 1
    for row in rows:
        diagnosis = json.loads(row["diagnosis_summary_json"]) if row["diagnosis_summary_json"] else {}
        print(
            f"depth={row['depth']:<2} run_id={row['run_id']} alpha_id={row['alpha_id']} "
            f"outcome={row['outcome_score'] if row['outcome_score'] is not None else 'n/a'} "
            f"fail_tags={','.join(diagnosis.get('fail_tags', [])) or '-'} "
            f"expr={row['expression'] or '-'}"
        )
    return 0


def cmd_run_full_pipeline(
    repository: SQLiteRepository,
    config: AppConfig,
    context: RunContext,
    count: int | None,
) -> int:
    logger = get_logger(__name__, run_id=context.run_id, stage="pipeline")
    init_run(repository, config, context, status="pipeline_started")
    bundle, _, regime_key, _ = load_research_context(config, context, stage="pipeline-load")
    repository.save_dataset_summary(context.run_id, bundle.summary())

    registry = build_registry(config.generation.allowed_operators)
    existing = repository.list_existing_normalized_expressions(context.run_id)
    total_count = count or (config.generation.template_count + config.generation.grammar_count)
    if config.adaptive_generation.enabled:
        snapshot = repository.alpha_history.load_snapshot(
            regime_key=regime_key,
            parent_pool_size=config.adaptive_generation.parent_pool_size,
        )
        engine = GuidedGenerator(
            generation_config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            memory_service=PatternMemoryService(),
        )
        generated = engine.generate(count=total_count, snapshot=snapshot, existing_normalized=existing)
    else:
        engine = AlphaGenerationEngine(config=config.generation, registry=registry)
        generated = engine.generate(count=total_count, existing_normalized=existing)
    inserted = repository.save_alpha_candidates(context.run_id, generated)
    logger.info("Pipeline generation inserted %s new alphas.", inserted)

    evaluations, metric_records, selection_records, regime_key, evaluation_timestamp = evaluate_run(repository, config, context)
    repository.save_metrics(metric_records)
    repository.replace_selections(context.run_id, selection_records)
    if evaluations:
        repository.alpha_history.persist_evaluations(
            run_id=context.run_id,
            regime_key=regime_key,
            evaluations=evaluations,
            pattern_decay=config.adaptive_generation.pattern_decay,
            prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
            created_at=evaluation_timestamp,
        )
    repository.update_run_status(context.run_id, "completed", finished=True)
    logger.info("Pipeline completed with %s selected alphas.", len(selection_records))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    log_level = args.log_level or config.runtime.log_level
    configure_logging(log_level)

    seed = args.seed if args.seed is not None else config.generation.random_seed
    repository = SQLiteRepository(config.storage.path)
    try:
        context = resolve_run_context(
            repository=repository,
            config_path=str(Path(args.config).resolve()),
            seed=seed,
            run_id=args.run_id,
            resume=args.resume,
            command=args.command,
        )

        if args.command == "load-data":
            return cmd_load_data(repository, config, context)
        if args.command == "generate":
            return cmd_generate(repository, config, context, count=args.count)
        if args.command == "evaluate":
            return cmd_evaluate(repository, config, context)
        if args.command == "top":
            return cmd_top(repository, context, limit=args.limit)
        if args.command == "report":
            return cmd_report(repository, config, context, limit=args.limit)
        if args.command == "memory-top-patterns":
            return cmd_memory_top_patterns(repository, config, context, limit=args.limit, kind=args.kind)
        if args.command == "memory-failed-patterns":
            return cmd_memory_failed_patterns(repository, config, context, limit=args.limit)
        if args.command == "memory-top-genes":
            return cmd_memory_top_genes(repository, config, context, limit=args.limit)
        if args.command == "lineage":
            return cmd_lineage(repository, context, alpha_id=args.alpha_id)
        if args.command == "mutate":
            return cmd_mutate(repository, config, context, from_top=args.from_top, count=args.count)
        if args.command == "run-full-pipeline":
            return cmd_run_full_pipeline(repository, config, context, count=args.count)
        parser.error(f"Unsupported command {args.command}")
        return 2
    finally:
        repository.close()


if __name__ == "__main__":
    raise SystemExit(main())
