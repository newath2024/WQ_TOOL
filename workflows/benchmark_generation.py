from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from core.config import AppConfig
from core.run_context import RunContext
from services.brain_batch_service import BrainBatchService
from services.models import BatchPreparationResult, CommandEnvironment
from services.runtime_service import build_command_environment, init_run
from storage.repository import SQLiteRepository


def benchmark_generation(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    *,
    count: int | None = None,
) -> dict[str, Any]:
    generation_count = count or config.loop.generation_batch_size
    baseline_config = _build_baseline_config(config)
    optimized_config = _build_optimized_config(config)
    baseline = _run_mode(
        source_repository=repository,
        config=baseline_config,
        environment=environment,
        mode_name="legacy_baseline",
        generation_count=generation_count,
    )
    optimized = _run_mode(
        source_repository=repository,
        config=optimized_config,
        environment=environment,
        mode_name="optimized",
        generation_count=generation_count,
    )
    return {
        "baseline": baseline,
        "optimized": optimized,
        "delta": {
            "generation_total_ms_pct": _percent_delta(
                baseline["measured_metrics"].get("generation_total_ms", 0.0),
                optimized["measured_metrics"].get("generation_total_ms", 0.0),
            ),
            "load_research_context_ms_pct": _percent_delta(
                baseline["measured_metrics"].get("load_research_context_ms", 0.0),
                optimized["measured_metrics"].get("load_research_context_ms", 0.0),
            ),
            "attempt_count_pct": _percent_delta(
                float(baseline["measured_metrics"].get("attempt_count", 0)),
                float(optimized["measured_metrics"].get("attempt_count", 0)),
            ),
        },
    }


def _run_mode(
    *,
    source_repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    mode_name: str,
    generation_count: int,
) -> dict[str, Any]:
    with TemporaryDirectory(prefix=f"wq-benchmark-{mode_name}-") as temp_dir:
        temp_repo_path = Path(temp_dir) / f"{mode_name}.sqlite3"
        temp_repository = SQLiteRepository(str(temp_repo_path))
        try:
            source_repository.connection.backup(temp_repository.connection)
            temp_context = RunContext.create(
                seed=environment.context.seed,
                config_path=environment.context.config_path,
                run_id=f"{environment.context.run_id}-{mode_name}",
            )
            temp_environment = build_command_environment(
                config_path=environment.config_path,
                command_name="benchmark-generation",
                context=temp_context,
            )
            init_run(temp_repository, config, temp_environment, status=f"benchmark_{mode_name}")
            batch_service = BrainBatchService(temp_repository)
            warmup_result = batch_service.prepare_service_batch(
                config=config,
                environment=temp_environment,
                count=generation_count,
                mutation_parent_ids=None,
                round_index=1,
            )
            measured_result = batch_service.prepare_service_batch(
                config=config,
                environment=temp_environment,
                count=generation_count,
                mutation_parent_ids=None,
                round_index=2,
            )
            return {
                "mode": mode_name,
                "run_id": temp_environment.context.run_id,
                "warmup_metrics": dict(warmup_result.generation_stage_metrics),
                "measured_metrics": dict(measured_result.generation_stage_metrics),
                "generated_count": len(measured_result.candidates),
                "selected_for_simulation": len(measured_result.selected),
                "top_fail_reasons": dict(measured_result.generation_stage_metrics.get("top_fail_reasons", {})),
            }
        finally:
            temp_repository.close()


def _build_baseline_config(config: AppConfig) -> AppConfig:
    return replace(
        config,
        generation=replace(
            config.generation,
            engine_validation_cache_enabled=False,
        ),
        adaptive_generation=replace(
            config.adaptive_generation,
            max_generation_seconds=0.0,
            max_attempt_multiplier=25,
            max_consecutive_failures=0,
            min_candidates_before_early_exit=max(
                config.adaptive_generation.min_candidates_before_early_exit,
                config.loop.generation_batch_size,
            ),
        ),
        service=replace(
            config.service,
            research_context_cache_enabled=False,
            research_context_cache_ttl_seconds=0,
        ),
    )


def _build_optimized_config(config: AppConfig) -> AppConfig:
    return replace(
        config,
        generation=replace(
            config.generation,
            engine_validation_cache_enabled=config.generation.engine_validation_cache_enabled,
        ),
        adaptive_generation=replace(
            config.adaptive_generation,
            max_generation_seconds=config.adaptive_generation.max_generation_seconds,
            max_attempt_multiplier=config.adaptive_generation.max_attempt_multiplier,
            max_consecutive_failures=config.adaptive_generation.max_consecutive_failures,
            min_candidates_before_early_exit=config.adaptive_generation.min_candidates_before_early_exit,
        ),
        service=replace(
            config.service,
            research_context_cache_enabled=config.service.research_context_cache_enabled,
            research_context_cache_ttl_seconds=config.service.research_context_cache_ttl_seconds,
        ),
    )


def _percent_delta(baseline: float, optimized: float) -> float:
    if baseline == 0:
        return 0.0
    return round(((optimized - baseline) / baseline) * 100.0, 3)
