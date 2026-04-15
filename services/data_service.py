from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from core.config import AppConfig, PeriodConfig
from core.logging import get_logger
from data.field_registry import (
    FieldRegistry,
    FieldScoreWeights,
    build_field_registry,
    load_runtime_field_values,
)
from data.loader import load_market_data
from data.schema import resolve_path
from features.transforms import build_research_matrices
from memory.pattern_memory import PatternMemoryService, RegionLearningContext
from services.regime_service import RegimeService
from services.models import CommandEnvironment, ResearchContext
from storage.models import FieldCatalogRecord, RegimeSnapshotRecord, RunFieldScoreRecord
from storage.repository import SQLiteRepository


@dataclass(frozen=True, slots=True)
class ResearchContextCacheKey:
    cache_key: str
    config_fingerprint: str
    input_fingerprint: str

    @property
    def short_key(self) -> str:
        return self.cache_key[:12]


@dataclass(slots=True)
class ResearchContextLoadProfile:
    cache_hit: bool = False
    cache_reason: str = "disabled"
    cache_key: str = ""
    config_fingerprint: str = ""
    input_fingerprint: str = ""
    load_research_context_ms: float = 0.0
    build_field_registry_ms: float = 0.0
    prepare_context_ms: float = 0.0
    field_registry_fingerprint: str = ""


@dataclass(slots=True)
class CachedResearchContextResult:
    research_context: ResearchContext
    profile: ResearchContextLoadProfile


@dataclass(slots=True)
class _CachedResearchContextEntry:
    research_context: ResearchContext
    profile: ResearchContextLoadProfile
    cached_at: float
    last_accessed_at: float


@dataclass(slots=True)
class _PersistedResearchMetadataState:
    cache_key: str
    dataset_fingerprint: str
    field_registry_fingerprint: str


def load_dataset(config: AppConfig, environment: CommandEnvironment, stage: str):
    """Load and log the configured dataset."""
    logger = get_logger(__name__, run_id=environment.context.run_id, stage=stage)
    bundle = load_market_data(config.data, logger, aux_config=config.aux_data)
    logger.info("Loaded market data summary: %s", bundle.summary())
    return bundle


def load_research_context(
    config: AppConfig,
    environment: CommandEnvironment,
    stage: str,
) -> ResearchContext:
    """Load bundle, matrices, and memory context for one command."""
    return _build_research_context(config, environment, stage=stage).research_context


class CachedResearchContextProvider:
    def __init__(
        self,
        *,
        enabled: bool = True,
        ttl_seconds: int = 0,
        max_entries: int = 4,
    ) -> None:
        self.enabled = enabled
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_entries = max(1, max_entries)
        self._entries: dict[str, _CachedResearchContextEntry] = {}
        self._persisted_metadata: dict[str, _PersistedResearchMetadataState] = {}
        self._last_cache_key: str | None = None

    def load(
        self,
        config: AppConfig,
        environment: CommandEnvironment,
        *,
        stage: str,
    ) -> CachedResearchContextResult:
        logger = get_logger(__name__, run_id=environment.context.run_id, stage=stage)
        started = time.perf_counter()
        cache_key = _build_research_context_cache_key(config)
        cache_reason = "disabled"
        if self.enabled:
            entry = self._entries.get(cache_key.cache_key)
            now = time.monotonic()
            if entry is not None and not self._is_expired(entry, now=now):
                entry.last_accessed_at = now
                profile = ResearchContextLoadProfile(
                    cache_hit=True,
                    cache_reason="cache_hit",
                    cache_key=cache_key.cache_key,
                    config_fingerprint=cache_key.config_fingerprint,
                    input_fingerprint=cache_key.input_fingerprint,
                    load_research_context_ms=0.0,
                    build_field_registry_ms=0.0,
                    prepare_context_ms=(time.perf_counter() - started) * 1000.0,
                    field_registry_fingerprint=entry.profile.field_registry_fingerprint,
                )
                logger.info("[gen-cache] research_context cache_hit key=%s", cache_key.short_key)
                logger.info("[gen-cache] field_registry cache_hit key=%s reason=reused_context", cache_key.short_key)
                self._last_cache_key = cache_key.cache_key
                return CachedResearchContextResult(research_context=entry.research_context, profile=profile)
            cache_reason = self._resolve_cache_miss_reason(cache_key)

        result = _build_research_context(config, environment, stage=stage)
        result.profile.cache_hit = False
        result.profile.cache_reason = cache_reason
        result.profile.cache_key = cache_key.cache_key
        result.profile.config_fingerprint = cache_key.config_fingerprint
        result.profile.input_fingerprint = cache_key.input_fingerprint
        result.profile.prepare_context_ms = result.profile.load_research_context_ms
        result.profile.field_registry_fingerprint = _fingerprint_field_registry(result.research_context.field_registry)
        logger.info(
            "[gen-cache] research_context cache_miss reason=%s key=%s prepare_context_ms=%.3f",
            cache_reason,
            cache_key.short_key,
            result.profile.prepare_context_ms,
        )
        logger.info(
            "[gen-cache] field_registry cache_miss reason=%s key=%s build_field_registry_ms=%.3f",
            cache_reason,
            cache_key.short_key,
            result.profile.build_field_registry_ms,
        )
        if self.enabled:
            self._entries[cache_key.cache_key] = _CachedResearchContextEntry(
                research_context=result.research_context,
                profile=result.profile,
                cached_at=time.monotonic(),
                last_accessed_at=time.monotonic(),
            )
            self._trim_entries()
        self._last_cache_key = cache_key.cache_key
        return result

    def persist_metadata(
        self,
        repository: SQLiteRepository,
        config: AppConfig,
        environment: CommandEnvironment,
        cache_result: CachedResearchContextResult,
        *,
        round_index: int = 0,
        research_context_override: ResearchContext | None = None,
        removed_field_names: tuple[str, ...] = (),
    ) -> dict[str, bool]:
        run_id = environment.context.run_id
        state = self._persisted_metadata.get(run_id)
        effective_context = research_context_override or cache_result.research_context
        dataset_fingerprint = effective_context.bundle.fingerprint
        field_registry_fingerprint = _fingerprint_field_registry(effective_context.field_registry)
        persist_dataset_summary = (
            state is None
            or state.dataset_fingerprint != dataset_fingerprint
            or state.cache_key != cache_result.profile.cache_key
        )
        persist_field_catalog = (
            state is None
            or state.field_registry_fingerprint != field_registry_fingerprint
        )
        persisted = persist_research_metadata(
            repository,
            config,
            environment,
            effective_context,
            round_index=round_index,
            persist_dataset_summary=persist_dataset_summary,
            persist_field_catalog=persist_field_catalog,
            persist_run_field_scores=persist_field_catalog,
            removed_field_names=removed_field_names,
        )
        self._persisted_metadata[run_id] = _PersistedResearchMetadataState(
            cache_key=cache_result.profile.cache_key,
            dataset_fingerprint=dataset_fingerprint,
            field_registry_fingerprint=field_registry_fingerprint,
        )
        return persisted

    def _is_expired(self, entry: _CachedResearchContextEntry, *, now: float) -> bool:
        return self.ttl_seconds > 0 and (now - entry.cached_at) >= self.ttl_seconds

    def _resolve_cache_miss_reason(self, cache_key: ResearchContextCacheKey) -> str:
        if not self._entries:
            return "cold_start"
        if cache_key.cache_key in self._entries:
            return "ttl_expired"
        if self._last_cache_key and self._last_cache_key != cache_key.cache_key:
            return "input_changed"
        return "cache_miss"

    def _trim_entries(self) -> None:
        if len(self._entries) <= self.max_entries:
            return
        for key, _ in sorted(self._entries.items(), key=lambda item: item[1].last_accessed_at)[: len(self._entries) - self.max_entries]:
            self._entries.pop(key, None)


def _build_research_context(
    config: AppConfig,
    environment: CommandEnvironment,
    *,
    stage: str,
) -> CachedResearchContextResult:
    logger = get_logger(__name__, run_id=environment.context.run_id, stage=stage)
    started = time.perf_counter()
    bundle = load_dataset(config, environment, stage=stage)
    matrices = build_research_matrices(bundle.get_timeframe_data(config.backtest.timeframe))
    runtime_bundle = load_runtime_field_values(
        config.generation.field_value_paths,
        default_timeframe=config.backtest.timeframe,
    )
    runtime_numeric_fields, runtime_group_fields = runtime_bundle.for_timeframe(config.backtest.timeframe)
    aligned_numeric_fields, aligned_group_fields = _align_runtime_fields(
        matrices=matrices,
        numeric_fields=runtime_numeric_fields,
        group_fields=runtime_group_fields,
    )
    matrices = matrices.with_additional_fields(
        numeric_fields=aligned_numeric_fields,
        group_fields=aligned_group_fields,
    )
    memory_service = PatternMemoryService()
    region_learning_context = memory_service.build_learning_context(bundle.fingerprint, config)
    regime_snapshot = RegimeService().resolve(
        matrices=matrices,
        config=config.adaptive_generation.regime_detection,
        region=region_learning_context.region,
        legacy_regime_key=region_learning_context.regime_key,
        global_regime_key=region_learning_context.global_regime_key,
    )
    score_weights = FieldScoreWeights(
        coverage=float(config.generation.field_score_weights.get("coverage", 0.50)),
        usage=float(config.generation.field_score_weights.get("usage", 0.30)),
        category=float(config.generation.field_score_weights.get("category", 0.20)),
    )
    field_registry_started = time.perf_counter()
    field_registry = build_field_registry(
        catalog_paths=config.generation.field_catalog_paths,
        runtime_numeric_fields=matrices.numeric_fields,
        runtime_group_fields=matrices.group_fields,
        category_weights=config.generation.category_weights,
        score_weights=score_weights,
        preferred_region=config.brain.region,
        preferred_universe=config.brain.universe,
        preferred_delay=config.brain.delay,
    )
    build_field_registry_ms = (time.perf_counter() - field_registry_started) * 1000.0
    logger.info(
        "Resolved field registry summary: total=%s runtime_numeric=%s runtime_group=%s catalog_only=%s catalog_generation=%s",
        len(field_registry.fields),
        len(field_registry.runtime_numeric_fields()),
        len(field_registry.runtime_group_fields()),
        sum(1 for spec in field_registry.fields.values() if not spec.runtime_available),
        config.generation.allow_catalog_fields_without_runtime,
    )
    profile = ResearchContextLoadProfile(
        cache_hit=False,
        cache_reason="rebuilt",
        load_research_context_ms=(time.perf_counter() - started) * 1000.0,
        build_field_registry_ms=build_field_registry_ms,
    )
    return CachedResearchContextResult(
        research_context=ResearchContext(
            bundle=bundle,
            matrices=matrices,
            region=region_learning_context.region,
            regime_key=region_learning_context.regime_key,
            global_regime_key=region_learning_context.global_regime_key,
            region_learning_context=region_learning_context,
            memory_service=memory_service,
            field_registry=field_registry,
            legacy_regime_key=region_learning_context.regime_key,
            market_regime_key=regime_snapshot.market_regime_key,
            effective_regime_key=regime_snapshot.effective_regime_key or region_learning_context.regime_key,
            regime_label=regime_snapshot.regime_label,
            regime_confidence=regime_snapshot.confidence,
            regime_features=dict(regime_snapshot.features),
        ),
        profile=profile,
    )


def persist_research_metadata(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    research_context: ResearchContext,
    *,
    round_index: int = 0,
    persist_dataset_summary: bool = True,
    persist_field_catalog: bool = True,
    persist_run_field_scores: bool = True,
    removed_field_names: tuple[str, ...] = (),
) -> dict[str, bool]:
    """Persist dataset summary and research context metadata for the active run."""
    if persist_dataset_summary:
        repository.save_dataset_summary(
            environment.context.run_id,
            research_context.bundle.summary(),
            dataset_fingerprint=research_context.bundle.fingerprint,
            selected_timeframe=config.backtest.timeframe,
            regime_key=research_context.regime_key,
            global_regime_key=research_context.global_regime_key,
            market_regime_key=research_context.market_regime_key,
            effective_regime_key=research_context.effective_regime_key,
            regime_label=research_context.regime_label,
            regime_confidence=research_context.regime_confidence,
            region=research_context.region,
        )
    timestamp = datetime.now(UTC).isoformat()
    repository.save_regime_snapshots(
        [
            RegimeSnapshotRecord(
                run_id=environment.context.run_id,
                round_index=round_index,
                region=research_context.region,
                legacy_regime_key=research_context.legacy_regime_key or research_context.regime_key,
                global_regime_key=research_context.global_regime_key,
                market_regime_key=research_context.market_regime_key,
                effective_regime_key=research_context.effective_regime_key or research_context.regime_key,
                regime_label=research_context.regime_label,
                confidence=research_context.regime_confidence,
                features_json=json.dumps(research_context.regime_features, sort_keys=True),
                created_at=timestamp,
            )
        ]
    )
    if removed_field_names:
        repository.delete_field_metadata(
            removed_field_names,
            run_id=environment.context.run_id,
        )
    if persist_field_catalog:
        repository.save_field_catalog(
            [
                FieldCatalogRecord(
                    field_name=spec.name,
                    dataset=spec.dataset,
                    field_type=spec.field_type,
                    coverage=spec.coverage,
                    alpha_usage_count=spec.alpha_usage_count,
                    category=spec.category,
                    delay=spec.delay,
                    region=spec.region,
                    universe=spec.universe,
                    runtime_available=spec.runtime_available,
                    description=spec.description,
                    subcategory=spec.subcategory,
                    user_count=spec.user_count,
                    category_weight=spec.category_weight,
                    field_score=spec.field_score,
                    updated_at=timestamp,
                )
                for spec in research_context.field_registry.fields.values()
            ]
        )
    if persist_run_field_scores:
        repository.replace_run_field_scores(
            environment.context.run_id,
            [
                RunFieldScoreRecord(
                    run_id=environment.context.run_id,
                    field_name=spec.name,
                    runtime_available=spec.runtime_available,
                    field_type=spec.field_type,
                    category=spec.category,
                    field_score=spec.field_score,
                    coverage=spec.coverage,
                    alpha_usage_count=spec.alpha_usage_count,
                    created_at=timestamp,
                )
                for spec in research_context.field_registry.fields.values()
            ],
        )
    return {
        "dataset_summary_persisted": persist_dataset_summary,
        "field_catalog_persisted": persist_field_catalog,
        "run_field_scores_persisted": persist_run_field_scores,
    }


def load_and_persist_dataset(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    stage: str = "load-data",
):
    """Load the configured dataset and persist its summary."""
    bundle = load_dataset(config, environment, stage=stage)
    learning_context = PatternMemoryService().build_learning_context(bundle.fingerprint, config)
    repository.save_dataset_summary(
        environment.context.run_id,
        bundle.summary(),
        dataset_fingerprint=bundle.fingerprint,
        selected_timeframe=config.backtest.timeframe,
        regime_key=learning_context.regime_key,
        global_regime_key=learning_context.global_regime_key,
        region=learning_context.region,
    )
    return bundle


def resolve_regime_key(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    stage: str,
) -> str:
    """Resolve the active regime key, preferring persisted run metadata when present."""
    run = repository.get_run(environment.context.run_id)
    if run and run.regime_key:
        return run.regime_key
    research_context = load_research_context(config, environment, stage=stage)
    sanitized_context, blocked_fields = sanitize_generation_research_context(
        repository,
        config,
        research_context,
        environment,
        stage=stage,
    )
    persist_research_metadata(
        repository,
        config,
        environment,
        sanitized_context,
        removed_field_names=blocked_fields,
    )
    return research_context.regime_key


def resolve_region_learning_context(
    repository: SQLiteRepository,
    config: AppConfig,
    environment: CommandEnvironment,
    stage: str,
) -> RegionLearningContext:
    run = repository.get_run(environment.context.run_id)
    if run and run.regime_key and run.global_regime_key:
        return RegionLearningContext(
            region=str(run.region or ""),
            regime_key=str(run.regime_key),
            global_regime_key=str(run.global_regime_key),
        )
    research_context = load_research_context(config, environment, stage=stage)
    sanitized_context, blocked_fields = sanitize_generation_research_context(
        repository,
        config,
        research_context,
        environment,
        stage=stage,
    )
    persist_research_metadata(
        repository,
        config,
        environment,
        sanitized_context,
        removed_field_names=blocked_fields,
    )
    return research_context.region_learning_context


def slice_frame_by_period(frame: pd.DataFrame, period: PeriodConfig) -> pd.DataFrame:
    """Slice a dataframe by inclusive period boundaries."""
    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end)
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def slice_series_by_period(series: pd.Series, period: PeriodConfig) -> pd.Series:
    """Slice a series by inclusive period boundaries."""
    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end)
    return series.loc[(series.index >= start) & (series.index <= end)]


def combine_series_for_periods(series: pd.Series, *periods: PeriodConfig) -> pd.Series:
    """Concatenate series slices for several periods."""
    pieces = [slice_series_by_period(series, period) for period in periods]
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


def resolve_field_registry(
    config: AppConfig,
    research_context: ResearchContext,
) -> FieldRegistry:
    """Return the active field registry for generation and validation."""
    allowed = set(config.generation.allowed_fields or [])
    if allowed:
        filtered = {
            name: spec
            for name, spec in research_context.field_registry.fields.items()
            if name in allowed or spec.operator_type == "group" or not spec.runtime_available
        }
        return FieldRegistry(fields=filtered)
    return research_context.field_registry


def resolve_generation_field_registry(
    repository: SQLiteRepository,
    config: AppConfig,
    research_context: ResearchContext,
    environment: CommandEnvironment,
    *,
    stage: str,
) -> FieldRegistry:
    sanitized_context, _ = sanitize_generation_research_context(
        repository,
        config,
        research_context,
        environment,
        stage=stage,
    )
    return resolve_field_registry(config, sanitized_context)


def list_invalid_generation_fields(
    repository: SQLiteRepository,
    config: AppConfig,
) -> set[str]:
    return repository.brain_results.list_invalid_generation_fields(
        region=config.brain.region,
        universe=config.brain.universe,
        delay=config.brain.delay,
    )


def sanitize_generation_research_context(
    repository: SQLiteRepository,
    config: AppConfig,
    research_context: ResearchContext,
    environment: CommandEnvironment,
    *,
    stage: str,
) -> tuple[ResearchContext, tuple[str, ...]]:
    invalid_fields = list_invalid_generation_fields(repository, config)
    filtered_registry, blocked_fields = filter_generation_field_registry(
        research_context.field_registry,
        blocked_fields=invalid_fields,
    )
    if not blocked_fields:
        return research_context, ()
    logger = get_logger(__name__, run_id=environment.context.run_id, stage=stage)
    logger.info(
        "Pruned invalid fields from source field registry: blocked=%s sample=%s",
        len(blocked_fields),
        list(blocked_fields[:8]),
    )
    sanitized_context = ResearchContext(
        bundle=research_context.bundle,
        matrices=research_context.matrices,
        region=research_context.region,
        regime_key=research_context.regime_key,
        global_regime_key=research_context.global_regime_key,
        region_learning_context=research_context.region_learning_context,
        memory_service=research_context.memory_service,
        field_registry=filtered_registry,
        legacy_regime_key=research_context.legacy_regime_key,
        market_regime_key=research_context.market_regime_key,
        effective_regime_key=research_context.effective_regime_key,
        regime_label=research_context.regime_label,
        regime_confidence=research_context.regime_confidence,
        regime_features=dict(research_context.regime_features),
    )
    generation_registry = resolve_field_registry(config, sanitized_context)
    numeric_fields = generation_registry.generation_numeric_fields(
        config.generation.allowed_fields,
        include_catalog_fields=config.generation.allow_catalog_fields_without_runtime,
    )
    if numeric_fields:
        return sanitized_context, blocked_fields
    profile = (
        f"region={config.brain.region or '-'} "
        f"universe={config.brain.universe or '-'} "
        f"delay={config.brain.delay}"
    )
    raise ValueError(
        "Generation blacklist removed all numeric fields for BRAIN profile "
        f"{profile}. blocked_count={len(blocked_fields)}"
    )


def filter_generation_field_registry(
    field_registry: FieldRegistry,
    *,
    blocked_fields: set[str],
) -> tuple[FieldRegistry, tuple[str, ...]]:
    if not blocked_fields:
        return field_registry, ()
    blocked = tuple(sorted(name for name in field_registry.fields if name in blocked_fields))
    if not blocked:
        return field_registry, ()
    filtered = {
        name: spec
        for name, spec in field_registry.fields.items()
        if name not in blocked_fields
    }
    return FieldRegistry(fields=filtered), blocked


def _align_runtime_fields(
    matrices,
    numeric_fields: dict[str, pd.DataFrame],
    group_fields: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    close = matrices.numeric_fields["close"]
    index = close.index
    columns = close.columns
    aligned_numeric = {
        name: frame.reindex(index=index, columns=columns)
        for name, frame in numeric_fields.items()
    }
    aligned_groups = {
        name: frame.reindex(index=index, columns=columns)
        for name, frame in group_fields.items()
    }
    return aligned_numeric, aligned_groups


def _build_research_context_cache_key(config: AppConfig) -> ResearchContextCacheKey:
    config_payload = {
        "data": asdict(config.data),
        "aux_data": asdict(config.aux_data),
        "generation": {
            "field_catalog_paths": list(config.generation.field_catalog_paths),
            "field_value_paths": list(config.generation.field_value_paths),
            "field_score_weights": dict(config.generation.field_score_weights),
            "category_weights": dict(config.generation.category_weights),
        },
        "backtest": {"timeframe": config.backtest.timeframe},
        "brain": {
            "region": config.brain.region,
            "universe": config.brain.universe,
            "delay": config.brain.delay,
        },
    }
    config_fingerprint = _digest_payload(config_payload)
    input_fingerprint = _digest_payload(_collect_input_snapshots(config))
    return ResearchContextCacheKey(
        cache_key=_digest_payload(
            {
                "config_fingerprint": config_fingerprint,
                "input_fingerprint": input_fingerprint,
            }
        ),
        config_fingerprint=config_fingerprint,
        input_fingerprint=input_fingerprint,
    )


def _collect_input_snapshots(config: AppConfig) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    paths = [
        config.data.path,
        config.aux_data.group_path,
        config.aux_data.factor_path,
        config.aux_data.mask_path,
        *config.generation.field_catalog_paths,
        *config.generation.field_value_paths,
    ]
    for raw_path in paths:
        if not raw_path:
            continue
        for path in _expand_snapshot_path(str(raw_path)):
            snapshots.append(_snapshot_path(path))
    return snapshots


def _expand_snapshot_path(raw_path: str) -> list[Path]:
    path = resolve_path(raw_path)
    if not path.exists():
        return [path]
    if path.is_file():
        return [path]
    files = sorted(
        [child for child in path.iterdir() if child.is_file() and child.suffix.lower() in {".csv", ".json"}]
    )
    return files or [path]


def _snapshot_path(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "is_dir": path.is_dir(),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _digest_payload(payload: object) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _fingerprint_field_registry(field_registry: FieldRegistry) -> str:
    payload = [
        {
            "name": spec.name,
            "dataset": spec.dataset,
            "field_type": spec.field_type,
            "coverage": spec.coverage,
            "alpha_usage_count": spec.alpha_usage_count,
            "category": spec.category,
            "delay": spec.delay,
            "region": spec.region,
            "universe": spec.universe,
            "runtime_available": spec.runtime_available,
            "description": spec.description,
            "subcategory": spec.subcategory,
            "user_count": spec.user_count,
            "category_weight": spec.category_weight,
            "field_score": spec.field_score,
        }
        for spec in sorted(field_registry.fields.values(), key=lambda item: item.name)
    ]
    return _digest_payload(payload)
