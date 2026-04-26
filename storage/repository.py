from __future__ import annotations

from typing import TYPE_CHECKING

from memory.pattern_memory import PatternMemoryService
from storage.alpha_history import AlphaHistoryStore
from storage.brain_result_store import BrainResultStore
from storage.repositories import (
    AlphaRepository,
    CrowdingRepository,
    DuplicateRepository,
    FieldRepository,
    MetricRepository,
    MutationRepository,
    QualityRepository,
    RecipeRepository,
    RegimeRepository,
    RunRepository,
    SelectionRepository,
    SimulationRepository,
)
from storage.repositories.quality_repository import (
    _optional_int,
    _quality_polish_signature,
    _quality_polish_transform_group,
)
from storage.service_dispatch_queue_store import ServiceDispatchQueueStore
from storage.service_runtime_store import ServiceRuntimeStore
from storage.sqlite import connect_sqlite
from storage.submission_store import SubmissionStore

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from domain.candidate import AlphaCandidate
    from storage.models import (
        AlphaRecord,
        CrowdingScoreRecord,
        DuplicateDecisionRecord,
        FieldCatalogRecord,
        MetricRecord,
        MutationOutcomeRecord,
        RegimeSnapshotRecord,
        RunFieldScoreRecord,
        RunRecord,
        SelectionRecord,
        SelectionScoreRecord,
        SimulationCacheRecord,
        StageMetricRecord,
        SubmissionTestRecord,
    )

__all__ = [
    "SQLiteRepository",
    "_optional_int",
    "_quality_polish_signature",
    "_quality_polish_transform_group",
]


class SQLiteRepository:
    def __init__(self, path: str) -> None:
        self.connection = connect_sqlite(path)
        self._memory_service = PatternMemoryService()
        self.alpha_history = AlphaHistoryStore(self.connection, self._memory_service)
        self.submissions = SubmissionStore(self.connection)
        self.brain_results = BrainResultStore(self.connection)
        self.service_runtime = ServiceRuntimeStore(self.connection)
        self.service_dispatch_queue = ServiceDispatchQueueStore(self.connection)
        self.runs = RunRepository(self.connection)
        self.alphas = AlphaRepository(self.connection, self._memory_service)
        self.fields = FieldRepository(self.connection)
        self.metrics = MetricRepository(self.connection)
        self.selections = SelectionRepository(self.connection)
        self.duplicates = DuplicateRepository(self.connection)
        self.crowding = CrowdingRepository(self.connection)
        self.regimes = RegimeRepository(self.connection)
        self.mutations = MutationRepository(self.connection)
        self.recipes = RecipeRepository(self.connection, self.alphas)
        self.quality = QualityRepository(self.connection, self.alphas)
        self.simulations = SimulationRepository(self.connection)

    def close(self) -> None:
        self.connection.close()

    def delete_field_metadata(self, field_names: Iterable[str], *, run_id: str | None = None) -> None:
        return self.fields.delete_field_metadata(field_names, run_id=run_id)

    def upsert_run(
        self,
        run_id: str,
        seed: int,
        config_path: str,
        config_snapshot: str,
        status: str,
        started_at: str,
        profile_name: str = "",
        selected_timeframe: str = "",
        global_regime_key: str = "",
        region: str = "",
        entry_command: str = "",
    ) -> None:
        return self.runs.upsert_run(run_id, seed, config_path, config_snapshot, status, started_at, profile_name, selected_timeframe, global_regime_key, region, entry_command)

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.runs.get_run(run_id)

    def get_latest_run(self) -> RunRecord | None:
        return self.runs.get_latest_run()

    def update_run_status(self, run_id: str, status: str, finished: bool = False) -> None:
        return self.runs.update_run_status(run_id, status, finished)

    def save_dataset_summary(
        self,
        run_id: str,
        summary: dict,
        dataset_fingerprint: str | None = None,
        selected_timeframe: str | None = None,
        regime_key: str | None = None,
        global_regime_key: str | None = None,
        market_regime_key: str | None = None,
        effective_regime_key: str | None = None,
        regime_label: str | None = None,
        regime_confidence: float | None = None,
        region: str | None = None,
    ) -> None:
        return self.runs.save_dataset_summary(run_id, summary, dataset_fingerprint, selected_timeframe, regime_key, global_regime_key, market_regime_key, effective_regime_key, regime_label, regime_confidence, region)

    def save_alpha_candidates(self, run_id: str, candidates: list[AlphaCandidate]) -> int:
        return self.alphas.save_alpha_candidates(run_id, candidates)

    def _candidate_structural_signature_json(self, candidate: AlphaCandidate) -> str:
        return self.alphas._candidate_structural_signature_json(candidate)

    def list_alpha_records(self, run_id: str) -> list[AlphaRecord]:
        return self.alphas.list_alpha_records(run_id)

    def _list_recent_completed_parent_rows(
        self,
        *,
        run_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return self.alphas._list_recent_completed_parent_rows(run_id=run_id, limit=limit)

    def list_quality_polish_parent_rows(
        self,
        *,
        run_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return self.quality.list_quality_polish_parent_rows(run_id=run_id, limit=limit)

    def list_recipe_parent_rows(
        self,
        *,
        run_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return self.recipes.list_recipe_parent_rows(run_id=run_id, limit=limit)

    def list_quality_polish_usage_keys(self, run_id: str) -> dict[str, Any]:
        return self.quality.list_quality_polish_usage_keys(run_id)

    def list_recent_recipe_guided_usage_rows(
        self,
        *,
        run_id: str,
        before_round_index: int,
        lookback_rounds: int,
    ) -> list[dict[str, Any]]:
        return self.recipes.list_recent_recipe_guided_usage_rows(run_id=run_id, before_round_index=before_round_index, lookback_rounds=lookback_rounds)

    def get_latest_brain_quality_score(self, run_id: str, alpha_id: str) -> float | None:
        return self.metrics.get_latest_brain_quality_score(run_id, alpha_id)

    def list_recipe_bucket_result_rows(
        self,
        *,
        run_id: str,
        before_round_index: int,
        lookback_rounds: int,
    ) -> list[dict[str, Any]]:
        return self.recipes.list_recipe_bucket_result_rows(run_id=run_id, before_round_index=before_round_index, lookback_rounds=lookback_rounds)

    def list_generation_result_rows(
        self,
        *,
        run_id: str,
        before_round_index: int,
        lookback_rounds: int,
    ) -> list[dict[str, Any]]:
        return self.alphas.list_generation_result_rows(run_id=run_id, before_round_index=before_round_index, lookback_rounds=lookback_rounds)

    def get_alpha_reference_marker(self, run_id: str) -> tuple[int, str]:
        return self.alphas.get_alpha_reference_marker(run_id)

    def get_existing_alpha_ids_by_normalized(
        self,
        run_id: str,
        normalized_expressions: Iterable[str],
        *,
        exclude_alpha_ids: Iterable[str] = (),
    ) -> dict[str, str]:
        return self.alphas.get_existing_alpha_ids_by_normalized(run_id, normalized_expressions, exclude_alpha_ids=exclude_alpha_ids)

    def get_same_run_structural_references(
        self,
        *,
        run_id: str,
        limit: int,
    ) -> list[dict]:
        return self.alphas.get_same_run_structural_references(run_id=run_id, limit=limit)

    def update_alpha_structural_signature(
        self,
        *,
        run_id: str,
        alpha_id: str,
        structural_signature_json: str,
    ) -> None:
        return self.alphas.update_alpha_structural_signature(run_id=run_id, alpha_id=alpha_id, structural_signature_json=structural_signature_json)

    def save_field_catalog(self, records: list[FieldCatalogRecord]) -> None:
        return self.fields.save_field_catalog(records)

    def replace_run_field_scores(self, run_id: str, records: list[RunFieldScoreRecord]) -> None:
        return self.fields.replace_run_field_scores(run_id, records)

    def list_existing_normalized_expressions(self, run_id: str) -> set[str]:
        return self.alphas.list_existing_normalized_expressions(run_id)

    def list_run_field_scores(self, run_id: str) -> dict[str, float]:
        return self.fields.list_run_field_scores(run_id)

    def list_run_field_scores_for_runs(self, run_ids: Iterable[str]) -> dict[str, dict[str, float]]:
        return self.fields.list_run_field_scores_for_runs(run_ids)

    def list_meta_model_training_rows(
        self,
        *,
        run_id: str,
        round_index: int,
        lookback_rounds: int,
        use_cross_run_history: bool,
    ) -> list[dict[str, Any]]:
        return self.selections.list_meta_model_training_rows(run_id=run_id, round_index=round_index, lookback_rounds=lookback_rounds, use_cross_run_history=use_cross_run_history)

    def save_metrics(self, metric_records: list[MetricRecord]) -> None:
        return self.metrics.save_metrics(metric_records)

    def replace_submission_tests(self, run_id: str, alpha_id: str, records: list[SubmissionTestRecord]) -> None:
        return self.metrics.replace_submission_tests(run_id, alpha_id, records)

    def replace_selections(self, run_id: str, records: list[SelectionRecord]) -> None:
        return self.selections.replace_selections(run_id, records)

    def save_simulation_cache(self, record: SimulationCacheRecord) -> None:
        return self.simulations.save_simulation_cache(record)

    def get_cached_simulation(self, simulation_signature: str) -> SimulationCacheRecord | None:
        return self.simulations.get_cached_simulation(simulation_signature)

    def get_top_alpha_records(self, run_id: str, limit: int) -> list[AlphaRecord]:
        return self.selections.get_top_alpha_records(run_id, limit)

    def get_parent_refs(self, run_id: str) -> dict[str, list[dict[str, str]]]:
        return self.alphas.get_parent_refs(run_id)

    def get_generation_mix(self, run_id: str) -> list[dict]:
        return self.alphas.get_generation_mix(run_id)

    def get_cross_run_duplicate_references(
        self,
        *,
        run_id: str,
        global_regime_key: str,
        limit: int,
    ) -> list[dict]:
        return self.duplicates.get_cross_run_duplicate_references(run_id=run_id, global_regime_key=global_regime_key, limit=limit)

    def save_duplicate_decisions(self, records: list[DuplicateDecisionRecord]) -> None:
        return self.duplicates.save_duplicate_decisions(records)

    def save_crowding_scores(self, records: list[CrowdingScoreRecord]) -> None:
        return self.crowding.save_crowding_scores(records)

    def save_stage_metrics(self, records: list[StageMetricRecord]) -> None:
        return self.metrics.save_stage_metrics(records)

    def save_selection_scores(self, records: list[SelectionScoreRecord]) -> None:
        return self.selections.save_selection_scores(records)

    def save_regime_snapshots(self, records: list[RegimeSnapshotRecord]) -> None:
        return self.regimes.save_regime_snapshots(records)

    def save_mutation_outcomes(self, records: list[MutationOutcomeRecord]) -> None:
        return self.mutations.save_mutation_outcomes(records)

    def get_stage_metrics(self, run_id: str) -> list[dict]:
        return self.metrics.get_stage_metrics(run_id)

    def list_recent_generation_stage_metrics(
        self,
        run_id: str,
        *,
        limit: int,
        before_round_index: int | None = None,
    ) -> list[dict]:
        return self.metrics.list_recent_generation_stage_metrics(run_id, limit=limit, before_round_index=before_round_index)

    def get_duplicate_decision_summary(self, run_id: str) -> list[dict]:
        return self.duplicates.get_duplicate_decision_summary(run_id)

    def get_average_crowding_penalty(self, run_id: str) -> float:
        return self.crowding.get_average_crowding_penalty(run_id)

    def list_selection_scores(self, run_id: str, *, score_stage: str | None = None) -> list[dict]:
        return self.selections.list_selection_scores(run_id, score_stage=score_stage)

    def get_latest_regime_snapshot(self, run_id: str) -> dict | None:
        return self.regimes.get_latest_regime_snapshot(run_id)

    def list_mutation_outcomes(
        self,
        *,
        effective_regime_key: str | None = None,
        family_signature: str | None = None,
    ) -> list[dict]:
        return self.mutations.list_mutation_outcomes(effective_regime_key=effective_regime_key, family_signature=family_signature)

    def list_mutation_outcomes_with_motif(
        self,
        *,
        effective_regime_key: str | None = None,
        limit: int = 2000,
    ) -> list[dict]:
        return self.mutations.list_mutation_outcomes_with_motif(effective_regime_key=effective_regime_key, limit=limit)

    def get_top_selections(self, run_id: str, limit: int) -> list[dict]:
        return self.selections.get_top_selections(run_id, limit)

    def get_submission_tests_for_run(self, run_id: str) -> list[dict]:
        return self.metrics.get_submission_tests_for_run(run_id)

    def get_cache_stats(self, run_id: str) -> dict[str, int]:
        return self.metrics.get_cache_stats(run_id)

    def get_validation_metric_rows(self, run_id: str) -> list[dict]:
        return self.metrics.get_validation_metric_rows(run_id)

    @staticmethod
    def dataframe_to_json(frame: pd.DataFrame) -> str:
        return SimulationRepository.dataframe_to_json(frame)

    @staticmethod
    def dataframe_from_json(payload: str) -> pd.DataFrame:
        return SimulationRepository.dataframe_from_json(payload)

    @staticmethod
    def series_to_json(series: pd.Series) -> str:
        return SimulationRepository.series_to_json(series)

    @staticmethod
    def series_from_json(payload: str) -> pd.Series:
        return SimulationRepository.series_from_json(payload)
