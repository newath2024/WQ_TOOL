from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.config import MetaModelConfig
from domain.candidate import AlphaCandidate
from storage.repository import SQLiteRepository

try:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime fallback when deps are unavailable
    DictVectorizer = None
    LogisticRegression = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class MetaModelFeatureInput:
    alpha_id: str
    generation_mode: str
    template_name: str
    motif: str
    mutation_mode: str
    lineage_branch_key: str
    effective_regime_key: str
    primary_field_family: str
    field_families: tuple[str, ...]
    operator_path_key: str
    operator_path_head: str
    operator_path_depth: int
    complexity: int
    depth: int
    complexity_bucket: str
    horizon_bucket: str
    turnover_bucket: str
    field_score: float
    novelty_score: float
    family_diversity: float
    duplicate_risk: float
    crowding_penalty: float
    regime_fit: float
    heuristic_predicted_quality: float


@dataclass(frozen=True, slots=True)
class MetaModelPrediction:
    alpha_id: str
    heuristic_predicted_quality: float
    ml_positive_outcome_prob: float
    blended_predicted_quality: float
    train_rows: int
    positive_rows: int
    used: bool


@dataclass(frozen=True, slots=True)
class _TrainingExample:
    features: MetaModelFeatureInput
    label: int


class MetaModelService:
    def __init__(
        self,
        repository: SQLiteRepository | None,
        *,
        config: MetaModelConfig | None = None,
    ) -> None:
        self.repository = repository
        self.config = config or MetaModelConfig()

    def score_candidates(
        self,
        *,
        run_id: str,
        round_index: int,
        effective_regime_key: str,
        feature_inputs: list[MetaModelFeatureInput],
    ) -> dict[str, MetaModelPrediction]:
        fallback = self._fallback_predictions(feature_inputs)
        if (
            not self.config.enabled
            or self.repository is None
            or not SKLEARN_AVAILABLE
            or not feature_inputs
        ):
            return fallback

        training_rows = self.repository.list_meta_model_training_rows(
            run_id=run_id,
            round_index=round_index,
            lookback_rounds=self.config.lookback_rounds,
            use_cross_run_history=self.config.use_cross_run_history,
        )
        if not training_rows:
            return fallback
        field_scores_by_run = self.repository.list_run_field_scores_for_runs(row["run_id"] for row in training_rows)
        training_examples = [
            example
            for row in training_rows
            if (example := self._training_example_from_row(row, field_scores_by_run.get(str(row["run_id"]), {}))) is not None
        ]
        if len(training_examples) < self.config.min_train_rows:
            return fallback
        positive_rows = sum(example.label for example in training_examples)
        if positive_rows < self.config.min_positive_rows:
            return fallback

        prior_stats = self._build_prior_stats(training_examples)
        train_dicts = [
            self._feature_dict(example.features, prior_stats=prior_stats, label=example.label, leave_one_out=True)
            for example in training_examples
        ]
        labels = np.asarray([example.label for example in training_examples], dtype=float)
        if int(labels.sum()) == 0 or int(labels.sum()) == len(labels):
            return fallback

        vectorizer = DictVectorizer(sparse=True)
        train_matrix = vectorizer.fit_transform(train_dicts)
        selected_mask = self._selected_feature_mask(train_matrix)
        train_matrix = train_matrix[:, selected_mask]
        scaler = StandardScaler(with_mean=False)
        scaled_train = scaler.fit_transform(train_matrix)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=400,
            random_state=7,
            solver="liblinear",
        )
        try:
            model.fit(scaled_train, labels)
        except ValueError:
            return fallback

        infer_inputs = [
            self._with_effective_regime_key(feature_input, effective_regime_key)
            for feature_input in feature_inputs
        ]
        infer_dicts = [
            self._feature_dict(feature_input, prior_stats=prior_stats, label=None, leave_one_out=False)
            for feature_input in infer_inputs
        ]
        infer_matrix = vectorizer.transform(infer_dicts)[:, selected_mask]
        scaled_infer = scaler.transform(infer_matrix)
        probabilities = model.predict_proba(scaled_infer)[:, 1]
        results: dict[str, MetaModelPrediction] = {}
        for feature_input, probability in zip(infer_inputs, probabilities, strict=True):
            heuristic = float(feature_input.heuristic_predicted_quality)
            probability_value = float(max(0.0, min(1.0, float(probability))))
            blended = (1.0 - float(self.config.blend_weight)) * heuristic + float(self.config.blend_weight) * probability_value
            results[feature_input.alpha_id] = MetaModelPrediction(
                alpha_id=feature_input.alpha_id,
                heuristic_predicted_quality=heuristic,
                ml_positive_outcome_prob=probability_value,
                blended_predicted_quality=float(blended),
                train_rows=len(training_examples),
                positive_rows=positive_rows,
                used=True,
            )
        return results

    def _training_example_from_row(
        self,
        row: dict[str, Any],
        field_scores: dict[str, float],
    ) -> _TrainingExample | None:
        breakdown = self._decode_json_object(row.get("breakdown_json"))
        components = self._decode_json_object(breakdown.get("components"))
        heuristic_predicted_quality = float(
            components.get("heuristic_predicted_quality", components.get("predicted_quality", 0.0)) or 0.0
        )
        generation_metadata = self._decode_json_object(row.get("generation_metadata"))
        structural = self._decode_json_object(generation_metadata.get("canonical_structural_signature"))
        field_families = self._decode_text_list(row.get("field_families_json")) or self._decode_text_list(
            generation_metadata.get("field_families")
        )
        operator_path = self._decode_text_list(row.get("operator_path_json")) or self._decode_text_list(
            generation_metadata.get("operator_path")
        )
        complexity_bucket = str(row.get("complexity_bucket") or structural.get("complexity_bucket") or "")
        horizon_bucket = str(row.get("horizon_bucket") or structural.get("horizon_bucket") or "")
        turnover_bucket = str(row.get("turnover_bucket") or structural.get("turnover_bucket") or "")
        fields_used = self._decode_text_list(row.get("fields_used_json"))
        feature_input = MetaModelFeatureInput(
            alpha_id=str(row.get("alpha_id") or ""),
            generation_mode=str(row.get("generation_mode") or ""),
            template_name=str(row.get("template_name") or ""),
            motif=str(row.get("motif") or generation_metadata.get("motif") or row.get("template_name") or ""),
            mutation_mode=str(generation_metadata.get("mutation_mode") or ""),
            lineage_branch_key=str(generation_metadata.get("lineage_branch_key") or ""),
            effective_regime_key=str(row.get("effective_regime_key") or generation_metadata.get("regime_key") or ""),
            primary_field_family=field_families[0] if field_families else "none",
            field_families=tuple(field_families),
            operator_path_key=self._operator_path_key(operator_path),
            operator_path_head=operator_path[0] if operator_path else "none",
            operator_path_depth=len(operator_path),
            complexity=int(row.get("complexity") or structural.get("complexity") or 0),
            depth=int(row.get("depth") or structural.get("depth") or 0),
            complexity_bucket=complexity_bucket or "unknown",
            horizon_bucket=horizon_bucket or "unknown",
            turnover_bucket=turnover_bucket or "unknown",
            field_score=self._average_field_score(fields_used, field_scores),
            novelty_score=float(components.get("novelty", 0.0) or 0.0),
            family_diversity=float(components.get("family_diversity", 0.0) or 0.0),
            duplicate_risk=float(components.get("duplicate_risk", 0.0) or 0.0),
            crowding_penalty=float(components.get("crowding_penalty", 0.0) or 0.0),
            regime_fit=float(components.get("regime_fit", 0.0) or 0.0),
            heuristic_predicted_quality=heuristic_predicted_quality,
        )
        label = 1 if float(row.get("outcome_score") or 0.0) > 0.0 else 0
        if not feature_input.alpha_id:
            return None
        return _TrainingExample(features=feature_input, label=label)

    def _with_effective_regime_key(
        self,
        feature_input: MetaModelFeatureInput,
        effective_regime_key: str,
    ) -> MetaModelFeatureInput:
        if not effective_regime_key:
            return feature_input
        return MetaModelFeatureInput(
            alpha_id=feature_input.alpha_id,
            generation_mode=feature_input.generation_mode,
            template_name=feature_input.template_name,
            motif=feature_input.motif,
            mutation_mode=feature_input.mutation_mode,
            lineage_branch_key=feature_input.lineage_branch_key,
            effective_regime_key=effective_regime_key,
            primary_field_family=feature_input.primary_field_family,
            field_families=feature_input.field_families,
            operator_path_key=feature_input.operator_path_key,
            operator_path_head=feature_input.operator_path_head,
            operator_path_depth=feature_input.operator_path_depth,
            complexity=feature_input.complexity,
            depth=feature_input.depth,
            complexity_bucket=feature_input.complexity_bucket,
            horizon_bucket=feature_input.horizon_bucket,
            turnover_bucket=feature_input.turnover_bucket,
            field_score=feature_input.field_score,
            novelty_score=feature_input.novelty_score,
            family_diversity=feature_input.family_diversity,
            duplicate_risk=feature_input.duplicate_risk,
            crowding_penalty=feature_input.crowding_penalty,
            regime_fit=feature_input.regime_fit,
            heuristic_predicted_quality=feature_input.heuristic_predicted_quality,
        )

    def _build_prior_stats(self, training_examples: list[_TrainingExample]) -> dict[str, Any]:
        counters = {
            "motif": Counter[str](),
            "operator_path": Counter[str](),
            "field_family": Counter[str](),
            "effective_regime_key": Counter[str](),
        }
        positives = {
            "motif": Counter[str](),
            "operator_path": Counter[str](),
            "field_family": Counter[str](),
            "effective_regime_key": Counter[str](),
        }
        for example in training_examples:
            label = int(example.label)
            counters["motif"][example.features.motif] += 1
            counters["operator_path"][example.features.operator_path_key] += 1
            counters["field_family"][example.features.primary_field_family] += 1
            counters["effective_regime_key"][example.features.effective_regime_key] += 1
            positives["motif"][example.features.motif] += label
            positives["operator_path"][example.features.operator_path_key] += label
            positives["field_family"][example.features.primary_field_family] += label
            positives["effective_regime_key"][example.features.effective_regime_key] += label
        return {
            "counts": counters,
            "positives": positives,
            "global_positive_rate": float(sum(example.label for example in training_examples) / max(1, len(training_examples))),
        }

    def _feature_dict(
        self,
        feature_input: MetaModelFeatureInput,
        *,
        prior_stats: dict[str, Any],
        label: int | None,
        leave_one_out: bool,
    ) -> dict[str, float | str]:
        payload: dict[str, float | str] = {
            "generation_mode": feature_input.generation_mode or "unknown",
            "template_name": feature_input.template_name or "unknown",
            "motif": feature_input.motif or "unknown",
            "mutation_mode": feature_input.mutation_mode or "none",
            "lineage_branch_key": feature_input.lineage_branch_key or "none",
            "effective_regime_key": feature_input.effective_regime_key or "unknown",
            "primary_field_family": feature_input.primary_field_family or "none",
            "operator_path_key": feature_input.operator_path_key or "none",
            "operator_path_head": feature_input.operator_path_head or "none",
            "complexity_bucket": feature_input.complexity_bucket or "unknown",
            "horizon_bucket": feature_input.horizon_bucket or "unknown",
            "turnover_bucket": feature_input.turnover_bucket or "unknown",
            "field_score": float(feature_input.field_score),
            "novelty_score": float(feature_input.novelty_score),
            "family_diversity": float(feature_input.family_diversity),
            "duplicate_risk": float(feature_input.duplicate_risk),
            "crowding_penalty": float(feature_input.crowding_penalty),
            "regime_fit": float(feature_input.regime_fit),
            "heuristic_predicted_quality": float(feature_input.heuristic_predicted_quality),
            "complexity": float(feature_input.complexity),
            "depth": float(feature_input.depth),
            "operator_path_depth": float(feature_input.operator_path_depth),
            "motif_positive_prior": self._prior_value(
                prior_stats, "motif", feature_input.motif or "unknown", label=label, leave_one_out=leave_one_out
            ),
            "operator_path_positive_prior": self._prior_value(
                prior_stats,
                "operator_path",
                feature_input.operator_path_key or "none",
                label=label,
                leave_one_out=leave_one_out,
            ),
            "field_family_positive_prior": self._prior_value(
                prior_stats,
                "field_family",
                feature_input.primary_field_family or "none",
                label=label,
                leave_one_out=leave_one_out,
            ),
            "effective_regime_positive_prior": self._prior_value(
                prior_stats,
                "effective_regime_key",
                feature_input.effective_regime_key or "unknown",
                label=label,
                leave_one_out=leave_one_out,
            ),
        }
        for family in feature_input.field_families:
            payload[f"field_family::{family or 'none'}"] = 1.0
        return payload

    def _prior_value(
        self,
        prior_stats: dict[str, Any],
        category: str,
        key: str,
        *,
        label: int | None,
        leave_one_out: bool,
    ) -> float:
        counts = prior_stats["counts"][category]
        positives = prior_stats["positives"][category]
        total = int(counts.get(key, 0))
        positive = int(positives.get(key, 0))
        if leave_one_out and label is not None:
            total -= 1
            positive -= int(label)
        global_positive_rate = float(prior_stats["global_positive_rate"])
        if total <= 0:
            return global_positive_rate
        smoothing = 5.0
        return float((positive + smoothing * global_positive_rate) / (total + smoothing))

    def _selected_feature_mask(self, matrix) -> np.ndarray:
        if matrix.shape[1] == 0:
            return np.ones(0, dtype=bool)
        support = np.asarray((matrix != 0).sum(axis=0)).ravel()
        selected = support > 1
        if not selected.any():
            return np.ones(matrix.shape[1], dtype=bool)
        return selected

    def _fallback_predictions(
        self,
        feature_inputs: list[MetaModelFeatureInput],
    ) -> dict[str, MetaModelPrediction]:
        return {
            feature_input.alpha_id: MetaModelPrediction(
                alpha_id=feature_input.alpha_id,
                heuristic_predicted_quality=float(feature_input.heuristic_predicted_quality),
                ml_positive_outcome_prob=0.0,
                blended_predicted_quality=float(feature_input.heuristic_predicted_quality),
                train_rows=0,
                positive_rows=0,
                used=False,
            )
            for feature_input in feature_inputs
        }

    @staticmethod
    def feature_input_from_candidate(
        *,
        candidate: AlphaCandidate,
        structural_signature,
        effective_regime_key: str,
        field_score: float,
        novelty_score: float,
        family_diversity: float,
        duplicate_risk: float,
        crowding_penalty: float,
        regime_fit: float,
        heuristic_predicted_quality: float,
    ) -> MetaModelFeatureInput:
        generation_metadata = dict(candidate.generation_metadata or {})
        motif = str(generation_metadata.get("motif") or structural_signature.motif or candidate.template_name or "")
        field_families = tuple(dict.fromkeys(structural_signature.field_families or generation_metadata.get("field_families") or []))
        operator_path = tuple(structural_signature.operator_path or generation_metadata.get("operator_path") or [])
        return MetaModelFeatureInput(
            alpha_id=candidate.alpha_id,
            generation_mode=str(candidate.generation_mode or ""),
            template_name=str(candidate.template_name or ""),
            motif=motif or "unknown",
            mutation_mode=str(generation_metadata.get("mutation_mode") or ""),
            lineage_branch_key=str(generation_metadata.get("lineage_branch_key") or ""),
            effective_regime_key=str(effective_regime_key or generation_metadata.get("regime_key") or ""),
            primary_field_family=field_families[0] if field_families else "none",
            field_families=field_families,
            operator_path_key=MetaModelService._operator_path_key(operator_path),
            operator_path_head=operator_path[0] if operator_path else "none",
            operator_path_depth=len(operator_path),
            complexity=int(candidate.complexity),
            depth=int(candidate.depth),
            complexity_bucket=str(structural_signature.complexity_bucket or "unknown"),
            horizon_bucket=str(structural_signature.horizon_bucket or "unknown"),
            turnover_bucket=str(structural_signature.turnover_bucket or "unknown"),
            field_score=float(field_score),
            novelty_score=float(novelty_score),
            family_diversity=float(family_diversity),
            duplicate_risk=float(duplicate_risk),
            crowding_penalty=float(crowding_penalty),
            regime_fit=float(regime_fit),
            heuristic_predicted_quality=float(heuristic_predicted_quality),
        )

    @staticmethod
    def _average_field_score(fields_used: list[str], field_scores: dict[str, float]) -> float:
        if not fields_used:
            return 0.0
        values = [float(field_scores.get(field_name, 0.0)) for field_name in fields_used]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _decode_json_object(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str) and payload.strip():
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                return {}
            return decoded if isinstance(decoded, dict) else {}
        return {}

    @staticmethod
    def _decode_text_list(payload: Any) -> list[str]:
        if isinstance(payload, list):
            return [str(item) for item in payload if str(item)]
        if isinstance(payload, tuple):
            return [str(item) for item in payload if str(item)]
        if isinstance(payload, str) and payload.strip():
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                return []
            if isinstance(decoded, list):
                return [str(item) for item in decoded if str(item)]
        return []

    @staticmethod
    def _operator_path_key(operator_path: list[str] | tuple[str, ...]) -> str:
        normalized = [str(item) for item in operator_path[:4] if str(item)]
        return ">".join(normalized) if normalized else "none"
