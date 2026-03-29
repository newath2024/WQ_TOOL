from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


FIELD_TYPE_ALIASES = {
    "matrix": "matrix",
    "timeseries": "timeseries",
    "time_series": "timeseries",
    "ts": "timeseries",
    "vector": "vector",
    "group": "vector",
}

GROUP_FIELD_TYPES = {"vector", "group"}
NUMERIC_FIELD_TYPES = {"matrix", "timeseries"}


def normalize_field_type(value: Any) -> str:
    normalized = str(value or "matrix").strip().lower().replace("-", "_").replace(" ", "_")
    return FIELD_TYPE_ALIASES.get(normalized, normalized)


def normalize_category(value: Any) -> str:
    text = str(value or "other").strip().lower()
    slug = "".join(character if character.isalnum() else "_" for character in text)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "other"


@dataclass(frozen=True, slots=True)
class FieldScoreWeights:
    coverage: float = 0.50
    usage: float = 0.30
    category: float = 0.20


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    dataset: str
    field_type: str
    coverage: float
    alpha_usage_count: int
    category: str
    delay: int = 1
    region: str = ""
    universe: str = ""
    runtime_available: bool = False
    description: str = ""
    subcategory: str = ""
    user_count: int = 0
    category_weight: float = 0.50
    field_score: float = 0.0

    @property
    def operator_type(self) -> str:
        if self.field_type in GROUP_FIELD_TYPES:
            return "group"
        return "matrix"


@dataclass(slots=True)
class RuntimeFieldBundle:
    numeric_fields_by_timeframe: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    group_fields_by_timeframe: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    field_specs: dict[str, FieldSpec] = field(default_factory=dict)

    def for_timeframe(self, timeframe: str) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        return (
            dict(self.numeric_fields_by_timeframe.get(timeframe, {})),
            dict(self.group_fields_by_timeframe.get(timeframe, {})),
        )


@dataclass(slots=True)
class FieldRegistry:
    fields: dict[str, FieldSpec] = field(default_factory=dict)

    def get(self, name: str) -> FieldSpec:
        return self.fields[name]

    def contains(self, name: str) -> bool:
        return name in self.fields

    def runtime_numeric_fields(self, allowed: set[str] | None = None) -> list[FieldSpec]:
        return self._sorted_runtime_fields("matrix", allowed=allowed)

    def runtime_group_fields(self, allowed: set[str] | None = None) -> list[FieldSpec]:
        return self._sorted_runtime_fields("group", allowed=allowed)

    def field_types(self, allowed: set[str] | None = None) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for spec in self.fields.values():
            if allowed is not None and spec.name not in allowed:
                continue
            mapping[spec.name] = spec.operator_type
        return mapping

    def allowed_runtime_fields(self, allowed_fields: list[str]) -> set[str]:
        if allowed_fields:
            return {name for name in allowed_fields if name in self.fields and self.fields[name].runtime_available}
        return {name for name, spec in self.fields.items() if spec.runtime_available}

    def top_fields(self, limit: int, operator_type: str = "matrix", allowed: set[str] | None = None) -> list[FieldSpec]:
        return self._sorted_runtime_fields(operator_type, allowed=allowed)[:limit]

    def _sorted_runtime_fields(self, operator_type: str, allowed: set[str] | None = None) -> list[FieldSpec]:
        candidates = [
            spec
            for spec in self.fields.values()
            if spec.runtime_available and spec.operator_type == operator_type and (allowed is None or spec.name in allowed)
        ]
        return sorted(
            candidates,
            key=lambda item: (item.field_score, item.coverage, item.alpha_usage_count, item.name),
            reverse=True,
        )


def load_field_catalog(
    paths: list[str],
    category_weights: dict[str, float],
    score_weights: FieldScoreWeights,
) -> dict[str, FieldSpec]:
    if not paths:
        return {}
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            continue
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("rows", []) if isinstance(payload, dict) else payload
            if isinstance(rows, list):
                records.extend(item for item in rows if isinstance(item, dict))
            continue
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            records.extend(frame.to_dict("records"))

    if not records:
        return {}

    max_alpha_usage = max(int(record.get("alphaCount") or record.get("alpha_count") or 0) for record in records) or 1
    fields: dict[str, FieldSpec] = {}
    for record in records:
        spec = _field_spec_from_catalog_row(record, max_alpha_usage=max_alpha_usage, category_weights=category_weights, score_weights=score_weights)
        existing = fields.get(spec.name)
        if existing is None or _field_rank_tuple(spec) > _field_rank_tuple(existing):
            fields[spec.name] = spec
    return fields


def load_runtime_field_values(
    paths: list[str],
    default_timeframe: str,
) -> RuntimeFieldBundle:
    if not paths:
        return RuntimeFieldBundle()

    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        required = {"timestamp", "symbol", "field", "value"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Field value file '{path}' is missing required columns: {sorted(missing)}")
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False, errors="raise")
        frame["symbol"] = frame["symbol"].astype(str)
        frame["field"] = frame["field"].astype(str)
        if "timeframe" not in frame.columns:
            frame["timeframe"] = default_timeframe
        else:
            frame["timeframe"] = frame["timeframe"].astype(str)
        for column in ("dataset", "field_type", "category", "description", "subcategory", "region", "universe", "delay"):
            if column not in frame.columns:
                frame[column] = None
        duplicate_mask = frame.duplicated(subset=["timeframe", "timestamp", "symbol", "field"], keep=False)
        if duplicate_mask.any():
            duplicates = frame.loc[duplicate_mask, ["timeframe", "timestamp", "symbol", "field"]].head(10).to_dict("records")
            raise ValueError(f"Duplicate field value rows detected: {duplicates}")
        frames.append(frame)

    if not frames:
        return RuntimeFieldBundle()

    combined = pd.concat(frames, ignore_index=True)
    numeric_fields_by_timeframe: dict[str, dict[str, pd.DataFrame]] = {}
    group_fields_by_timeframe: dict[str, dict[str, pd.DataFrame]] = {}
    field_specs: dict[str, FieldSpec] = {}

    for field_name, field_frame in combined.groupby("field", sort=True):
        inferred_type = _infer_runtime_field_type(field_frame)
        spec = _field_spec_from_runtime_values(field_name, field_frame, inferred_type)
        field_specs[field_name] = spec
        for timeframe, timeframe_frame in field_frame.groupby("timeframe", sort=True):
            pivot = timeframe_frame.pivot(index="timestamp", columns="symbol", values="value").sort_index()
            if inferred_type in NUMERIC_FIELD_TYPES:
                pivot = pivot.apply(pd.to_numeric, errors="coerce")
                numeric_fields_by_timeframe.setdefault(str(timeframe), {})[field_name] = pivot.astype(float)
            else:
                group_fields_by_timeframe.setdefault(str(timeframe), {})[field_name] = pivot.where(pivot.notna())

    return RuntimeFieldBundle(
        numeric_fields_by_timeframe=numeric_fields_by_timeframe,
        group_fields_by_timeframe=group_fields_by_timeframe,
        field_specs=field_specs,
    )


def build_field_registry(
    *,
    catalog_paths: list[str],
    runtime_numeric_fields: dict[str, pd.DataFrame],
    runtime_group_fields: dict[str, pd.DataFrame],
    category_weights: dict[str, float],
    score_weights: FieldScoreWeights,
) -> FieldRegistry:
    catalog_fields = load_field_catalog(catalog_paths, category_weights=category_weights, score_weights=score_weights)
    runtime_fields = _runtime_specs_from_matrices(
        numeric_fields=runtime_numeric_fields,
        group_fields=runtime_group_fields,
        category_weights=category_weights,
        score_weights=score_weights,
    )
    merged = dict(catalog_fields)

    for name, runtime_spec in runtime_fields.items():
        existing = merged.get(name)
        if existing is None:
            merged[name] = runtime_spec
            continue
        merged[name] = _merge_field_specs(existing, runtime_spec, category_weights=category_weights, score_weights=score_weights)

    return FieldRegistry(fields=merged)


def _field_spec_from_catalog_row(
    record: dict[str, Any],
    *,
    max_alpha_usage: int,
    category_weights: dict[str, float],
    score_weights: FieldScoreWeights,
) -> FieldSpec:
    dataset = record.get("dataset")
    category = record.get("category")
    subcategory = record.get("subcategory")
    dataset_name = dataset.get("name") if isinstance(dataset, dict) else record.get("dataset_name") or record.get("dataset")
    category_name = category.get("name") if isinstance(category, dict) else record.get("category_name") or category
    subcategory_name = (
        subcategory.get("name")
        if isinstance(subcategory, dict)
        else record.get("subcategory_name") or record.get("subcategory")
    )
    field_type = normalize_field_type(record.get("type") or record.get("field_type") or "matrix")
    coverage = float(record.get("coverage") or record.get("dateCoverage") or record.get("date_coverage") or 0.0)
    alpha_usage = int(record.get("alphaCount") or record.get("alpha_count") or 0)
    normalized_category = normalize_category(category_name)
    category_weight = _resolve_category_weight(normalized_category, category_weights)
    field_score = compute_field_score(
        coverage=coverage,
        alpha_usage_count=alpha_usage,
        category_weight=category_weight,
        max_alpha_usage_count=max_alpha_usage,
        score_weights=score_weights,
    )
    return FieldSpec(
        name=str(record.get("id") or record.get("name")),
        dataset=str(dataset_name or ""),
        field_type=field_type,
        coverage=coverage,
        alpha_usage_count=alpha_usage,
        category=normalized_category,
        delay=int(record.get("delay") or 1),
        region=str(record.get("region") or ""),
        universe=str(record.get("universe") or ""),
        runtime_available=False,
        description=str(record.get("description") or ""),
        subcategory=normalize_category(subcategory_name),
        user_count=int(record.get("userCount") or record.get("user_count") or 0),
        category_weight=category_weight,
        field_score=field_score,
    )


def _field_spec_from_runtime_values(field_name: str, frame: pd.DataFrame, field_type: str) -> FieldSpec:
    coverage = _runtime_coverage(frame)
    category = normalize_category(frame["category"].dropna().iloc[0]) if frame["category"].notna().any() else "other"
    category_weight = _resolve_category_weight(category, {})
    return FieldSpec(
        name=field_name,
        dataset=str(frame["dataset"].dropna().iloc[0]) if frame["dataset"].notna().any() else "runtime",
        field_type=field_type,
        coverage=coverage,
        alpha_usage_count=0,
        category=category,
        delay=int(frame["delay"].dropna().iloc[0]) if frame["delay"].notna().any() else 1,
        region=str(frame["region"].dropna().iloc[0]) if frame["region"].notna().any() else "",
        universe=str(frame["universe"].dropna().iloc[0]) if frame["universe"].notna().any() else "",
        runtime_available=True,
        description=str(frame["description"].dropna().iloc[0]) if frame["description"].notna().any() else "",
        subcategory=normalize_category(frame["subcategory"].dropna().iloc[0]) if frame["subcategory"].notna().any() else "",
        user_count=0,
        category_weight=category_weight,
        field_score=coverage,
    )


def _runtime_specs_from_matrices(
    *,
    numeric_fields: dict[str, pd.DataFrame],
    group_fields: dict[str, pd.DataFrame],
    category_weights: dict[str, float],
    score_weights: FieldScoreWeights,
) -> dict[str, FieldSpec]:
    specs: dict[str, FieldSpec] = {}
    for name, frame in numeric_fields.items():
        category = _category_from_field_name(name)
        category_weight = _resolve_category_weight(category, category_weights)
        coverage = _matrix_coverage(frame)
        specs[name] = FieldSpec(
            name=name,
            dataset="runtime",
            field_type="matrix",
            coverage=coverage,
            alpha_usage_count=0,
            category=category,
            runtime_available=True,
            category_weight=category_weight,
            field_score=compute_field_score(
                coverage=coverage,
                alpha_usage_count=0,
                category_weight=category_weight,
                max_alpha_usage_count=1,
                score_weights=score_weights,
            ),
        )
    for name, frame in group_fields.items():
        category = "group"
        category_weight = _resolve_category_weight(category, category_weights)
        coverage = _matrix_coverage(frame)
        specs[name] = FieldSpec(
            name=name,
            dataset="runtime",
            field_type="vector",
            coverage=coverage,
            alpha_usage_count=0,
            category=category,
            runtime_available=True,
            category_weight=category_weight,
            field_score=compute_field_score(
                coverage=coverage,
                alpha_usage_count=0,
                category_weight=category_weight,
                max_alpha_usage_count=1,
                score_weights=score_weights,
            ),
        )
    return specs


def _merge_field_specs(
    left: FieldSpec,
    right: FieldSpec,
    *,
    category_weights: dict[str, float],
    score_weights: FieldScoreWeights,
) -> FieldSpec:
    coverage = max(left.coverage, right.coverage)
    alpha_usage = max(left.alpha_usage_count, right.alpha_usage_count)
    category = left.category if left.category != "other" else right.category
    category_weight = _resolve_category_weight(category, category_weights)
    field_score = max(left.field_score, right.field_score)
    if left.runtime_available or right.runtime_available:
        field_score = max(field_score, compute_field_score(
            coverage=coverage,
            alpha_usage_count=0,
            category_weight=category_weight,
            max_alpha_usage_count=1,
            score_weights=score_weights,
        ))
    return FieldSpec(
        name=left.name,
        dataset=left.dataset or right.dataset,
        field_type=left.field_type if left.field_type != "matrix" or right.field_type == "matrix" else right.field_type,
        coverage=coverage,
        alpha_usage_count=alpha_usage,
        category=category,
        delay=left.delay or right.delay,
        region=left.region or right.region,
        universe=left.universe or right.universe,
        runtime_available=left.runtime_available or right.runtime_available,
        description=left.description or right.description,
        subcategory=left.subcategory or right.subcategory,
        user_count=max(left.user_count, right.user_count),
        category_weight=category_weight,
        field_score=field_score,
    )


def compute_field_score(
    *,
    coverage: float,
    alpha_usage_count: int,
    category_weight: float,
    max_alpha_usage_count: int,
    score_weights: FieldScoreWeights,
) -> float:
    coverage_norm = max(0.0, min(1.0, float(coverage)))
    usage_norm = math.log1p(max(alpha_usage_count, 0)) / math.log1p(max(max_alpha_usage_count, 1))
    return (
        score_weights.coverage * coverage_norm
        + score_weights.usage * usage_norm
        + score_weights.category * max(0.0, min(1.0, category_weight))
    )


def _infer_runtime_field_type(frame: pd.DataFrame) -> str:
    if frame["field_type"].notna().any():
        normalized = normalize_field_type(frame["field_type"].dropna().iloc[0])
        if normalized in GROUP_FIELD_TYPES:
            return "vector"
        if normalized in NUMERIC_FIELD_TYPES:
            return normalized
    numeric = pd.to_numeric(frame["value"], errors="coerce")
    observed = frame["value"].notna().sum()
    if observed <= 0:
        return "matrix"
    numeric_ratio = float(numeric.notna().sum()) / float(observed)
    return "matrix" if numeric_ratio >= 0.95 else "vector"


def _resolve_category_weight(category: str, category_weights: dict[str, float]) -> float:
    if category in category_weights:
        return float(category_weights[category])
    prefix = category.split("_", 1)[0]
    if prefix in category_weights:
        return float(category_weights[prefix])
    return 0.50


def _matrix_coverage(frame: pd.DataFrame) -> float:
    total = float(frame.shape[0] * frame.shape[1]) if not frame.empty else 0.0
    if total <= 0:
        return 0.0
    return float(frame.notna().sum().sum()) / total


def _runtime_coverage(frame: pd.DataFrame) -> float:
    return 1.0 if not frame.empty else 0.0


def _category_from_field_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized in {"open", "high", "low", "close", "returns"}:
        return "price"
    if normalized in {"volume", "adv20"}:
        return "volume"
    return "other"


def _field_rank_tuple(spec: FieldSpec) -> tuple[float, float, int]:
    return (spec.field_score, spec.coverage, spec.alpha_usage_count)
