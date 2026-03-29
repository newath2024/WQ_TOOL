from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_text(value: Any) -> str:
    text = str(value or "")
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _slugify(value: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in value.strip().lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


@dataclass(frozen=True, slots=True)
class OperatorKnowledge:
    name: str
    signature: str
    tier: str | None
    scope: tuple[str, ...]
    summary: str
    details: str
    examples: tuple[str, ...]
    tips: tuple[str, ...]
    constraints: tuple[str, ...]
    semantic_tags: tuple[str, ...]
    parameter_requirements: dict[str, str]
    preferred_motifs: tuple[str, ...]
    turnover_hint: float | None = None


def load_operator_knowledge(paths: Iterable[str]) -> dict[str, OperatorKnowledge]:
    knowledge_by_name: dict[str, OperatorKnowledge] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        operators = payload.get("operators", []) if isinstance(payload, dict) else []
        if not isinstance(operators, list):
            continue
        for entry in operators:
            if not isinstance(entry, dict):
                continue
            knowledge = _build_knowledge(entry)
            if knowledge is None:
                continue
            existing = knowledge_by_name.get(knowledge.name)
            if existing is None or _richness_score(knowledge) > _richness_score(existing):
                knowledge_by_name[knowledge.name] = knowledge
    return knowledge_by_name


def _build_knowledge(entry: dict[str, Any]) -> OperatorKnowledge | None:
    signature = _normalize_text(entry.get("signature", ""))
    name = _normalize_text(entry.get("name", signature.split("(", 1)[0]))
    if not name:
        return None
    summary = _normalize_text(entry.get("summary", ""))
    details = _normalize_text(entry.get("details", "") or entry.get("detail_text", ""))
    examples = tuple(_normalize_text(item) for item in entry.get("examples", []) if _normalize_text(item))
    tips = tuple(_normalize_text(item) for item in entry.get("tips", []) if _normalize_text(item))
    constraints = tuple(_normalize_text(item) for item in entry.get("constraints", []) if _normalize_text(item))
    behavior_tags = tuple(_slugify(item) for item in entry.get("behavior_tags", []) if _slugify(item))
    scope_tokens = tuple(
        token for token in (_slugify(part) for part in str(entry.get("scope", "")).split(",")) if token
    )
    semantic_tags = _infer_semantic_tags(
        name=name,
        signature=signature,
        summary=summary,
        details=details,
        examples=examples,
        tips=tips,
        constraints=constraints,
        behavior_tags=behavior_tags,
        scope=scope_tokens,
    )
    return OperatorKnowledge(
        name=name,
        signature=signature,
        tier=_normalize_text(entry.get("tier", "")).lower() or None,
        scope=scope_tokens,
        summary=summary,
        details=details,
        examples=examples,
        tips=tips,
        constraints=constraints,
        semantic_tags=semantic_tags,
        parameter_requirements=_infer_parameter_requirements(signature=signature, details=details, constraints=constraints),
        preferred_motifs=_infer_preferred_motifs(name=name, tags=semantic_tags, examples=examples),
        turnover_hint=_infer_turnover_hint(
            name=name,
            tags=semantic_tags,
            summary=summary,
            details=details,
            tips=tips,
            constraints=constraints,
        ),
    )


def _infer_semantic_tags(
    *,
    name: str,
    signature: str,
    summary: str,
    details: str,
    examples: tuple[str, ...],
    tips: tuple[str, ...],
    constraints: tuple[str, ...],
    behavior_tags: tuple[str, ...],
    scope: tuple[str, ...],
) -> tuple[str, ...]:
    tags: set[str] = set(behavior_tags)
    text = " ".join(part for part in (name, signature, summary, details, *examples, *tips, *constraints) if part).lower()

    if name.startswith("ts_") or "rolling" in text or "past d" in text or "lookback" in text:
        tags.add("time_series")
    if name.startswith("group_") or "within each group" in text or "group" in signature.lower():
        tags.update({"group_aware", "group_operator"})
    if name in {"rank", "zscore", "sign", "abs"} or name.startswith("group_rank") or name.startswith("group_zscore"):
        tags.update({"normalization", "wrapper_safe"})
    if "correlation" in text or "covariance" in text:
        tags.update({"cross_field", "pairwise"})
    if "volatility" in text or "standard deviation" in text or "std_dev" in name or name.endswith("_std"):
        tags.add("volatility")
    if "moving average" in text or "rolling mean" in text or "smoothing" in text or "smooth" in text:
        tags.add("smoothing")
    if "reduce turnover" in text or "reducing turnover" in text or "help reduce turnover" in text or "more stable" in text:
        tags.update({"reduces_turnover", "smoothing"})
    if name in {"ts_delta", "returns"} or "difference between the current value" in text or "changed over a given time window" in text:
        tags.update({"change_sensitive", "momentum_oriented"})
    if "nan" in text or "missing" in text:
        tags.add("nan_aware")
    if "treat nans as zero" in text or "treat nans as zeros" in text or "replaces missing" in text or "improve coverage" in text:
        tags.update({"coverage_improving", "nan_aware"})
    if name == "log" or "undefined for zero or negative" in text or "input x should be positive" in text:
        tags.add("requires_positive_input")
    if "selection" in scope:
        tags.add("selection_capable")
    if "regular" in scope or "combo" in scope:
        tags.add("alpha_capable")
    if any("adv20" in example.lower() or "volume" in example.lower() for example in examples):
        tags.add("liquidity_context")
    return tuple(sorted(tags))


def _infer_parameter_requirements(*, signature: str, details: str, constraints: tuple[str, ...]) -> dict[str, str]:
    requirements: dict[str, str] = {}
    signature_lower = signature.lower()
    details_lower = " ".join((details, *constraints)).lower()
    if "(x, d" in signature_lower or ", d)" in signature_lower or "lookback" in signature_lower or "past d" in details_lower:
        requirements["window"] = "positive_int"
    if "group" in signature_lower:
        requirements["group"] = "group_field"
    if "filter" in signature_lower:
        requirements["filter"] = "bool_literal"
    if "factor" in signature_lower:
        requirements["factor"] = "unit_interval" if "0 to 1" in details_lower else "numeric_literal"
    return requirements


def _infer_preferred_motifs(*, name: str, tags: tuple[str, ...], examples: tuple[str, ...]) -> tuple[str, ...]:
    motifs: list[str] = []
    tag_set = set(tags)
    if "momentum_oriented" in tag_set or name in {"ts_delay", "ts_delta"}:
        motifs.extend(["momentum", "liquidity_conditioned_signal"])
    if "smoothing" in tag_set or name in {"ts_mean", "ts_decay_linear", "ts_sum"}:
        motifs.extend(["mean_reversion", "spread", "ratio", "residualized_signal"])
    if "volatility" in tag_set:
        motifs.extend(["volatility_adjusted_momentum", "regime_conditioned_signal"])
    if "group_aware" in tag_set:
        motifs.append("group_relative_signal")
    if "cross_field" in tag_set:
        motifs.extend(["spread", "ratio", "residualized_signal"])
    if "coverage_improving" in tag_set:
        motifs.append("liquidity_conditioned_signal")
    if "liquidity_context" in tag_set or any("adv20" in item.lower() or "volume" in item.lower() for item in examples):
        motifs.append("liquidity_conditioned_signal")
    seen: list[str] = []
    for motif in motifs:
        if motif not in seen:
            seen.append(motif)
    return tuple(seen)


def _infer_turnover_hint(
    *,
    name: str,
    tags: tuple[str, ...],
    summary: str,
    details: str,
    tips: tuple[str, ...],
    constraints: tuple[str, ...],
) -> float | None:
    tag_set = set(tags)
    text = " ".join((summary, details, *tips, *constraints)).lower()
    if "reduces_turnover" in tag_set or name in {"ts_decay_linear", "ts_mean"}:
        return -0.30
    if name in {"ts_std_dev", "ts_corr", "ts_covariance", "ts_rank", "ts_sum"}:
        return -0.10
    if "change_sensitive" in tag_set or name in {"ts_delta", "returns"}:
        return 0.80
    if name in {"rank", "zscore", "group_rank", "group_zscore"}:
        return 0.08
    if name == "sign":
        return 0.20
    if "stability" in text:
        return -0.12
    return None


def _richness_score(knowledge: OperatorKnowledge) -> tuple[int, int, int, int, int, int, int]:
    return (
        len(knowledge.details),
        len(knowledge.summary),
        len(knowledge.examples),
        len(knowledge.tips),
        len(knowledge.constraints),
        len(knowledge.semantic_tags),
        len(knowledge.preferred_motifs),
    )
