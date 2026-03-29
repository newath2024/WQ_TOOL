from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader

from features.registry import BRAIN_DEFAULT_OPERATORS


TIER_PATTERN = re.compile(r"(base|genius|expert|master|grandmaster)$", re.IGNORECASE)
SUPPORTED_SIGNATURE_MARKERS: dict[str, str] = {
    "abs": "abs(x)",
    "log": "log(x)",
    "sign": "sign(x)",
    "rank": "rank(x)",
    "zscore": "zscore(x)",
    "ts_delay": "ts_delay(x, d)",
    "ts_delta": "ts_delta(x, d)",
    "ts_mean": "ts_mean(x, d)",
    "ts_std_dev": "ts_std_dev(x, d)",
    "ts_min": "ts_min(x, d)",
    "ts_max": "ts_max(x, d)",
    "ts_corr": "ts_corr(x, y, d)",
    "ts_covariance": "ts_covariance(y, x, d)",
    "ts_decay_linear": "ts_decay_linear(",
    "ts_rank": "ts_rank(x, d",
    "ts_sum": "ts_sum(x, d)",
    "group_rank": "group_rank(x, group)",
    "group_zscore": "group_zscore(x, group)",
    "group_neutralize": "group_neutralize(x, group)",
}


@dataclass(slots=True)
class OperatorCatalogEntry:
    signature: str
    name: str
    tier: str | None
    scope: str
    summary: str
    details: str


def _normalize_text(value: str) -> str:
    cleaned = (
        value.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _load_export_rows(export_json_path: Path) -> list[dict]:
    payload = json.loads(export_json_path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def _summary_from_block(block: str) -> tuple[str, str, str | None]:
    lines = [_normalize_text(line) for line in block.splitlines() if _normalize_text(line)]
    tier = None
    scope_parts: list[str] = []
    summary_parts: list[str] = []
    if len(lines) >= 2 and lines[1].lower() in {"base", "genius", "expert", "master", "grandmaster"}:
        tier = lines[1].lower()
        index = 2
        while index < len(lines):
            if "Show less" in lines[index]:
                break
            tokens = [token.strip(" ,").lower() for token in lines[index].split() if token.strip(" ,")]
            if tokens and all(token in {"combo", "regular", "selection"} for token in tokens):
                scope_parts.append(lines[index].strip(" ,"))
                index += 1
                continue
            summary_parts.append(lines[index])
            index += 1
            if summary_parts:
                break
    summary = _normalize_text(" ".join(summary_parts))
    scope = _normalize_text(", ".join(scope_parts))
    return summary, scope, tier


def _parse_operator_field(raw_value: str) -> tuple[str, str, str | None]:
    normalized = _normalize_text(raw_value)
    tier_match = TIER_PATTERN.search(normalized)
    tier = tier_match.group(1).lower() if tier_match else None
    signature = TIER_PATTERN.sub("", normalized).strip()
    name = signature.split("(", 1)[0].strip()
    return signature, name, tier


def _find_block(signature: str, full_text: str, ordered_signatures: list[str]) -> str:
    start = full_text.find(signature)
    if start < 0:
        alt_signature = signature.replace(" ", "")
        start = full_text.replace(" ", "").find(alt_signature)
        if start < 0:
            return ""
    end = len(full_text)
    for other in ordered_signatures:
        if other == signature:
            continue
        candidate = full_text.find(other, start + len(signature))
        if candidate >= 0:
            end = min(end, candidate)
    return full_text[start:end]


def _extract_details(block: str, summary: str) -> str:
    if not block:
        return ""
    text = _normalize_text(block)
    if "Show less" in text:
        text = text.split("Show less", 1)[1].strip()
    if summary and text.startswith(summary):
        text = text[len(summary) :].strip()
    text = re.sub(
        r"\b(Simulate Alphas Learn Data Labs Genius|Simulation Settings|Open example alpha in Simulate)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return _normalize_text(text)


def _build_rows_from_pdf_supported_markers(pdf_text: str) -> list[dict]:
    starts: list[tuple[int, str, str]] = []
    for name in BRAIN_DEFAULT_OPERATORS:
        marker = SUPPORTED_SIGNATURE_MARKERS.get(name)
        if not marker:
            continue
        position = pdf_text.find(marker)
        if position >= 0:
            starts.append((position, name, marker))
    starts.sort()

    rows: list[dict] = []
    for index, (position, name, marker) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(pdf_text)
        block = pdf_text[position:end]
        summary, scope, tier = _summary_from_block(block)
        signature = marker if marker.endswith(")") else marker.rstrip()
        if marker.endswith("("):
            signature = f"{name}(...)"
        elif marker.endswith(", d"):
            signature = f"{name}(...)"
        rows.append(
            {
                "Operator": f"{signature}{tier or ''}",
                "Scope": scope,
                "Description": summary,
            }
        )
    return rows


def build_catalog(pdf_path: Path, export_json_path: Path | None = None) -> dict[str, object]:
    pdf_text = _extract_pdf_text(pdf_path)
    rows = (
        _load_export_rows(export_json_path)
        if export_json_path is not None and export_json_path.exists()
        else _build_rows_from_pdf_supported_markers(pdf_text)
    )
    ordered_signatures = []
    parsed_rows = []
    for row in rows:
        signature, name, tier = _parse_operator_field(str(row.get("Operator", "")))
        if not signature:
            continue
        ordered_signatures.append(signature)
        parsed_rows.append((row, signature, name, tier))

    entries: list[OperatorCatalogEntry] = []
    for row, signature, name, tier in parsed_rows:
        summary = _normalize_text(str(row.get("Description", "")))
        block = _find_block(signature, pdf_text, ordered_signatures)
        details = _extract_details(block, summary)
        entries.append(
            OperatorCatalogEntry(
                signature=signature,
                name=name,
                tier=tier,
                scope=_normalize_text(str(row.get("Scope", ""))),
                summary=summary,
                details=details,
            )
        )

    return {
        "source_pdf": str(pdf_path),
        "source_export_json": str(export_json_path) if export_json_path is not None else None,
        "generated_at": datetime.now(UTC).isoformat(),
        "operator_count": len(entries),
        "operators": [asdict(entry) for entry in entries],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a JSON operator catalog from a BRAIN operators PDF and exported table JSON.")
    parser.add_argument("--pdf", required=True, help="Path to Operators.pdf")
    parser.add_argument("--export-json", required=False, help="Optional path to operators table export JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON catalog")
    args = parser.parse_args()

    export_json_path = Path(args.export_json) if args.export_json else None
    catalog = build_catalog(Path(args.pdf), export_json_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
