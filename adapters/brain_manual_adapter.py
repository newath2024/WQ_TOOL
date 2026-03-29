from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from adapters.simulation_adapter import SimulationAdapter


class BrainManualAdapter(SimulationAdapter):
    """Manual workflow adapter for environments without direct BRAIN API automation."""

    def __init__(self, export_root: str | Path = "outputs/brain_manual") -> None:
        self.export_root = Path(export_root)

    def submit_simulation(self, expression: str, sim_config: dict) -> dict:
        submissions = self.batch_submit([expression], sim_config)
        return submissions[0]

    def batch_submit(self, expressions: list[str], sim_config: dict) -> list[dict]:
        batch_id = str(sim_config.get("batch_id") or f"manual-{uuid4().hex[:12]}")
        candidate_payloads = list(sim_config.get("candidate_payloads") or [])
        payload_by_expression = {
            str(payload.get("expression")): payload
            for payload in candidate_payloads
            if payload.get("expression")
        }
        export_path = self.export_candidates_for_manual_simulation(
            expressions=expressions,
            sim_config=sim_config,
            batch_id=batch_id,
            payload_by_expression=payload_by_expression,
        )
        submitted_at = datetime.now(UTC).isoformat()
        submissions: list[dict] = []
        for index, expression in enumerate(expressions, start=1):
            metadata = payload_by_expression.get(expression, {})
            submissions.append(
                {
                    "job_id": str(metadata.get("job_id") or f"{batch_id}-{index:04d}"),
                    "candidate_id": str(metadata.get("candidate_id") or ""),
                    "expression": expression,
                    "status": "manual_pending",
                    "backend": "manual",
                    "batch_id": batch_id,
                    "submitted_at": submitted_at,
                    "export_path": str(export_path),
                    "raw_submission": {
                        "export_path": str(export_path),
                        "batch_id": batch_id,
                        "manual_mode": True,
                    },
                }
            )
        return submissions

    def get_simulation_status(self, job_id: str) -> dict:
        return {"job_id": job_id, "status": "manual_pending"}

    def get_simulation_result(self, job_id: str) -> dict:
        return {"job_id": job_id, "status": "manual_pending", "raw_result": {}}

    def export_candidates_for_manual_simulation(
        self,
        *,
        expressions: list[str],
        sim_config: dict,
        batch_id: str,
        payload_by_expression: dict[str, dict],
    ) -> Path:
        export_dir = Path(str(sim_config.get("manual_export_dir") or self.export_root)).expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / f"brain_candidates_{batch_id}.csv"
        rows: list[dict[str, object]] = []
        for index, expression in enumerate(expressions, start=1):
            payload = payload_by_expression.get(expression, {})
            rows.append(
                {
                    "batch_id": batch_id,
                    "job_id": payload.get("job_id") or f"{batch_id}-{index:04d}",
                    "candidate_id": payload.get("candidate_id", ""),
                    "expression": expression,
                    "generation_mode": payload.get("generation_mode", ""),
                    "template_name": payload.get("template_name", ""),
                    "fields_used": "|".join(str(item) for item in payload.get("fields_used", []) if str(item)),
                    "operators_used": "|".join(
                        str(item) for item in payload.get("operators_used", []) if str(item)
                    ),
                    "run_id": payload.get("run_id", ""),
                    "round_index": payload.get("round_index", 0),
                    "sim_config_json": json.dumps(sim_config, sort_keys=True),
                    "generation_metadata_json": json.dumps(payload.get("generation_metadata", {}), sort_keys=True),
                }
            )
        with export_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["expression"])
            writer.writeheader()
            writer.writerows(rows)
        return export_path

    def import_manual_results(self, path: str | Path) -> list[dict]:
        import_path = Path(path).expanduser().resolve()
        rows: list[dict] = []
        with import_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = {"status"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Manual result file '{import_path}' is missing required columns: {sorted(missing)}"
                )
            for raw_row in reader:
                metrics = {
                    key: _to_float(raw_row.get(key))
                    for key in ("sharpe", "fitness", "turnover", "drawdown", "returns", "margin")
                }
                raw_result_json = raw_row.get("raw_result_json") or "{}"
                try:
                    raw_result = json.loads(raw_result_json)
                except json.JSONDecodeError:
                    raw_result = {"raw_result_text": raw_result_json}
                rows.append(
                    {
                        "job_id": raw_row.get("job_id", ""),
                        "candidate_id": raw_row.get("candidate_id", ""),
                        "batch_id": raw_row.get("batch_id", ""),
                        "expression": raw_row.get("expression", ""),
                        "status": str(raw_row["status"]).strip().lower(),
                        "region": raw_row.get("region", ""),
                        "universe": raw_row.get("universe", ""),
                        "delay": _to_int(raw_row.get("delay"), default=1),
                        "neutralization": raw_row.get("neutralization", ""),
                        "decay": _to_int(raw_row.get("decay"), default=0),
                        "metrics": metrics,
                        "submission_eligible": _to_bool(raw_row.get("submission_eligible")),
                        "rejection_reason": raw_row.get("rejection_reason") or None,
                        "raw_result": raw_result,
                        "simulated_at": raw_row.get("simulated_at")
                        or datetime.now(UTC).isoformat(),
                    }
                )
        return rows


def _to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _to_int(value: str | None, *, default: int) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _to_bool(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value '{value}' from manual result file.")
