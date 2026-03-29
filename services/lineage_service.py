from __future__ import annotations

import json

from services.models import CommandEnvironment, LineageViewRow
from storage.repository import SQLiteRepository


def get_lineage(
    repository: SQLiteRepository,
    environment: CommandEnvironment,
    alpha_id: str,
) -> list[LineageViewRow]:
    """Return lineage rows for a given alpha id."""
    rows = repository.alpha_history.get_lineage(run_id=environment.context.run_id, alpha_id=alpha_id)
    result: list[LineageViewRow] = []
    for row in rows:
        diagnosis = json.loads(row["diagnosis_summary_json"]) if row["diagnosis_summary_json"] else {}
        result.append(
            LineageViewRow(
                depth=int(row["depth"]),
                run_id=str(row["run_id"]),
                alpha_id=str(row["alpha_id"]),
                outcome_score=float(row["outcome_score"]) if row["outcome_score"] is not None else None,
                fail_tags=tuple(diagnosis.get("fail_tags", [])),
                expression=str(row["expression"] or "-"),
            )
        )
    return result
