from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any


from storage.repositories.alpha_repository import AlphaRepository


class QualityRepository:
    def __init__(self, connection: sqlite3.Connection, alpha_repository: AlphaRepository) -> None:
        self.connection = connection
        self.alphas = alpha_repository

    def list_quality_polish_parent_rows(
        self,
        *,
        run_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return self.alphas._list_recent_completed_parent_rows(run_id=run_id, limit=limit)

    def list_quality_polish_usage_keys(self, run_id: str) -> dict[str, Any]:
        rows = self.connection.execute(
            """
            SELECT alpha_id, normalized_expression, generation_metadata, created_at
            FROM alphas
            WHERE run_id = ?
              AND generation_mode = 'quality_polish'
            """,
            (run_id,),
        ).fetchall()
        signatures: set[str] = set()
        parent_transform_keys: set[str] = set()
        usage_rows: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(row["generation_metadata"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            parent_alpha_id = str(metadata.get("polish_parent_alpha_id") or "").strip()
            transform = str(metadata.get("polish_transform") or "").strip()
            normalized_expression = str(row["normalized_expression"] or "").strip()
            signature = str(metadata.get("polish_signature") or "").strip()
            if not signature and parent_alpha_id and transform and normalized_expression:
                signature = _quality_polish_signature(
                    run_id=run_id,
                    parent_alpha_id=parent_alpha_id,
                    transform=transform,
                    normalized_expression=normalized_expression,
                )
            if signature:
                signatures.add(signature)
            parent_transform_key = str(metadata.get("polish_parent_transform_key") or "").strip()
            if not parent_transform_key and parent_alpha_id and transform:
                parent_transform_key = f"{parent_alpha_id}:{transform}"
            if parent_transform_key:
                parent_transform_keys.add(parent_transform_key)
            usage_rows.append(
                {
                    "alpha_id": str(row["alpha_id"] or "").strip(),
                    "polish_signature": signature,
                    "polish_parent_transform_key": parent_transform_key,
                    "polish_parent_alpha_id": parent_alpha_id,
                    "polish_transform": transform,
                    "polish_transform_group": str(
                        metadata.get("polish_transform_group")
                        or _quality_polish_transform_group(transform)
                    ).strip(),
                    "normalized_expression": normalized_expression,
                    "polish_round_index": _optional_int(metadata.get("polish_round_index")),
                    "created_at": str(row["created_at"] or ""),
                }
            )
        return {
            "signatures": signatures,
            "parent_transform_keys": parent_transform_keys,
            "usage_rows": usage_rows,
        }


def _quality_polish_signature(
    *,
    run_id: str,
    parent_alpha_id: str,
    transform: str,
    normalized_expression: str,
) -> str:
    payload = "\x1f".join((str(run_id), str(parent_alpha_id), str(transform), str(normalized_expression)))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _quality_polish_transform_group(transform: str) -> str:
    normalized = str(transform or "").strip()
    if normalized.startswith("window_perturb"):
        return "window_perturb"
    for prefix in ("smooth_ts_mean", "smooth_ts_decay_linear", "smooth_ts_rank"):
        if normalized.startswith(prefix):
            return prefix
    return normalized


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
