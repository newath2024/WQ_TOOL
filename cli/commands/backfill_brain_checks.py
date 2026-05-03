from __future__ import annotations

import argparse
import json

from core.brain_checks import first_synthetic_rejection_message, names_json, summarize_brain_checks
from core.config import AppConfig
from core.quality_score import MultiObjectiveQualityScorer
from services.models import CommandEnvironment
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "backfill-brain-checks",
        help="Parse stored BRAIN raw results, persist check summaries, and recompute quality scores.",
        parents=[common],
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for incremental backfills.")
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    del environment
    clauses: list[str] = []
    params: list[object] = []
    if args.run_id:
        clauses.append("run_id = ?")
        params.append(str(args.run_id))
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_sql = "LIMIT ?" if args.limit is not None else ""
    if args.limit is not None:
        params.append(max(0, int(args.limit)))
    rows = repository.connection.execute(
        f"""
        SELECT *
        FROM brain_results
        {where_sql}
        ORDER BY created_at ASC, job_id ASC
        {limit_sql}
        """,
        tuple(params),
    ).fetchall()
    updated = 0
    ready_count = 0
    for row in rows:
        existing_rejection = str(row["rejection_reason"] or "") or None
        summary = summarize_brain_checks(
            row["raw_result_json"],
            status=str(row["status"] or ""),
            rejection_reason=existing_rejection,
        )
        synthetic_rejection = first_synthetic_rejection_message(summary)
        rejection_reason = existing_rejection or synthetic_rejection
        if (
            existing_rejection
            and str(row["status"] or "").strip().lower() == "completed"
            and not synthetic_rejection
            and existing_rejection == str(summary.blocking_message or "")
        ):
            rejection_reason = None
        if rejection_reason != existing_rejection:
            summary = summarize_brain_checks(
                row["raw_result_json"],
                status=str(row["status"] or ""),
                rejection_reason=rejection_reason,
            )
        quality_score = MultiObjectiveQualityScorer.score(
            metrics={
                "fitness": row["fitness"],
                "sharpe": row["sharpe"],
                "turnover": row["turnover"],
                "drawdown": row["drawdown"],
                "returns": row["returns"],
                "margin": row["margin"],
            },
            submission_eligible=None if row["submission_eligible"] is None else bool(row["submission_eligible"]),
            rejection_reason=rejection_reason,
            status=str(row["status"] or ""),
            check_summary=summary,
            quality_config=config.quality_score,
        )
        if summary.derived_submit_ready:
            ready_count += 1
        repository.connection.execute(
            """
            UPDATE brain_results
            SET check_summary_json = ?,
                hard_fail_checks_json = ?,
                warning_checks_json = ?,
                blocking_warning_checks_json = ?,
                derived_submit_ready = ?,
                rejection_reason = ?,
                quality_score = ?
            WHERE job_id = ?
            """,
            (
                json.dumps(summary.to_dict(), sort_keys=True),
                names_json(summary.hard_fail_checks),
                names_json(summary.warning_checks),
                names_json(summary.blocking_warning_checks),
                None if summary.derived_submit_ready is None else int(summary.derived_submit_ready),
                rejection_reason or None,
                float(quality_score),
                row["job_id"],
            ),
        )
        updated += 1
    repository.connection.commit()
    print(f"rows_scanned: {len(rows)}")
    print(f"rows_updated: {updated}")
    print(f"derived_submit_ready: {ready_count}")
    return 0
