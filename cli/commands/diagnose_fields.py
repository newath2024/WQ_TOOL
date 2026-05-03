from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from core.config import AppConfig
from data.field_registry import FieldSpec
from domain.candidate import AlphaCandidate
from services.brain_service import BrainService
from services.data_service import load_research_context
from services.models import CommandEnvironment
from services.runtime_service import init_run
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "diagnose-fields",
        help="Build opt-in BRAIN field diagnostic expressions; submit only with --submit.",
        parents=[common],
    )
    parser.add_argument("--top", type=int, default=20, help="Number of top-scored fields to diagnose.")
    parser.add_argument(
        "--max-expressions",
        type=int,
        default=None,
        help="Maximum diagnostic expressions to print or submit.",
    )
    parser.add_argument("--windows", default="5,22,66,252", help="Comma-separated windows for update-frequency checks.")
    parser.add_argument("--bounds", default="0,1,10", help="Comma-separated thresholds for abs(field) bound checks.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diagnostics without submitting. This is the default.",
    )
    mode_group.add_argument("--submit", action="store_true", help="Submit and poll diagnostics through the configured BRAIN API.")
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    context = load_research_context(config, environment, stage="diagnose-fields")
    fields = context.field_registry.generation_numeric_fields(
        config.generation.allowed_fields,
        include_catalog_fields=config.generation.allow_catalog_fields_without_runtime,
    )[: max(0, int(args.top))]
    specs = build_field_diagnostic_specs(
        fields=fields,
        windows=_parse_ints(args.windows),
        bounds=_parse_floats(args.bounds),
        max_expressions=args.max_expressions,
        run_id=environment.context.run_id,
    )
    if not args.submit:
        for spec in specs:
            print(f"{spec['field_name']}\t{spec['diagnostic_name']}\t{spec['expression']}")
        print(f"dry_run: true")
        print(f"expression_count: {len(specs)}")
        return 0

    init_run(repository, config, environment, status="diagnosing_fields")
    candidates = [_candidate_from_spec(spec, created_at=datetime.now(UTC).isoformat()) for spec in specs]
    if not candidates:
        print("expression_count: 0")
        repository.update_run_status(environment.context.run_id, "field_diagnostics_empty")
        return 0
    brain_service = BrainService(repository, config.brain)
    batch_id = f"diag-{environment.context.run_id[:8]}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    sim_config = brain_service.build_simulation_config(
        config=config,
        environment=environment,
        round_index=0,
        batch_id=batch_id,
        candidates=candidates,
    )
    sim_config.update(
        {
            "neutralization": "NONE",
            "decay": 0,
            "visualization": False,
            "simulation_profile": "field_diagnostics_none_decay0",
        }
    )
    _insert_planned_diagnostics(
        repository,
        run_id=environment.context.run_id,
        specs=specs,
        sim_config=sim_config,
    )
    submitted_batch = brain_service.submit_candidates(
        candidates,
        config=config,
        environment=environment,
        round_index=0,
        batch_size=len(candidates),
        sim_config_override=sim_config,
    )
    batch = brain_service.poll_batch(submitted_batch, config=config, environment=environment)
    specs_by_candidate = {str(spec["candidate_id"]): spec for spec in specs}
    for result in batch.results:
        spec = specs_by_candidate.get(result.candidate_id)
        if spec is None:
            continue
        _update_diagnostic_result(repository, spec=spec, result=result, sim_config=sim_config)
    repository.update_run_status(environment.context.run_id, "field_diagnostics_completed")
    print(f"run_id: {environment.context.run_id}")
    print(f"batch_id: {batch.batch_id}")
    print(f"submitted: {batch.submitted_count}")
    print(f"completed: {batch.completed_count}")
    return 0


def build_field_diagnostic_specs(
    *,
    fields: list[FieldSpec],
    windows: list[int],
    bounds: list[float],
    max_expressions: int | None,
    run_id: str = "",
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for field in fields:
        specs.append(_spec(field, "raw", {}, field.name, run_id=run_id))
        specs.append(
            _spec(field, "non_zero_coverage", {}, f"{field.name} != 0 ? 1 : -1", run_id=run_id)
        )
        for window in windows:
            specs.append(
                _spec(
                    field,
                    "update_frequency",
                    {"window": int(window)},
                    f"ts_std_dev({field.name},{int(window)}) != 0 ? 1 : -1",
                    run_id=run_id,
                )
            )
        for bound in bounds:
            bound_text = _format_number(bound)
            specs.append(
                _spec(
                    field,
                    "absolute_bound",
                    {"threshold": float(bound)},
                    f"abs({field.name}) > {bound_text} ? 1 : -1",
                    run_id=run_id,
                )
            )
    if max_expressions is not None:
        return specs[: max(0, int(max_expressions))]
    return specs


def _spec(
    field: FieldSpec,
    name: str,
    params: dict[str, Any],
    expression: str,
    *,
    run_id: str = "",
) -> dict[str, Any]:
    fingerprint_payload = f"{run_id}\x1f{field.name}\x1f{name}\x1f{json.dumps(params, sort_keys=True)}"
    candidate_id = "diag_" + hashlib.sha1(fingerprint_payload.encode("utf-8")).hexdigest()[:16]
    return {
        "candidate_id": candidate_id,
        "diagnostic_id": candidate_id,
        "field_name": field.name,
        "diagnostic_name": name,
        "params": params,
        "expression": expression,
    }


def _candidate_from_spec(spec: dict[str, Any], *, created_at: str) -> AlphaCandidate:
    return AlphaCandidate(
        alpha_id=str(spec["candidate_id"]),
        expression=str(spec["expression"]),
        normalized_expression=str(spec["expression"]),
        generation_mode="field_diagnostic",
        parent_ids=(),
        complexity=1,
        created_at=created_at,
        template_name=str(spec["diagnostic_name"]),
        fields_used=(str(spec["field_name"]),),
        operators_used=tuple(_diagnostic_operators(str(spec["diagnostic_name"]))),
        depth=2,
        generation_metadata={"diagnostic_params": dict(spec["params"])},
    )


def _insert_planned_diagnostics(
    repository: SQLiteRepository,
    *,
    run_id: str,
    specs: list[dict[str, Any]],
    sim_config: dict[str, object],
) -> None:
    timestamp = datetime.now(UTC).isoformat()
    for spec in specs:
        repository.connection.execute(
            """
            INSERT INTO field_diagnostics
            (diagnostic_id, run_id, field_name, region, universe, delay, neutralization, decay,
             diagnostic_name, params_json, expression, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(diagnostic_id) DO UPDATE SET
                run_id = excluded.run_id,
                field_name = excluded.field_name,
                region = excluded.region,
                universe = excluded.universe,
                delay = excluded.delay,
                neutralization = excluded.neutralization,
                decay = excluded.decay,
                diagnostic_name = excluded.diagnostic_name,
                params_json = excluded.params_json,
                expression = excluded.expression,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                spec["diagnostic_id"],
                run_id,
                spec["field_name"],
                str(sim_config.get("region") or ""),
                str(sim_config.get("universe") or ""),
                int(sim_config.get("delay") or 1),
                str(sim_config.get("neutralization") or "NONE"),
                int(sim_config.get("decay") or 0),
                spec["diagnostic_name"],
                json.dumps(spec["params"], sort_keys=True),
                spec["expression"],
                "planned",
                timestamp,
                timestamp,
            ),
        )
    repository.connection.commit()


def _update_diagnostic_result(
    repository: SQLiteRepository,
    *,
    spec: dict[str, Any],
    result,
    sim_config: dict[str, object],
) -> None:
    summary = dict(result.check_summary or {})
    long_count = _optional_int(summary.get("long_count"))
    short_count = _optional_int(summary.get("short_count"))
    universe_size = _universe_size(str(sim_config.get("universe") or ""))
    coverage_ratio = None
    if universe_size and long_count is not None and short_count is not None:
        coverage_ratio = float(long_count + short_count) / float(universe_size)
    timestamp = datetime.now(UTC).isoformat()
    repository.connection.execute(
        """
        UPDATE field_diagnostics
        SET job_id = ?,
            status = ?,
            long_count = ?,
            short_count = ?,
            coverage_ratio = ?,
            raw_result_json = ?,
            updated_at = ?
        WHERE diagnostic_id = ?
        """,
        (
            result.job_id,
            result.status,
            long_count,
            short_count,
            coverage_ratio,
            json.dumps(result.raw_result, sort_keys=True),
            timestamp,
            spec["diagnostic_id"],
        ),
    )
    repository.connection.commit()


def _diagnostic_operators(name: str) -> list[str]:
    if name == "update_frequency":
        return ["ts_std_dev"]
    if name == "absolute_bound":
        return ["abs"]
    return []


def _parse_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw or "").split(","):
        try:
            value = int(part.strip())
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return list(dict.fromkeys(values)) or [5, 22, 66, 252]


def _parse_floats(raw: str) -> list[float]:
    values: list[float] = []
    for part in str(raw or "").split(","):
        try:
            value = float(part.strip())
        except ValueError:
            continue
        values.append(value)
    return list(dict.fromkeys(values)) or [0.0, 1.0, 10.0]


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(float(value))


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _universe_size(universe: str) -> int | None:
    digits = "".join(character for character in str(universe or "") if character.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None
