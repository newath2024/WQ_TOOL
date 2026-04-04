"""evolution-report: Rolling metrics CLI for tracking pipeline health over time."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.config import AppConfig
from core.logging import get_logger
from services.models import CommandEnvironment

if TYPE_CHECKING:
    from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "evolution-report",
        help="Display rolling evolution metrics for pipeline health.",
        parents=[common],
    )
    parser.add_argument(
        "--window",
        default="24h",
        help="Time window for aggregation: e.g. 1h, 6h, 24h, 7d (default: 24h).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        default=False,
        help="Aggregate across all runs, not just the current one.",
    )
    parser.set_defaults(command_handler=handle_evolution_report)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GenerationQuality:
    total_generated: int = 0
    valid_count: int = 0
    nesting_fail_count: int = 0
    duplicate_count: int = 0
    parse_fail_count: int = 0
    other_fail_count: int = 0
    avg_complexity: float = 0.0
    failure_reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class SimulationQuality:
    total_simulated: int = 0
    eligible_count: int = 0
    avg_sharpe: float = 0.0
    avg_fitness: float = 0.0
    avg_turnover: float = 0.0
    best_sharpe: float = 0.0
    rejection_reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class ProfileStats:
    name: str
    sim_count: int = 0
    eligible_count: int = 0
    avg_sharpe: float = 0.0


@dataclass
class MotifStats:
    name: str
    count: int = 0
    avg_outcome: float = 0.0


@dataclass
class EvolutionReport:
    window_label: str
    generation: GenerationQuality = field(default_factory=GenerationQuality)
    simulation: SimulationQuality = field(default_factory=SimulationQuality)
    profiles: list[ProfileStats] = field(default_factory=list)
    top_motifs: list[MotifStats] = field(default_factory=list)
    worst_motifs: list[MotifStats] = field(default_factory=list)
    motif_count: int = 0
    run_count: int = 0


# ---------------------------------------------------------------------------
# Time window parsing
# ---------------------------------------------------------------------------

_WINDOW_REGEX = re.compile(r"^(\d+)([hHdDmM])$")
_UNIT_SECONDS = {"h": 3600, "d": 86400, "m": 60}


def _parse_window_seconds(window: str) -> int:
    match = _WINDOW_REGEX.match(window.strip())
    if not match:
        raise ValueError(f"Invalid window format: '{window}'. Expected e.g. 1h, 6h, 24h, 7d.")
    value = int(match.group(1))
    unit = match.group(2).lower()
    return value * _UNIT_SECONDS[unit]


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _build_report(conn: sqlite3.Connection, window_seconds: int, run_id: str | None, window_label: str) -> EvolutionReport:
    report = EvolutionReport(window_label=window_label)

    # --- Generation quality from round_stage_metrics ---
    run_filter = "AND rsm.run_id = ?" if run_id else ""
    params: list[Any] = [window_seconds]
    if run_id:
        params.append(run_id)

    rows = conn.execute(
        f"""
        SELECT rsm.run_id, rsm.round_index, rsm.stage, rsm.metrics_json, rsm.created_at
        FROM round_stage_metrics rsm
        WHERE rsm.created_at >= datetime('now', '-' || ? || ' seconds')
          {run_filter}
        ORDER BY rsm.created_at
        """,
        params,
    ).fetchall()

    run_ids = set()
    total_generated = 0
    total_valid = 0
    total_nesting = 0
    total_dup = 0
    total_parse = 0
    total_complexity = 0.0
    complexity_count = 0
    all_failure_reasons: Counter[str] = Counter()

    for row in rows:
        stage = row[2]
        run_ids.add(row[0])
        if stage not in ("pre_sim", "generation"):
            continue
        try:
            m = json.loads(row[3])
        except (json.JSONDecodeError, TypeError):
            continue

        generated = int(m.get("generated", 0) or m.get("total_attempts", 0) or 0)
        valid = int(m.get("valid_candidates", 0) or m.get("kept_after_dedup", 0) or 0)
        total_generated += generated
        total_valid += valid

        frc = m.get("failure_reason_counts", {})
        for reason, count in frc.items():
            all_failure_reasons[reason] += int(count)

        total_nesting += int(frc.get("validation_invalid_nesting", 0))
        total_dup += int(frc.get("duplicate_normalized_expression", 0)) + int(frc.get("structural_duplicate_expression", 0))
        total_parse += int(frc.get("parse_failed", 0))

        avg_cplx = m.get("avg_complexity")
        if avg_cplx is not None:
            total_complexity += float(avg_cplx)
            complexity_count += 1

    report.run_count = len(run_ids)
    report.generation.total_generated = total_generated
    report.generation.valid_count = total_valid
    report.generation.nesting_fail_count = total_nesting
    report.generation.duplicate_count = total_dup
    report.generation.parse_fail_count = total_parse
    report.generation.other_fail_count = max(0, total_generated - total_valid - total_nesting - total_dup - total_parse)
    report.generation.avg_complexity = (total_complexity / complexity_count) if complexity_count > 0 else 0.0
    report.generation.failure_reasons = dict(all_failure_reasons.most_common(10))

    # --- Simulation results from brain_results ---
    sim_params: list[Any] = [window_seconds]
    sim_run_filter = "AND br.run_id = ?" if run_id else ""
    if run_id:
        sim_params.append(run_id)

    sim_rows = conn.execute(
        f"""
        SELECT
            br.status,
            br.sharpe,
            br.fitness,
            br.turnover,
            br.submission_eligible,
            br.rejection_reason,
            sb.sim_config_snapshot
        FROM brain_results br
        LEFT JOIN submission_batches sb ON br.batch_id = sb.batch_id
        WHERE br.created_at >= datetime('now', '-' || ? || ' seconds')
          {sim_run_filter}
        """,
        sim_params,
    ).fetchall()

    eligible_sharpes: list[float] = []
    eligible_fitnesses: list[float] = []
    eligible_turnovers: list[float] = []
    rejection_counter: Counter[str] = Counter()
    profile_data: dict[str, dict[str, Any]] = {}
    best_sharpe = -999.0

    for row in sim_rows:
        report.simulation.total_simulated += 1
        status = row[0]
        sharpe = row[1]
        fitness = row[2]
        turnover = row[3]
        eligible = row[4]
        rejection = row[5]
        config_snapshot = row[6]

        if eligible:
            report.simulation.eligible_count += 1
            if sharpe is not None:
                eligible_sharpes.append(float(sharpe))
            if fitness is not None:
                eligible_fitnesses.append(float(fitness))
            if turnover is not None:
                eligible_turnovers.append(float(turnover))
        elif rejection:
            rejection_counter[str(rejection)] += 1

        if sharpe is not None:
            best_sharpe = max(best_sharpe, float(sharpe))

        # Profile breakdown
        profile_name = "default"
        if config_snapshot:
            try:
                snap = json.loads(config_snapshot) if isinstance(config_snapshot, str) else config_snapshot
                profile_name = str(snap.get("simulation_profile", "default"))
            except (json.JSONDecodeError, TypeError):
                pass
        if profile_name not in profile_data:
            profile_data[profile_name] = {"count": 0, "eligible": 0, "sharpes": []}
        profile_data[profile_name]["count"] += 1
        if eligible:
            profile_data[profile_name]["eligible"] += 1
            if sharpe is not None:
                profile_data[profile_name]["sharpes"].append(float(sharpe))

    report.simulation.avg_sharpe = (sum(eligible_sharpes) / len(eligible_sharpes)) if eligible_sharpes else 0.0
    report.simulation.avg_fitness = (sum(eligible_fitnesses) / len(eligible_fitnesses)) if eligible_fitnesses else 0.0
    report.simulation.avg_turnover = (sum(eligible_turnovers) / len(eligible_turnovers)) if eligible_turnovers else 0.0
    report.simulation.best_sharpe = best_sharpe if best_sharpe > -999 else 0.0
    report.simulation.rejection_reasons = dict(rejection_counter.most_common(10))

    for pname, pdata in sorted(profile_data.items()):
        avg_s = (sum(pdata["sharpes"]) / len(pdata["sharpes"])) if pdata["sharpes"] else 0.0
        report.profiles.append(ProfileStats(name=pname, sim_count=pdata["count"], eligible_count=pdata["eligible"], avg_sharpe=avg_s))

    # --- Motif stats from alpha_cases ---
    motif_params: list[Any] = [window_seconds]
    motif_run_filter = "AND ac.run_id = ?" if run_id else ""
    if run_id:
        motif_params.append(run_id)

    motif_rows = conn.execute(
        f"""
        SELECT ac.motif, COUNT(*) as cnt, AVG(ac.outcome_score) as avg_out
        FROM alpha_cases ac
        WHERE ac.created_at >= datetime('now', '-' || ? || ' seconds')
          AND ac.motif <> ''
          {motif_run_filter}
        GROUP BY ac.motif
        ORDER BY avg_out DESC
        """,
        motif_params,
    ).fetchall()

    all_motifs = [MotifStats(name=r[0], count=int(r[1]), avg_outcome=float(r[2] or 0)) for r in motif_rows]
    report.motif_count = len(all_motifs)
    report.top_motifs = all_motifs[:3]
    report.worst_motifs = [m for m in reversed(all_motifs) if m.avg_outcome < 0][:3]

    return report


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "—"
    return f"{num / denom * 100:.1f}%"


def _format_report(report: EvolutionReport) -> str:
    lines: list[str] = []
    w = 52
    g = report.generation
    s = report.simulation

    lines.append("╔" + "═" * w + "╗")
    lines.append(f"║{'Evolution Report (' + report.window_label + ')':^{w}}║")
    lines.append("╠" + "═" * w + "╣")

    # Generation
    lines.append(f"║{'GENERATION QUALITY':<{w}}║")
    lines.append(f"║  {'Total generated:':<28}{g.total_generated:>{w - 30}}║")
    lines.append(f"║  {'Valid rate:':<28}{_pct(g.valid_count, g.total_generated) + f' ({g.valid_count})':>{w - 30}}║")
    lines.append(f"║  {'Nesting fail:':<28}{_pct(g.nesting_fail_count, g.total_generated) + f' ({g.nesting_fail_count})':>{w - 30}}║")
    lines.append(f"║  {'Duplicate rate:':<28}{_pct(g.duplicate_count, g.total_generated) + f' ({g.duplicate_count})':>{w - 30}}║")
    lines.append(f"║  {'Parse fail:':<28}{_pct(g.parse_fail_count, g.total_generated) + f' ({g.parse_fail_count})':>{w - 30}}║")
    if g.avg_complexity > 0:
        lines.append(f"║  {'Avg complexity:':<28}{g.avg_complexity:>{w - 30}.1f}║")

    if g.failure_reasons:
        lines.append("╠" + "─" * w + "╣")
        lines.append(f"║{'FAILURE BREAKDOWN':<{w}}║")
        for reason, count in list(g.failure_reasons.items())[:6]:
            short_reason = reason[:30]
            lines.append(f"║  {short_reason:<30}{count:>{w - 32}}║")

    # Simulation
    lines.append("╠" + "═" * w + "╣")
    lines.append(f"║{'SIMULATION RESULTS':<{w}}║")
    if s.total_simulated == 0:
        lines.append(f"║  {'No simulation data in window.':<{w}}║")
    else:
        lines.append(f"║  {'Simulated:':<28}{s.total_simulated:>{w - 30}}║")
        lines.append(f"║  {'Eligible rate:':<28}{_pct(s.eligible_count, s.total_simulated) + f' ({s.eligible_count})':>{w - 30}}║")
        if s.avg_sharpe != 0:
            lines.append(f"║  {'Avg sharpe (eligible):':<28}{s.avg_sharpe:>{w - 30}.3f}║")
        if s.avg_fitness != 0:
            lines.append(f"║  {'Avg fitness (eligible):':<28}{s.avg_fitness:>{w - 30}.3f}║")
        if s.avg_turnover != 0:
            lines.append(f"║  {'Avg turnover (eligible):':<28}{s.avg_turnover:>{w - 30}.3f}║")
        if s.best_sharpe > 0:
            lines.append(f"║  {'Best sharpe:':<28}{s.best_sharpe:>{w - 30}.3f}║")
        if s.rejection_reasons:
            lines.append("╠" + "─" * w + "╣")
            lines.append(f"║{'REJECTION REASONS':<{w}}║")
            for reason, count in list(s.rejection_reasons.items())[:5]:
                short_r = reason[:30]
                lines.append(f"║  {short_r:<30}{count:>{w - 32}}║")

    # Profile comparison
    if report.profiles:
        lines.append("╠" + "═" * w + "╣")
        lines.append(f"║{'PROFILE COMPARISON':<{w}}║")
        for p in report.profiles:
            rate = _pct(p.eligible_count, p.sim_count)
            sharpe_str = f"sharpe={p.avg_sharpe:.2f}" if p.avg_sharpe else ""
            label = f'"{p.name}"'
            detail = f"{p.sim_count} sim, {rate} ✓ {sharpe_str}"
            lines.append(f"║  {label:<16}{detail:>{w - 18}}║")

    # Motifs
    if report.top_motifs or report.worst_motifs:
        lines.append("╠" + "═" * w + "╣")
        if report.top_motifs:
            lines.append(f"║{'TOP MOTIFS':<{w}}║")
            for m in report.top_motifs:
                detail = f"avg_out={m.avg_outcome:+.3f}  n={m.count}"
                lines.append(f"║  {m.name:<22}{detail:>{w - 24}}║")
        if report.worst_motifs:
            lines.append(f"║{'WORST MOTIFS':<{w}}║")
            for m in report.worst_motifs:
                detail = f"avg_out={m.avg_outcome:+.3f}  n={m.count}"
                lines.append(f"║  {m.name:<22}{detail:>{w - 24}}║")

    lines.append("╠" + "═" * w + "╣")
    lines.append(f"║  {'Runs in window:':<28}{report.run_count:>{w - 30}}║")
    lines.append(f"║  {'Distinct motifs:':<28}{report.motif_count:>{w - 30}}║")
    lines.append("╚" + "═" * w + "╝")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handle_evolution_report(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    logger = get_logger(__name__, run_id=environment.context.run_id, stage="evolution-report")
    window_label = args.window.strip()
    try:
        window_seconds = _parse_window_seconds(window_label)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    run_id = None if args.all_runs else environment.context.run_id
    conn = repository.connection

    report = _build_report(conn, window_seconds, run_id, window_label)

    if report.generation.total_generated == 0 and report.simulation.total_simulated == 0:
        print(f"No data in window ({window_label}).")
        return 0

    print(_format_report(report))
    return 0
