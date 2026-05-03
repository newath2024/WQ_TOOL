from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

from cli.commands import (
    backfill_brain_checks,
    benchmark_generation,
    brain_login,
    diagnose_fields,
    evaluate,
    evolution_report,
    export_brain_candidates,
    generate,
    import_brain_results,
    lineage,
    load_data,
    memory,
    mutate,
    pipeline,
    recover_brain_jobs,
    report,
    run_brain_sim,
    run_closed_loop,
    run_service,
    service_status,
    sync_field_catalog,
)
from core.config import load_config
from core.logging import configure_logging
from services.runtime_service import build_command_environment, open_repository, resolve_run_context

_SUBCOMMANDS = {
    "load-data",
    "generate",
    "evaluate",
    "top",
    "report",
    "evolution-report",
    "memory-top-patterns",
    "memory-failed-patterns",
    "memory-top-genes",
    "lineage",
    "mutate",
    "run-full-pipeline",
    "sync-field-catalog",
    "export-brain-candidates",
    "import-brain-results",
    "run-brain-sim",
    "run-closed-loop",
    "run-service",
    "service-status",
    "recover-brain-jobs",
    "brain-login",
    "benchmark-generation",
    "backfill-brain-checks",
    "diagnose-fields",
}
_GLOBAL_OPTIONS_WITH_VALUES = {"--config", "--run-id", "--seed", "--log-level"}
_GLOBAL_FLAG_OPTIONS = {"--resume"}


def _add_common_arguments(parser: argparse.ArgumentParser, suppress_defaults: bool) -> None:
    default = argparse.SUPPRESS if suppress_defaults else "config/default.yaml"
    parser.add_argument("--config", default=default, help="Path to the YAML config file.")
    parser.add_argument(
        "--run-id",
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="Run identifier to reuse or inspect.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="Reuse the latest run when no run-id is provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="Override the generation seed.",
    )
    parser.add_argument(
        "--log-level",
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="Override the configured log level.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    common = argparse.ArgumentParser(add_help=False)
    _add_common_arguments(common, suppress_defaults=True)

    parser = argparse.ArgumentParser(description="Automated alpha generation research platform")
    _add_common_arguments(parser, suppress_defaults=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    load_data.register(subparsers, common)
    generate.register(subparsers, common)
    evaluate.register(subparsers, common)
    report.register(subparsers, common)
    memory.register(subparsers, common)
    lineage.register(subparsers, common)
    mutate.register(subparsers, common)
    pipeline.register(subparsers, common)
    sync_field_catalog.register(subparsers, common)
    export_brain_candidates.register(subparsers, common)
    import_brain_results.register(subparsers, common)
    run_brain_sim.register(subparsers, common)
    run_closed_loop.register(subparsers, common)
    run_service.register(subparsers, common)
    service_status.register(subparsers, common)
    recover_brain_jobs.register(subparsers, common)
    brain_login.register(subparsers, common)
    benchmark_generation.register(subparsers, common)
    backfill_brain_checks.register(subparsers, common)
    diagnose_fields.register(subparsers, common)
    evolution_report.register(subparsers, common)
    return parser


def _normalize_argv(argv: Iterable[str] | None) -> list[str]:
    """Inject a default pipeline command when the user omits a subcommand."""
    tokens = list(argv) if argv is not None else sys.argv[1:]
    if not tokens:
        print("No command provided. Defaulting to 'run-full-pipeline'.", file=sys.stderr)
        return ["run-full-pipeline"]
    if any(token in _SUBCOMMANDS for token in tokens):
        return tokens
    if any(token in {"-h", "--help"} for token in tokens):
        return tokens

    prefix: list[str] = []
    remainder: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _GLOBAL_OPTIONS_WITH_VALUES:
            prefix.append(token)
            if index + 1 < len(tokens):
                prefix.append(tokens[index + 1])
                index += 2
                continue
            remainder = tokens[index:]
            break
        if token in _GLOBAL_FLAG_OPTIONS:
            prefix.append(token)
            index += 1
            continue
        remainder = tokens[index:]
        break
    else:
        remainder = []

    print("No command provided. Defaulting to 'run-full-pipeline'.", file=sys.stderr)
    return [*prefix, "run-full-pipeline", *remainder]


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entrypoint used by both `python main.py` and console scripts."""
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    config = load_config(args.config)
    log_level = args.log_level or config.runtime.log_level
    configure_logging(log_level)

    seed = args.seed if args.seed is not None else config.generation.random_seed
    repository = open_repository(config)
    try:
        context = resolve_run_context(
            repository=repository,
            config_path=str(Path(args.config).resolve()),
            seed=seed,
            run_id=args.run_id,
            resume=args.resume,
            command=args.command,
        )
        environment = build_command_environment(
            config_path=str(Path(args.config).resolve()),
            command_name=args.command,
            context=context,
        )
        return args.command_handler(args, config, repository, environment)
    finally:
        repository.close()
