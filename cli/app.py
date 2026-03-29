from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from cli.commands import evaluate, generate, lineage, load_data, memory, mutate, pipeline, report
from core.config import load_config
from core.logging import configure_logging
from services.runtime_service import build_command_environment, open_repository, resolve_run_context


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
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entrypoint used by both `python main.py` and console scripts."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
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
