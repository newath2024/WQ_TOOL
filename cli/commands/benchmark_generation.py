from __future__ import annotations

import argparse
import json

from core.config import AppConfig
from storage.repository import SQLiteRepository
from workflows.benchmark_generation import benchmark_generation


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "benchmark-generation",
        help="Benchmark baseline versus optimized local alpha generation.",
        parents=[common],
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override generation batch size used during the benchmark.",
    )
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment,
) -> int:
    result = benchmark_generation(
        repository,
        config,
        environment,
        count=args.count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
