from __future__ import annotations

import argparse

from core.config import AppConfig
from services.brain_service import BrainService
from services.models import CommandEnvironment
from services.runtime_service import init_run
from storage.repository import SQLiteRepository


def register(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "brain-login",
        help="Authenticate to the BRAIN API interactively and cache the session cookie locally.",
        parents=[common],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any saved session and force a fresh login.",
    )
    parser.add_argument(
        "--show-password",
        action="store_true",
        help="Show the password as you type instead of using hidden terminal input.",
    )
    parser.set_defaults(command_handler=handle)


def handle(
    args: argparse.Namespace,
    config: AppConfig,
    repository: SQLiteRepository,
    environment: CommandEnvironment,
) -> int:
    init_run(repository, config, environment, status="brain_authenticating")
    if config.brain.backend != "api":
        raise ValueError("brain-login requires `brain.backend: api` in the active config.")
    service = BrainService(repository, config.brain)
    adapter = service.adapter
    if not hasattr(adapter, "ensure_authenticated"):
        raise TypeError("The configured BRAIN adapter does not support interactive authentication.")
    result = adapter.ensure_authenticated(force=args.force, show_password=args.show_password)
    print(f"run_id: {environment.context.run_id}")
    print(f"backend: {config.brain.backend}")
    print(f"auth_mode: {result.get('mode', 'unknown')}")
    if result.get("session_path"):
        print(f"session_path: {result['session_path']}")
    repository.update_run_status(environment.context.run_id, "brain_authenticated")
    return 0
