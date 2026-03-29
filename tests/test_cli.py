from __future__ import annotations

from cli.app import build_parser
from main import build_alpha_simulation_signature, main


def test_main_module_is_thin_wrapper() -> None:
    assert main.__module__ == "cli.app"
    assert build_alpha_simulation_signature.__module__ == "services.evaluation_service"


def test_parser_dispatches_to_command_modules() -> None:
    parser = build_parser()
    args = parser.parse_args(["report", "--limit", "3"])

    assert args.command == "report"
    assert args.command_handler.__module__ == "cli.commands.report"
