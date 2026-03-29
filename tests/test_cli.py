from __future__ import annotations

from cli.app import _normalize_argv, build_parser
from main import build_alpha_simulation_signature, main


def test_main_module_is_thin_wrapper() -> None:
    assert main.__module__ == "cli.app"
    assert build_alpha_simulation_signature.__module__ == "services.evaluation_service"


def test_parser_dispatches_to_command_modules() -> None:
    parser = build_parser()
    args = parser.parse_args(["report", "--limit", "3"])

    assert args.command == "report"
    assert args.command_handler.__module__ == "cli.commands.report"


def test_parser_registers_new_brain_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-closed-loop"])

    assert args.command == "run-closed-loop"
    assert args.command_handler.__module__ == "cli.commands.run_closed_loop"
    service_args = parser.parse_args(["run-service"])
    assert service_args.command == "run-service"
    assert service_args.command_handler.__module__ == "cli.commands.run_service"
    service_status_args = parser.parse_args(["service-status"])
    assert service_status_args.command == "service-status"
    assert service_status_args.command_handler.__module__ == "cli.commands.service_status"
    login_args = parser.parse_args(["brain-login"])
    assert login_args.command == "brain-login"
    login_visible_args = parser.parse_args(["brain-login", "--show-password"])
    assert login_visible_args.command == "brain-login"
    assert login_visible_args.show_password is True
    assert login_args.command_handler.__module__ == "cli.commands.brain_login"


def test_missing_command_defaults_to_pipeline() -> None:
    assert _normalize_argv([]) == ["run-full-pipeline"]
    assert _normalize_argv(["--config", "config/dev.yaml"]) == [
        "--config",
        "config/dev.yaml",
        "run-full-pipeline",
    ]
    assert _normalize_argv(["--config", "config/dev.yaml", "--count", "8"]) == [
        "--config",
        "config/dev.yaml",
        "run-full-pipeline",
        "--count",
        "8",
    ]
