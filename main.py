from __future__ import annotations

import sys
from typing import Any, Sequence

_MIN_PYTHON = (3, 11)
_RUNTIME_DEPENDENCIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "yaml": "PyYAML",
}


def _recommended_setup() -> str:
    return (
        "Recommended setup on Windows PowerShell:\n"
        "  py -3.12 -m venv .venv\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  python -m pip install -U pip\n"
        '  python -m pip install -e ".[dev]"'
    )


def _unsupported_python_message() -> str:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return (
        "WQ Tool requires Python 3.11 or newer.\n"
        f"Current interpreter: {sys.executable}\n"
        f"Current version: {version}\n\n"
        f"{_recommended_setup()}\n"
    )


def _missing_dependency_message(module_name: str) -> str:
    package_name = _RUNTIME_DEPENDENCIES.get(module_name, module_name)
    return (
        "WQ Tool runtime dependencies are not installed for the current interpreter.\n"
        f"Current interpreter: {sys.executable}\n"
        f"Missing module: {module_name} (install package: {package_name})\n\n"
        "This usually means VS Code is using the wrong Python interpreter.\n"
        "Select Python 3.12 or your project .venv in 'Python: Select Interpreter'.\n\n"
        f"{_recommended_setup()}\n"
        "Or install into the current interpreter:\n"
        f"  \"{sys.executable}\" -m pip install -e \".[dev]\"\n"
    )


def _bootstrap_failure_main(argv: Sequence[str] | None = None) -> int:
    del argv
    assert _BOOTSTRAP_ERROR is not None
    sys.stderr.write(_BOOTSTRAP_ERROR)
    if not _BOOTSTRAP_ERROR.endswith("\n"):
        sys.stderr.write("\n")
    return 2


_BOOTSTRAP_ERROR: str | None = None

if sys.version_info < _MIN_PYTHON:
    _BOOTSTRAP_ERROR = _unsupported_python_message()
else:
    try:
        from cli.app import main as main
        from services.evaluation_service import (
            build_alpha_simulation_signature as build_alpha_simulation_signature,
        )
    except ModuleNotFoundError as exc:
        if exc.name in _RUNTIME_DEPENDENCIES:
            _BOOTSTRAP_ERROR = _missing_dependency_message(exc.name)
        else:
            raise

if _BOOTSTRAP_ERROR is not None:

    def main(argv: Sequence[str] | None = None) -> int:
        return _bootstrap_failure_main(argv)

    def build_alpha_simulation_signature(*args: Any, **kwargs: Any) -> str:
        del args, kwargs
        raise RuntimeError(_BOOTSTRAP_ERROR)


__all__ = ["build_alpha_simulation_signature", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
