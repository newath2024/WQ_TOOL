from __future__ import annotations

from cli.app import main
from services.evaluation_service import build_alpha_simulation_signature

__all__ = ["build_alpha_simulation_signature", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
