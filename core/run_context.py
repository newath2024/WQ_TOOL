from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


@dataclass(slots=True)
class RunContext:
    run_id: str
    seed: int
    started_at: str
    config_path: str

    @classmethod
    def create(cls, seed: int, config_path: str | Path, run_id: str | None = None) -> "RunContext":
        return cls(
            run_id=run_id or uuid4().hex[:12],
            seed=seed,
            started_at=datetime.now(timezone.utc).isoformat(),
            config_path=str(config_path),
        )

