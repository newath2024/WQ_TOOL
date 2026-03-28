from __future__ import annotations

import logging
from typing import Any


class RunLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = dict(self.extra)
        call_extra = kwargs.pop("extra", {})
        extra.update(call_extra)
        kwargs["extra"] = {
            "run_id": extra.get("run_id", "-"),
            "stage": extra.get("stage", "-"),
        }
        return msg, kwargs


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | run=%(run_id)s stage=%(stage)s | %(message)s",
    )


def get_logger(name: str, run_id: str = "-", stage: str = "-") -> RunLoggerAdapter:
    return RunLoggerAdapter(logging.getLogger(name), {"run_id": run_id, "stage": stage})
