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
            "tick_id": extra.get("tick_id", "-"),
            "batch_id": extra.get("batch_id", "-"),
            "job_id": extra.get("job_id", "-"),
        }
        return msg, kwargs


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=(
            "%(asctime)s | %(levelname)s | run=%(run_id)s stage=%(stage)s "
            "tick=%(tick_id)s batch=%(batch_id)s job=%(job_id)s | %(message)s"
        ),
    )


def get_logger(
    name: str,
    run_id: str = "-",
    stage: str = "-",
    tick_id: str | int = "-",
    batch_id: str = "-",
    job_id: str = "-",
) -> RunLoggerAdapter:
    return RunLoggerAdapter(
        logging.getLogger(name),
        {
            "run_id": run_id,
            "stage": stage,
            "tick_id": tick_id,
            "batch_id": batch_id,
            "job_id": job_id,
        },
    )
