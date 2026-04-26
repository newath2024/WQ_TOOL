from __future__ import annotations

import time

from core.logging import get_logger
from domain.simulation import SimulationResult
from services.models import ServiceTickOutcome
from services.progress_log import append_progress_event
from storage.models import (
    ServiceRuntimeRecord,
)

class BatchPoller:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def _poll_pending_batches(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        tick_id: int,
        batch_ids: list[str],
    ) -> ServiceTickOutcome:
        logger = get_logger(
            __name__,
            run_id=runtime.service_run_id,
            stage="service-poll",
            tick_id=tick_id,
        )
        poll_started = time.perf_counter()
        completed_results: list[SimulationResult] = []
        failed_results: list[SimulationResult] = []
        new_result_count = 0
        active_batch_id = batch_ids[0] if batch_ids else None

        for batch_id in batch_ids:
            refreshed = self.brain_service.poll_batch_once(
                batch_id,
                config=self.config,
                environment=self.environment,
                stuck_job_after_seconds=self.config.service.stuck_job_after_seconds,
            )
            logger.info(
                "Polled batch=%s status=%s pending=%s new_results=%s",
                batch_id,
                refreshed.status,
                refreshed.pending_count,
                len(refreshed.results),
            )
            new_result_count += len(refreshed.results)
            completed_count = sum(1 for result in refreshed.results if result.status == "completed")
            failed_count = sum(1 for result in refreshed.results if result.status in {"failed", "rejected", "timeout"})
            if refreshed.results or refreshed.status == "completed":
                append_progress_event(
                    self.config,
                    self.environment,
                    event="batch_polled",
                    stage="service-poll",
                    status=refreshed.status,
                    tick_id=tick_id,
                    round_index=refreshed.jobs[0].round_index if refreshed.jobs else None,
                    batch_id=batch_id,
                    payload={
                        "pending_count": refreshed.pending_count,
                        "new_result_count": len(refreshed.results),
                        "completed_count": completed_count,
                        "failed_count": failed_count,
                    },
                )
            for result in refreshed.results:
                if result.status == "completed":
                    completed_results.append(result)
                elif result.status in {"failed", "rejected", "timeout"}:
                    failed_results.append(result)
            if refreshed.status == "completed":
                self._learn_from_completed_batch(run_id=runtime.service_run_id, batch_id=batch_id)
            self._sync_service_round(
                run_id=runtime.service_run_id,
                batch_id=batch_id,
                status_override=refreshed.status,
            )

        pending_count = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
        next_active_batch = None
        if pending_count:
            next_pending = self.repository.submissions.list_pending_submissions(runtime.service_run_id)
            next_active_batch = next_pending[0].batch_id if next_pending else active_batch_id
        status = "running" if pending_count else "idle"
        return self._with_queue_metrics(
            ServiceTickOutcome(
                status=status,
                pending_job_count=pending_count,
                new_result_count=new_result_count,
                active_batch_id=next_active_batch,
                completed_count=len(completed_results),
                failed_count=len(failed_results),
                poll_pending_ms=(time.perf_counter() - poll_started) * 1000.0,
            ),
            service_name=runtime.service_name,
            run_id=runtime.service_run_id,
        )

