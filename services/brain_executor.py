"""
Brain Executor — thay thế logic submit/poll/pool của BrainService bằng
SimulationExecutor + SimulationQueue từ brain_client/.

Triết lý:
  - Giữ nguyên BrainApiAdapter cho authentication + submit (entry-point ổ định).
  - Thay thế polling loop và batch-pooling bằng executor.queue và executor.poll_results.
  - Kết quả được persist vào repository ngay khi từng simulation hoàn thành
    (on_result callback), không cần chờ cả batch.

Auth: KHÔNG thay đổi — vẫn dùng BrainApiAdapter.ensure_authenticated().
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from brain_client.client import BrainClient, BrainConfig, SimulationResult as _ClientResult
from brain_client.config import ExecutorConfig
from brain_client.executor import SimulationExecutor as _SimExecutor
from brain_client.queue import PendingSimulation, SimulationQueue
from core.config import AppConfig, BrainConfig as AppBrainConfig
from core.logging import get_logger
from services.models import (
    BrainSimulationBatch,
    CommandEnvironment,
    SimulationJob,
    SimulationResult,
)
from storage.models import BrainResultRecord, SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository

if TYPE_CHECKING:
    from adapters.brain_api_adapter import BrainApiAdapter

logger = logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "rejected", "timeout", "COMPLETE", "ERROR"}
PENDING_STATUSES = {"submitted", "running", "SUBMITTED", "POLLING"}


# ---------------------------------------------------------------------------
# ProgressUrlBrainAdapter: implement interface giống BrainApiAdapter nhưng
# dùng progress_url pattern từ BrainClient (submit → poll progress → get alpha).
# Auth vẫn delegate sang BrainApiAdapter đã được khởi tạo bên ngoài.
# ---------------------------------------------------------------------------


class _BrainClientProtocol(Protocol):
    """Subset của BrainClient mà ProgressUrlBrainAdapter cần."""

    def submit_simulation(self, expression: str, region: str, dataset_id: str,
                         universe: str, neutralization: str,
                         delay: int, decay: int) -> str:
        """Submit, trả về progress_url (Location header)."""
        ...

    def poll_simulation(self, progress_url: str, max_retries: int,
                        poll_interval: int) -> Optional[str]:
        """Poll progress_url, trả về alpha_id khi COMPLETE/WARNING."""
        ...

    def get_alpha_results(self, alpha_id: str) -> dict:
        """Lấy metrics từ alpha_id."""
        ...


class ProgressUrlBrainAdapter:
    """
    Wrapper quanh BrainClient — implement interface tương thích với
    BrainApiAdapter để BrainService có thể dùng mà không cần sửa caller.

    Auth được delegate sang BrainApiAdapter thật (không thay đổi).

    Interface tương thích:
      - submit_simulation(expression, sim_config) → job dict với job_id
      - get_simulation_status(job_id) → status dict
      - get_simulation_result(job_id) → result dict
    """

    def __init__(
        self,
        brain_client: _BrainClientProtocol,
        auth_adapter: "BrainApiAdapter | None" = None,
    ) -> None:
        self._client = brain_client
        self._auth_adapter = auth_adapter
        # Map job_id → progress_url (để poll)
        self._progress_urls: dict[str, str] = {}
        self._job_locks: dict[str, Lock] = {}
        self._lock = Lock()

    # ---- BrainApiAdapter-compatible interface --------------------------------

    def submit_simulation(self, expression: str, sim_config: dict) -> dict:
        """Submit alpha, trả về dict với job_id + progress_url lưu lại."""
        if self._auth_adapter:
            self._auth_adapter.ensure_authenticated()

        region = str(sim_config.get("region", "USA")).upper()
        universe = str(sim_config.get("universe", "TOP3000")).upper()
        neutralization = str(sim_config.get("neutralization", "SUBINDUSTRY")).upper()
        delay = int(sim_config.get("delay", 1))
        decay = int(sim_config.get("decay", 0))

        # Lấy progress_url (Location header)
        progress_url = self._client.submit_simulation(
            expression=expression,
            region=region,
            dataset_id="DEFAULT",
            universe=universe,
            neutralization=neutralization,
            delay=delay,
            decay=decay,
        )

        # Sinh job_id từ progress_url
        job_id = progress_url.rstrip("/").split("/")[-1]

        with self._lock:
            self._progress_urls[job_id] = progress_url
            self._job_locks.setdefault(job_id, Lock())

        return {
            "job_id": job_id,
            "progress_url": progress_url,
            "expression": expression,
            "status": "submitted",
            "raw_submission": {"progress_url": progress_url, "region": region,
                               "universe": universe, "neutralization": neutralization},
        }

    def get_simulation_status(self, job_id: str) -> dict:
        """
        Poll trạng thái simulation.
        - Trả về {'job_id': ..., 'status': 'completed', 'alpha_id': ...} nếu xong
        - Trả về {'job_id': ..., 'status': 'running'} nếu đang chạy
        - Raise TimeoutError nếu hết retries
        """
        if self._auth_adapter:
            self._auth_adapter.ensure_authenticated()

        progress_url = self._progress_urls.get(job_id)
        if not progress_url:
            raise ValueError(f"Unknown job_id: {job_id}")

        with self._job_locks.get(job_id, self._lock):
            try:
                alpha_id = self._client.poll_simulation(
                    progress_url,
                    max_retries=60,
                    poll_interval=10,
                )
            except TimeoutError:
                return {"job_id": job_id, "status": "running", "retry_after": 10.0}

        if alpha_id:
            return {
                "job_id": job_id,
                "status": "completed",
                "alpha_id": alpha_id,
                "progress_url": progress_url,
            }
        return {"job_id": job_id, "status": "running", "retry_after": 10.0}

    def get_simulation_result(self, job_id: str) -> dict:
        """Lấy metrics chi tiết từ alpha_id sau khi COMPLETE."""
        if self._auth_adapter:
            self._auth_adapter.ensure_authenticated()

        progress_url = self._progress_urls.get(job_id)
        status_resp = self.get_simulation_status(job_id)
        alpha_id = status_resp.get("alpha_id")

        if not alpha_id:
            return {
                "job_id": job_id,
                "status": status_resp.get("status", "running"),
                "metrics": {},
                "raw_result": {},
            }

        metrics = self._client.get_alpha_results(alpha_id)

        return {
            "job_id": job_id,
            "status": "completed",
            "alpha_id": alpha_id,
            "metrics": metrics,
            "raw_result": {
                "simulation": {"alpha": alpha_id, "status": "COMPLETE"},
                "alpha": metrics,
                "recordsets": {},
            },
            "submission_eligible": None,
            "rejection_reason": None,
        }


# ---------------------------------------------------------------------------
# BrainExecutor — high-level service layer
# ---------------------------------------------------------------------------

@dataclass
class _ExecutorConfig:
    """Config cho BrainExecutor, đọc từ AppBrainConfig."""
    max_concurrent_simulations: int = 5
    polling_interval: int = 30
    simulation_timeout: int = 600
    min_wait_time: int = 300


@dataclass
class _JobHandle:
    """Internal handle cho một đang chạy simulation."""
    job_id: str
    progress_url: str
    expression: str
    candidate_id: str
    sim_config: dict
    submitted_at: str
    poll_count: int = 0
    status: str = "submitted"


class BrainExecutor:
    """
    Thay thế BrainService cho submit/polling/pooling.

    Dùng:
      - BrainApiAdapter (giữ nguyên auth, không thay đổi)
      - ProgressUrlBrainAdapter (dùng progress_url pattern)
      - SimulationQueue từ brain_client (thread-safe, FIFO, concurrent limit)
      - pool_batch_alphas / SimulationExecutor pattern cho batch pooling

    KHÔNG thay đổi auth logic — BrainApiAdapter.ensure_authenticated() được giữ y nguyên.

    Persistence:
      Kết quả được lưu vào repository ngay khi từng simulation hoàn thành
      (on_result callback), thông qua _persist_result.
    """

    def __init__(
        self,
        repository: SQLiteRepository,
        brain_config: AppBrainConfig,
        auth_adapter: "BrainApiAdapter",
        *,
        max_concurrent: int | None = None,
        polling_interval: int | None = None,
        simulation_timeout: int | None = None,
        min_wait_time: int | None = None,
    ) -> None:
        self.repository = repository
        self.brain_config = brain_config
        self._auth_adapter = auth_adapter

        # Đọc config từ AppBrainConfig hoặc override
        self._exec_config = _ExecutorConfig(
            max_concurrent_simulations=(
                max_concurrent
                if max_concurrent is not None
                else getattr(brain_config, "max_concurrent_simulations", 5)
            ),
            polling_interval=(
                polling_interval
                if polling_interval is not None
                else getattr(brain_config, "polling_interval", 30)
            ),
            simulation_timeout=(
                simulation_timeout
                if simulation_timeout is not None
                else getattr(brain_config, "simulation_timeout", 600)
            ),
            min_wait_time=(
                min_wait_time
                if min_wait_time is not None
                else getattr(brain_config, "min_wait_time", 300)
            ),
        )

        # Khởi tạo BrainClient + ProgressUrl adapter
        self._brain_client = self._build_brain_client(brain_config)
        self._progress_adapter = ProgressUrlBrainAdapter(
            brain_client=self._brain_client,
            auth_adapter=auth_adapter,
        )

        # Queue quản lý concurrent simulations
        self._queue = SimulationQueue(
            max_size=self._exec_config.max_concurrent_simulations
        )

        # Map job_id → _JobHandle
        self._handles: dict[str, _JobHandle] = {}
        self._handles_lock = Lock()

        logger.info(
            "[BrainExecutor] Initialized | max_concurrent=%d | "
            "polling_interval=%ds | timeout=%ds | min_wait=%ds",
            self._exec_config.max_concurrent_simulations,
            self._exec_config.polling_interval,
            self._exec_config.simulation_timeout,
            self._exec_config.min_wait_time,
        )

    # -------------------------------------------------------------------------
    # Public API — tương thích với BrainService.simulate_candidates
    # -------------------------------------------------------------------------

    def simulate_candidates(
        self,
        candidates: list,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        round_index: int = 0,
        batch_size: int | None = None,
    ) -> BrainSimulationBatch:
        """
        Submit nhiều alpha, quản lý concurrent simulation bằng queue,
        poll và persist kết quả, trả về BrainSimulationBatch.

        Đây là entry point chính — tương đương BrainService.simulate_candidates
        nhưng dùng executor.queue thay vì SQLite polling.
        """
        batch = self.submit_candidates(
            candidates,
            config=config,
            environment=environment,
            round_index=round_index,
            batch_size=batch_size,
        )
        return self.poll_batch(
            batch,
            config=config,
            environment=environment,
        )

    def submit_candidates(
        self,
        candidates: list,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        round_index: int = 0,
        batch_size: int | None = None,
    ) -> BrainSimulationBatch:
        """Submit candidates vào queue. KHÔNG block — trả về ngay."""
        from generator.engine import AlphaCandidate

        selected: list[AlphaCandidate] = list(
            candidates[: (batch_size or self._exec_config.max_concurrent_simulations * 3)]
        )
        if not selected:
            return BrainSimulationBatch(
                batch_id=f"brain-{environment.context.run_id[:8]}-empty",
                backend=self.brain_config.backend,
                status="empty",
                jobs=(),
                results=(),
            )

        batch_id = f"brain-{environment.context.run_id[:8]}-r{round_index:02d}-{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(UTC).isoformat()

        # Build sim_config chuẩn
        sim_config = self._build_sim_config(
            config=config,
            environment=environment,
            round_index=round_index,
            batch_id=batch_id,
            candidates=selected,
        )

        # Lưu batch record
        self._save_batch_record(
            batch_id=batch_id,
            run_id=environment.context.run_id,
            round_index=round_index,
            candidate_count=len(selected),
            sim_config=sim_config,
            timestamp=timestamp,
        )

        jobs: list[SimulationJob] = []

        for candidate in selected:
            try:
                handle, job = self._submit_single(
                    candidate=candidate,
                    batch_id=batch_id,
                    run_id=environment.context.run_id,
                    round_index=round_index,
                    sim_config=sim_config,
                    timestamp=timestamp,
                )
                jobs.append(job)
            except Exception as exc:
                logger.warning(
                    "[BrainExecutor] Failed to submit %s: %s",
                    candidate.alpha_id,
                    exc,
                )

        batch_status = "submitted" if jobs else "failed"
        self._update_batch_status(
            batch_id=batch_id,
            status=batch_status,
            updated_at=datetime.now(UTC).isoformat(),
        )

        logger.info(
            "[BrainExecutor] Submitted %d candidates batch=%s",
            len(jobs),
            batch_id,
        )

        return BrainSimulationBatch(
            batch_id=batch_id,
            backend=self.brain_config.backend,
            status=batch_status,
            jobs=tuple(jobs),
            results=(),
        )

    def poll_batch(
        self,
        batch: BrainSimulationBatch,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
    ) -> BrainSimulationBatch:
        """
        Poll batch đang chạy cho đến khi tất cả hoàn thành hoặc timeout.

        Dùng queue-based polling: lấy pending từ queue, poll từng cái,
        persist ngay khi hoàn thành.
        """
        deadline = time.monotonic() + float(
            getattr(config.loop, "timeout_seconds", self._exec_config.simulation_timeout)
        )
        all_results: dict[str, SimulationResult] = {}

        # Load lại batch từ DB để có đầy đủ jobs
        current = self._load_batch(batch.batch_id) or batch

        while time.monotonic() < deadline:
            current = self._poll_batch_once(
                current.batch_id,
                config=config,
                environment=environment,
            )
            for result in current.results:
                all_results[result.job_id] = result

            if current.status in {"completed", "manual_pending"} or current.pending_count == 0:
                return BrainSimulationBatch(
                    batch_id=current.batch_id,
                    backend=current.backend,
                    status=current.status,
                    jobs=current.jobs,
                    results=tuple(all_results.values()),
                    export_path=current.export_path,
                )

            time.sleep(float(config.loop.poll_interval_seconds))

        # Timeout — đánh dấu các job còn pending
        timeout_results = self._timeout_pending_jobs(
            current.batch_id,
            reason="poll_timeout_live",
        )
        for result in timeout_results:
            all_results[result.job_id] = result

        refreshed = self._load_batch(current.batch_id) or current
        return BrainSimulationBatch(
            batch_id=refreshed.batch_id,
            backend=refreshed.backend,
            status=refreshed.status,
            jobs=refreshed.jobs,
            results=tuple(all_results.values()),
            export_path=refreshed.export_path,
        )

    # -------------------------------------------------------------------------
    # Pool batch alphas — high-level entry point giống brain_client
    # -------------------------------------------------------------------------

    def pool_batch(
        self,
        alphas: list[str],
        *,
        region: str = "USA",
        universe: str = "TOP3000",
        neutralization: str = "SUBINDUSTRY",
        delay: int = 1,
        decay: int = 0,
        max_concurrent: int | None = None,
        on_result: Callable[[str, _ClientResult], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        timeout: int = 600,
        repository: SQLiteRepository | None = None,
        batch_id: str | None = None,
        run_id: str | None = None,
        round_index: int = 0,
        min_wait_time: int = 300,
    ) -> dict[str, _ClientResult]:
        """
        Submit N alphas, duy trì max_concurrent running, block cho đến khi
        tất cả hoàn thành. Kết quả được persist vào repository nếu provided.

        Tương đương brain_client.executor.pool_batch_alphas nhưng:
          - Dùng ProgressUrlBrainAdapter thay vì BrainClient trực tiếp
          - Thêm on_result callback để persist vào SQLite
          - Queue luôn giữ đầy max_concurrent — 1 done → 1 filled ngay

        Returns:
            dict[expression → _ClientResult] cho mọi alpha.
            Mọi result có status trong ("COMPLETE", "ERROR") — không bao giờ PENDING.
        """
        if not alphas:
            return {}

        batch_id = batch_id or f"pool-{uuid.uuid4().hex[:8]}"
        run_id = run_id or "pool-run"
        max_concurrent = max_concurrent or self._exec_config.max_concurrent_simulations

        results: dict[str, _ClientResult] = {}
        pending: set[str] = set(alphas)
        queue = SimulationQueue(max_size=max_concurrent)

        def _make_sim_config(expr: str, index: int) -> dict:
            return {
                "region": region,
                "universe": universe,
                "neutralization": neutralization,
                "delay": delay,
                "decay": decay,
            }

        def _persist(expr: str, result: _ClientResult) -> None:
            if repository is None:
                return
            if result.status == "COMPLETE" and result.alpha_id:
                _save_result_to_repository(
                    repository,
                    job_id=f"{batch_id}-{expr[:16]}",
                    expression=expr,
                    result=result,
                    run_id=run_id,
                    round_index=round_index,
                    region=region,
                    universe=universe,
                    neutralization=neutralization,
                    delay=delay,
                    decay=decay,
                    batch_id=batch_id,
                )
            if on_result:
                try:
                    on_result(expr, result)
                except Exception as exc:
                    logger.warning("[pool] on_result callback error: %s", exc)
            results[expr] = result

        # Submit initial batch — điền queue lên max (submit thật để lấy progress_url)
        initial_count = min(len(alphas), max_concurrent)
        for expr in list(alphas)[:initial_count]:
            pending.discard(expr)
            try:
                job = self._progress_adapter.submit_simulation(expr, _make_sim_config(expr, 0))
                progress_url = job.get("progress_url", f"pool://{batch_id}/{expr[:16]}")
                queue.add(expr, progress_url)
                logger.debug("[pool] Initial fill: %s (queue=%d/%d)", expr[:30], len(queue), max_concurrent)
            except Exception as exc:
                logger.warning("[pool] Initial submit error %s: %s", expr[:40], exc)
                _persist(expr, _ClientResult(alpha_id=None, status="ERROR", error_message=str(exc)))

        # Poll loop — mỗi vòng poll ĐÚNG 1 simulation cũ nhất (FIFO).
        # Khi done → submit 1 alpha mới ngay → giữ queue luôn đầy = max.
        poll_count = 0
        while pending or not queue.is_full():
            poll_count += 1

            # Queue chưa đầy → submit ngay thêm (không poll)
            while pending and not queue.is_full():
                expr = list(pending)[0]
                pending.discard(expr)
                try:
                    job = self._progress_adapter.submit_simulation(expr, _make_sim_config(expr, 0))
                    progress_url = job.get("progress_url", f"pool://{batch_id}/{expr[:16]}")
                    queue.add(expr, progress_url)
                    logger.debug("[pool] Slot filled: %s (queue=%d/%d)", expr[:30], len(queue), max_concurrent)
                except Exception as exc:
                    logger.warning("[pool] Submit error %s: %s", expr[:40], exc)
                    _persist(expr, _ClientResult(alpha_id=None, status="ERROR", error_message=str(exc)))

            # Nếu queue đã trống → đã xong hết
            if queue.is_full() and not pending:
                break

            # Chờ đủ min_wait, poll simulation cũ nhất trong queue
            ready = queue.get_pending(min_wait_time=min_wait_time)
            if not ready:
                time.sleep(self._exec_config.polling_interval)
                continue

            sim = ready[0]  # Chỉ poll cái cũ nhất (FIFO)
            try:
                status_resp = self._progress_adapter.get_simulation_status(sim.simulation_id)
                if status_resp.get("status") == "completed":
                    full = self._progress_adapter.get_simulation_result(sim.simulation_id)
                    client_res = _ClientResult(
                        alpha_id=status_resp.get("alpha_id"),
                        status="COMPLETE",
                        metrics=full.get("metrics", {}),
                    )
                    queue.remove(sim.simulation_id)
                    _persist(sim.expression, client_res)
                    logger.debug("[pool] Done: %s -> immediately filling slot (queue=%d/%d)",
                                 sim.expression[:30], len(queue), max_concurrent)
                else:
                    queue.update_status(sim.simulation_id, "SUBMITTED")
            except Exception as exc:
                logger.warning("[pool] Poll error %s: %s", sim.expression[:40], exc)
                queue.remove(sim.simulation_id)
                _persist(sim.expression, _ClientResult(alpha_id=None, status="ERROR", error_message=str(exc)))

            if progress_callback:
                progress_callback(len(results), len(alphas))

        logger.info("[pool] All %d alphas resolved (poll_count=%d).", len(alphas), poll_count)
        return results


    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _submit_single(
        self,
        candidate,
        batch_id: str,
        run_id: str,
        round_index: int,
        sim_config: dict,
        timestamp: str,
    ) -> tuple[_JobHandle, SimulationJob]:
        """Submit 1 alpha, thêm vào queue, lưu submission record."""
        payload = self._progress_adapter.submit_simulation(
            candidate.expression, sim_config
        )
        job_id = payload["job_id"]
        progress_url = payload.get("progress_url", f"/simulations/{job_id}")

        handle = _JobHandle(
            job_id=job_id,
            progress_url=progress_url,
            expression=candidate.expression,
            candidate_id=candidate.alpha_id,
            sim_config=sim_config,
            submitted_at=timestamp,
        )

        # Thêm vào queue (chờ nếu full)
        while self._queue.is_full():
            self._poll_oldest_in_queue()
            time.sleep(self._exec_config.polling_interval)

        self._queue.add(candidate.expression, progress_url)

        with self._handles_lock:
            self._handles[job_id] = handle

        # Lưu submission record
        self.repository.submissions.upsert_submissions(
            [
                SubmissionRecord(
                    job_id=job_id,
                    batch_id=batch_id,
                    run_id=run_id,
                    round_index=round_index,
                    candidate_id=candidate.alpha_id,
                    expression=candidate.expression,
                    backend=self.brain_config.backend,
                    status="submitted",
                    sim_config_snapshot="{}",
                    submitted_at=timestamp,
                    updated_at=timestamp,
                    completed_at=None,
                    raw_submission_json="{}",
                    error_message=None,
                    timeout_deadline_at=None,
                )
            ]
        )

        job = SimulationJob(
            job_id=job_id,
            candidate_id=candidate.alpha_id,
            expression=candidate.expression,
            backend=self.brain_config.backend,
            status="submitted",
            submitted_at=timestamp,
            sim_config_snapshot=sim_config,
            run_id=run_id,
            batch_id=batch_id,
            round_index=round_index,
        )
        return handle, job

    def _poll_batch_once(
        self,
        batch_id: str,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
    ) -> BrainSimulationBatch:
        """Poll tất cả jobs trong batch, persist kết quả ngay khi hoàn thành."""
        batch = self._load_batch(batch_id)
        if batch is None:
            raise ValueError(f"Unknown submission batch: {batch_id}")

        results: list[SimulationResult] = []
        now = datetime.now(UTC).isoformat()

        for job in batch.jobs:
            if job.status not in PENDING_STATUSES:
                continue

            submission = self.repository.submissions.get_submission(job.job_id)
            if submission is None:
                continue
            if submission.next_poll_after and submission.next_poll_after > now:
                continue

            try:
                status_payload = self._progress_adapter.get_simulation_status(job.job_id)
                status = self._normalize_status(status_payload.get("status"))
                retry_after = _optional_float(status_payload.get("retry_after"))

                if status in TERMINAL_STATUSES:
                    result = self._finalize_terminal_job(
                        job,
                        status=status,
                        status_payload=status_payload,
                        updated_at=now,
                    )
                    results.append(result)
                    # Remove khỏi queue nếu có
                    self._remove_from_queue(job.job_id)
                    continue

                # Đang chạy — cập nhật next_poll_after
                next_poll_after = _shift_iso(
                    now,
                    retry_after if retry_after is not None
                    else self._exec_config.polling_interval,
                )
                self.repository.submissions.update_submission_runtime(
                    job.job_id,
                    updated_at=now,
                    status=status,
                    retry_count=submission.retry_count,
                    last_polled_at=now,
                    next_poll_after=next_poll_after,
                    timeout_deadline_at=None,
                )

            except Exception as exc:
                logger.warning("[BrainExecutor] Poll error for %s: %s", job.job_id, exc)
                retry_count = (submission.retry_count or 0) + 1
                if retry_count > (self.brain_config.max_retries or 10):
                    result = self._failed_job(job, updated_at=now, error_message=str(exc))
                    results.append(result)
                    self._remove_from_queue(job.job_id)
                else:
                    self.repository.submissions.update_submission_runtime(
                        job.job_id,
                        updated_at=now,
                        retry_count=retry_count,
                        last_polled_at=now,
                        error_message=str(exc),
                    )

        refreshed = self._refresh_batch_status(batch_id, updated_at=now, fallback=batch)
        return BrainSimulationBatch(
            batch_id=refreshed.batch_id,
            backend=refreshed.backend,
            status=refreshed.status,
            jobs=refreshed.jobs,
            results=tuple(results),
            export_path=refreshed.export_path,
        )

    def _poll_oldest_in_queue(self) -> None:
        """Poll simulation cũ nhất trong queue để giải phóng slot."""
        oldest = self._queue.get_oldest()
        if not oldest:
            return
        logger.debug(
            "[BrainExecutor] Force-poll oldest sim %s (%s)",
            oldest.expression[:40],
            oldest.simulation_id,
        )
        try:
            status_resp = self._progress_adapter.get_simulation_status(oldest.simulation_id)
            if status_resp.get("status") == "completed":
                self._queue.remove(oldest.simulation_id)
            else:
                self._queue.update_status(oldest.simulation_id, "SUBMITTED")
        except Exception as exc:
            logger.warning("[BrainExecutor] Force-poll error: %s", exc)

    def _poll_single_from_queue(self, sim_id: str) -> Optional[SimulationResult]:
        """Poll 1 simulation từ queue, persist ngay khi xong."""
        sim = self._queue.get(sim_id)
        if not sim:
            return None

        try:
            status_resp = self._progress_adapter.get_simulation_status(sim_id)
            status = self._normalize_status(status_resp.get("status"))

            if status in TERMINAL_STATUSES:
                handle = None
                with self._handles_lock:
                    handle = self._handles.get(sim_id)

                if handle:
                    result = self._finalize_terminal_job_from_handle(
                        handle=handle,
                        status=status,
                        status_payload=status_resp,
                        updated_at=datetime.now(UTC).isoformat(),
                    )
                else:
                    result = self._build_error_result(
                        job_id=sim_id,
                        expression=sim.expression,
                        error_message="Handle not found",
                        updated_at=datetime.now(UTC).isoformat(),
                    )
                self._queue.remove(sim_id)
                return result
        except Exception as exc:
            logger.warning("[BrainExecutor] _poll_single_from_queue error: %s", exc)
            self._queue.remove(sim_id)
            return self._build_error_result(
                job_id=sim_id,
                expression=sim.expression,
                error_message=str(exc),
                updated_at=datetime.now(UTC).isoformat(),
            )
        return None

    def _remove_from_queue(self, job_id: str) -> None:
        """Xóa job khỏi queue và handles."""
        self._queue.remove(job_id)
        with self._handles_lock:
            self._handles.pop(job_id, None)

    def _finalize_terminal_job(
        self,
        job: SimulationJob,
        *,
        status: str,
        status_payload: dict,
        updated_at: str,
    ) -> SimulationResult:
        """Finalize job đã hoàn thành, persist vào DB."""
        if status == "completed":
            try:
                result_payload = self._progress_adapter.get_simulation_result(job.job_id)
            except Exception as exc:
                result_payload = {
                    "job_id": job.job_id,
                    "status": status,
                    "error_message": str(exc),
                    "raw_result": status_payload,
                }
        else:
            result_payload = {
                "job_id": job.job_id,
                "status": status,
                "error_message": status_payload.get("error_message"),
                "raw_result": status_payload,
            }

        result = self._normalize_result(job=job, payload=result_payload)

        self.repository.submissions.update_submission_runtime(
            job.job_id,
            status=result.status,
            updated_at=updated_at,
            completed_at=result.simulated_at,
            error_message=result.rejection_reason,
            last_polled_at=updated_at,
            next_poll_after=None,
            timeout_deadline_at=None,
        )
        self.repository.brain_results.save_results(
            [self._to_result_record(result=result, job=job, created_at=updated_at)]
        )
        return result

    def _finalize_terminal_job_from_handle(
        self,
        handle: _JobHandle,
        *,
        status: str,
        status_payload: dict,
        updated_at: str,
    ) -> SimulationResult:
        """Finalize job từ _JobHandle (dùng trong queue-based polling)."""
        # Tạo synthetic job
        job = SimulationJob(
            job_id=handle.job_id,
            candidate_id=handle.candidate_id,
            expression=handle.expression,
            backend=self.brain_config.backend,
            status=status,
            submitted_at=handle.submitted_at,
            sim_config_snapshot=handle.sim_config,
            run_id="",
            batch_id="",
            round_index=0,
        )

        if status == "completed":
            try:
                result_payload = self._progress_adapter.get_simulation_result(handle.job_id)
            except Exception as exc:
                result_payload = {
                    "job_id": handle.job_id,
                    "status": status,
                    "error_message": str(exc),
                    "raw_result": status_payload,
                }
        else:
            result_payload = {
                "job_id": handle.job_id,
                "status": status,
                "error_message": status_payload.get("error_message"),
                "raw_result": status_payload,
            }

        result = self._normalize_result(job=job, payload=result_payload)

        self.repository.brain_results.save_results(
            [self._to_result_record(result=result, job=job, created_at=updated_at)]
        )
        return result

    def _failed_job(
        self,
        job: SimulationJob,
        *,
        updated_at: str,
        error_message: str,
    ) -> SimulationResult:
        result = self._build_error_result(
            job_id=job.job_id,
            expression=job.expression,
            error_message=error_message,
            updated_at=updated_at,
            job=job,
        )
        self.repository.submissions.update_submission_runtime(
            job.job_id,
            status="failed",
            updated_at=updated_at,
            completed_at=updated_at,
            error_message=error_message,
            last_polled_at=updated_at,
            next_poll_after=None,
            timeout_deadline_at=None,
        )
        self.repository.brain_results.save_results(
            [self._to_result_record(result=result, job=job, created_at=updated_at)]
        )
        return result

    def _timeout_pending_jobs(
        self,
        batch_id: str,
        reason: str,
    ) -> list[SimulationResult]:
        """Đánh dấu các job còn pending trong batch là timeout."""
        batch = self._load_batch(batch_id)
        if not batch:
            return []
        results: list[SimulationResult] = []
        updated_at = datetime.now(UTC).isoformat()
        for job in batch.jobs:
            if job.status in PENDING_STATUSES:
                result = SimulationResult(
                    expression=job.expression,
                    job_id=job.job_id,
                    status="timeout",
                    region=str(job.sim_config_snapshot.get("region", "")),
                    universe=str(job.sim_config_snapshot.get("universe", "")),
                    delay=int(job.sim_config_snapshot.get("delay", 1)),
                    neutralization=str(job.sim_config_snapshot.get("neutralization", "")),
                    decay=int(job.sim_config_snapshot.get("decay", 0)),
                    metrics={name: None for name in
                             ("sharpe", "fitness", "turnover", "drawdown", "returns", "margin")},
                    submission_eligible=None,
                    rejection_reason=reason,
                    raw_result={},
                    simulated_at=updated_at,
                    candidate_id=job.candidate_id,
                    batch_id=job.batch_id,
                    run_id=job.run_id,
                    round_index=job.round_index,
                    backend=self.brain_config.backend,
                    metric_source="external_brain",
                )
                self.repository.submissions.update_submission_runtime(
                    job.job_id,
                    status="timeout",
                    updated_at=updated_at,
                    completed_at=updated_at,
                    error_message=reason,
                    last_polled_at=updated_at,
                    next_poll_after=None,
                    timeout_deadline_at=None,
                )
                self.repository.brain_results.save_results(
                    [self._to_result_record(result=result, job=job, created_at=updated_at)]
                )
                results.append(result)
                self._remove_from_queue(job.job_id)
        return results

    def _build_error_result(
        self,
        job_id: str,
        expression: str,
        error_message: str,
        updated_at: str,
        job: SimulationJob | None = None,
    ) -> SimulationResult:
        if job:
            return SimulationResult(
                expression=expression,
                job_id=job_id,
                status="failed",
                region=str(job.sim_config_snapshot.get("region", "")),
                universe=str(job.sim_config_snapshot.get("universe", "")),
                delay=int(job.sim_config_snapshot.get("delay", 1)),
                neutralization=str(job.sim_config_snapshot.get("neutralization", "")),
                decay=int(job.sim_config_snapshot.get("decay", 0)),
                metrics={name: None for name in
                         ("sharpe", "fitness", "turnover", "drawdown", "returns", "margin")},
                submission_eligible=None,
                rejection_reason=error_message,
                raw_result={},
                simulated_at=updated_at,
                candidate_id=job.candidate_id if job else None,
                batch_id=job.batch_id if job else "",
                run_id=job.run_id if job else "",
                round_index=job.round_index if job else 0,
                backend=self.brain_config.backend,
                metric_source="external_brain",
            )
        return SimulationResult(
            expression=expression,
            job_id=job_id,
            status="failed",
            region="",
            universe="",
            delay=1,
            neutralization="",
            decay=0,
            metrics={name: None for name in
                     ("sharpe", "fitness", "turnover", "drawdown", "returns", "margin")},
            submission_eligible=None,
            rejection_reason=error_message,
            raw_result={},
            simulated_at=updated_at,
            batch_id="",
            backend=self.brain_config.backend,
            metric_source="external_brain",
        )

    def _normalize_result(
        self,
        *,
        job: SimulationJob,
        payload: dict,
    ) -> SimulationResult:
        """Convert payload dict → SimulationResult (giống BrainService)."""
        raw_result = payload.get("raw_result") if isinstance(payload.get("raw_result"), dict) else payload
        metrics_source = raw_result.get("metrics") if isinstance(raw_result.get("metrics"), dict) else {}
        metrics = {
            "sharpe": _optional_float(metrics_source.get("sharpe")),
            "fitness": _optional_float(metrics_source.get("fitness")),
            "turnover": _optional_float(metrics_source.get("turnover")),
            "drawdown": _optional_float(
                metrics_source.get("drawdown")
                or metrics_source.get("max_drawdown")
            ),
            "returns": _optional_float(
                metrics_source.get("returns")
                or metrics_source.get("return")
            ),
            "margin": _optional_float(metrics_source.get("margin")),
        }
        return SimulationResult(
            expression=str(payload.get("expression") or job.expression),
            job_id=job.job_id,
            status=self._normalize_status(payload.get("status")),
            region=str(raw_result.get("region") or job.sim_config_snapshot.get("region", "")),
            universe=str(raw_result.get("universe") or job.sim_config_snapshot.get("universe", "")),
            delay=int(raw_result.get("delay") or job.sim_config_snapshot.get("delay", 1)),
            neutralization=str(
                raw_result.get("neutralization") or job.sim_config_snapshot.get("neutralization", "")
            ),
            decay=int(raw_result.get("decay") or job.sim_config_snapshot.get("decay", 0)),
            metrics=metrics,
            submission_eligible=payload.get("submission_eligible"),
            rejection_reason=str(payload.get("rejection_reason") or payload.get("error_message") or ""),
            raw_result=dict(raw_result),
            simulated_at=str(
                payload.get("simulated_at")
                or raw_result.get("simulated_at")
                or datetime.now(UTC).isoformat()
            ),
            candidate_id=job.candidate_id,
            batch_id=job.batch_id,
            run_id=job.run_id,
            round_index=job.round_index,
            backend=job.backend,
            metric_source="external_brain",
        )

    def _to_result_record(
        self,
        *,
        result: SimulationResult,
        job: SimulationJob,
        created_at: str,
    ) -> BrainResultRecord:
        return BrainResultRecord(
            job_id=result.job_id,
            run_id=job.run_id,
            round_index=job.round_index,
            batch_id=job.batch_id,
            candidate_id=job.candidate_id,
            expression=result.expression,
            status=result.status,
            region=result.region,
            universe=result.universe,
            delay=result.delay,
            neutralization=result.neutralization,
            decay=result.decay,
            sharpe=result.metrics.get("sharpe"),
            fitness=result.metrics.get("fitness"),
            turnover=result.metrics.get("turnover"),
            drawdown=result.metrics.get("drawdown"),
            returns=result.metrics.get("returns"),
            margin=result.metrics.get("margin"),
            submission_eligible=result.submission_eligible,
            rejection_reason=result.rejection_reason,
            raw_result_json="{}",
            metric_source=result.metric_source,
            simulated_at=result.simulated_at,
            created_at=created_at,
        )

    @staticmethod
    def _normalize_status(value: object) -> str:
        normalized = str(value or "submitted").strip().lower()
        mapping = {
            "submitted": "submitted",
            "queued": "submitted",
            "pending": "submitted",
            "running": "running",
            "in_progress": "running",
            "done": "completed",
            "complete": "completed",
            "completed": "completed",
            "complete": "completed",
            "failed": "failed",
            "error": "failed",
            "rejected": "rejected",
            "reject": "rejected",
            "timeout": "timeout",
            "manual_pending": "manual_pending",
            "manual": "manual_pending",
            "warning": "completed",
        }
        return mapping.get(normalized, normalized.lower() if normalized else "submitted")

    def _load_batch(self, batch_id: str) -> BrainSimulationBatch | None:
        batch_record = self.repository.submissions.get_batch(batch_id)
        if batch_record is None:
            return None
        submission_records = self.repository.submissions.list_submissions(
            run_id=batch_record.run_id,
            batch_id=batch_id,
        )
        jobs = tuple(
            SimulationJob(
                job_id=record.job_id,
                candidate_id=record.candidate_id,
                expression=record.expression,
                backend=record.backend,
                status=record.status,
                submitted_at=record.submitted_at,
                sim_config_snapshot={},
                run_id=record.run_id,
                batch_id=record.batch_id,
                round_index=record.round_index,
            )
            for record in submission_records
        )
        return BrainSimulationBatch(
            batch_id=batch_record.batch_id,
            backend=batch_record.backend,
            status=batch_record.status,
            jobs=jobs,
            results=(),
            export_path=batch_record.export_path,
        )

    def _refresh_batch_status(
        self,
        batch_id: str,
        *,
        updated_at: str,
        fallback: BrainSimulationBatch | None = None,
    ) -> BrainSimulationBatch:
        refreshed = self._load_batch(batch_id) or fallback
        if refreshed is None:
            raise ValueError(f"Unknown submission batch: {batch_id}")

        if refreshed.jobs and all(
            job.status in TERMINAL_STATUSES or job.status == "manual_pending"
            for job in refreshed.jobs
        ):
            batch_status = "completed"
        elif any(job.status in PENDING_STATUSES for job in refreshed.jobs):
            batch_status = "running"
        else:
            batch_status = refreshed.status

        self.repository.submissions.update_batch_status(
            batch_id,
            status=batch_status,
            updated_at=updated_at,
        )
        return BrainSimulationBatch(
            batch_id=batch_id,
            backend=refreshed.backend,
            status=batch_status,
            jobs=refreshed.jobs,
            results=(),
            export_path=refreshed.export_path,
        )

    def _save_batch_record(
        self,
        batch_id: str,
        run_id: str,
        round_index: int,
        candidate_count: int,
        sim_config: dict,
        timestamp: str,
    ) -> None:
        self.repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id=batch_id,
                run_id=run_id,
                round_index=round_index,
                backend=self.brain_config.backend,
                status="submitting",
                candidate_count=candidate_count,
                sim_config_snapshot="{}",
                export_path=None,
                notes_json="{}",
                created_at=timestamp,
                updated_at=timestamp,
            )
        )

    def _update_batch_status(
        self,
        batch_id: str,
        status: str,
        updated_at: str,
    ) -> None:
        self.repository.submissions.update_batch_status(
            batch_id,
            status=status,
            updated_at=updated_at,
        )

    def _build_sim_config(
        self,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        round_index: int,
        batch_id: str,
        candidates: list,
    ) -> dict:
        """Build sim_config giống BrainService.build_simulation_config."""
        return {
            "backend": self.brain_config.backend,
            "region": getattr(self.brain_config, "region", "USA"),
            "universe": getattr(self.brain_config, "universe", "TOP3000"),
            "delay": int(getattr(self.brain_config, "delay", 1)),
            "neutralization": getattr(self.brain_config, "neutralization", "SUBINDUSTRY"),
            "decay": int(getattr(self.brain_config, "decay", 0)),
            "truncation": float(getattr(self.brain_config, "truncation", 0.01)),
            "pasteurization": getattr(self.brain_config, "pasteurization", True),
            "unit_handling": getattr(self.brain_config, "unit_handling", "VERIFY"),
            "nan_handling": getattr(self.brain_config, "nan_handling", "OFF"),
            "instrument_type": "EQUITY",
            "simulation_type": "REGULAR",
            "language": "FASTEXPR",
            "visualization": False,
            "test_period": "P1Y6M",
            "batch_id": batch_id,
            "run_id": environment.context.run_id,
            "round_index": round_index,
        }

    def _build_brain_client(self, brain_config: AppBrainConfig) -> BrainClient:
        """Khởi tạo BrainClient từ AppBrainConfig."""
        client_config = BrainConfig(
            api_base_url=brain_config.api_base_url or "https://api.worldquantbrain.com",
            username=getattr(brain_config, "username", None),
            password=getattr(brain_config, "password", None),
        )
        return BrainClient(client_config)


# ---------------------------------------------------------------------------
# Persistence helpers (dùng trong pool_batch callback)
# ---------------------------------------------------------------------------

def _save_result_to_repository(
    repository: SQLiteRepository,
    job_id: str,
    expression: str,
    result: _ClientResult,
    run_id: str,
    round_index: int,
    region: str,
    universe: str,
    neutralization: str,
    delay: int,
    decay: int,
    batch_id: str,
) -> None:
    """Lưu _ClientResult vào brain_results table."""
    metrics = result.metrics or {}
    record = BrainResultRecord(
        job_id=job_id,
        run_id=run_id,
        round_index=round_index,
        batch_id=batch_id,
        candidate_id="",
        expression=expression,
        status="completed" if result.status == "COMPLETE" else "failed",
        region=region,
        universe=universe,
        delay=delay,
        neutralization=neutralization,
        decay=decay,
        sharpe=_optional_float(metrics.get("sharpe")),
        fitness=_optional_float(metrics.get("fitness")),
        turnover=_optional_float(metrics.get("turnover")),
        drawdown=_optional_float(metrics.get("drawdown")),
        returns=_optional_float(metrics.get("returns")),
        margin=_optional_float(metrics.get("margin")),
        submission_eligible=None,
        rejection_reason=result.error_message,
        raw_result_json="{}",
        metric_source="external_brain",
        simulated_at=datetime.now(UTC).isoformat(),
        created_at=datetime.now(UTC).isoformat(),
    )
    repository.brain_results.save_results([record])


# ---------------------------------------------------------------------------
# Standalone pool_batch_alphas — dùng trực tiếp, không cần BrainService
# ---------------------------------------------------------------------------

def run_pool_batch(
    alphas: list[str],
    *,
    brain_config: AppBrainConfig | None = None,
    auth_adapter: "BrainApiAdapter | None" = None,
    region: str = "USA",
    universe: str = "TOP3000",
    neutralization: str = "SUBINDUSTRY",
    delay: int = 1,
    decay: int = 0,
    max_concurrent: int = 5,
    polling_interval: int = 30,
    min_wait_time: int = 300,
    simulation_timeout: int = 600,
    on_result: Callable[[str, _ClientResult], None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    timeout: int = 600,
    repository: SQLiteRepository | None = None,
    batch_id: str | None = None,
    run_id: str | None = None,
    round_index: int = 0,
) -> dict[str, _ClientResult]:
    """
    Entry point độc lập — tương đương brain_client.executor.pool_batch_alphas
    nhưng dùng ProgressUrlBrainAdapter (giữ auth qua auth_adapter).

    Parameters
    ----------
    alphas : list[str]
        Danh sách alpha expressions cần simulate.
    brain_config : AppBrainConfig | None
        Cấu hình từ core.config.
    auth_adapter : BrainApiAdapter | None
        Adapter chứa logic authentication (giữ nguyên, không thay đổi).
    region, universe, neutralization, delay, decay
        Simulation settings gửi lên BRAIN API.
    max_concurrent : int
        Số lượng simulation chạy đồng thời (default 5).
    on_result : callable | None
        Callback(expression, _ClientResult) được gọi khi mỗi simulation hoàn thành.
        Dùng để persist vào DB hoặc logging.
    progress_callback : callable | None
        Callback(completed, total) sau mỗi poll cycle.
    timeout : int
        Timeout tổng cho toàn batch.
    repository : SQLiteRepository | None
        Repository để persist kết quả. Nếu None, chỉ trả kết quả trong dict.
    batch_id, run_id, round_index
        Metadata cho batch record.

    Returns
    -------
    dict[str, _ClientResult]
        Mapping expression → SimulationResult. Mọi result có status
        trong ("COMPLETE", "ERROR") — không bao giờ PENDING.
    """
    # Khởi tạo BrainClient
    client_config = BrainConfig(
        api_base_url=(
            brain_config.api_base_url if brain_config
            else "https://api.worldquantbrain.com"
        ),
        username=getattr(brain_config, "username", None) if brain_config else None,
        password=getattr(brain_config, "password", None) if brain_config else None,
    )
    client = BrainClient(client_config)

    # ProgressUrl adapter (auth vẫn qua auth_adapter)
    adapter = ProgressUrlBrainAdapter(brain_client=client, auth_adapter=auth_adapter)

    batch_id = batch_id or f"pool-{uuid.uuid4().hex[:8]}"
    run_id = run_id or "pool-run"
    results: dict[str, _ClientResult] = {}
    pending: set[str] = set(alphas)
    queue = SimulationQueue(max_size=max_concurrent)

    def make_sim_config() -> dict:
        return {
            "region": region,
            "universe": universe,
            "neutralization": neutralization,
            "delay": delay,
            "decay": decay,
        }

    def persist(expr: str, res: _ClientResult) -> None:
        if repository and res.status == "COMPLETE" and res.alpha_id:
            _save_result_to_repository(
                repository,
                job_id=f"{batch_id}-{expr[:16]}",
                expression=expr,
                result=res,
                run_id=run_id,
                round_index=round_index,
                region=region,
                universe=universe,
                neutralization=neutralization,
                delay=delay,
                decay=decay,
                batch_id=batch_id,
            )
        if on_result:
            try:
                on_result(expr, res)
            except Exception as exc:
                logger.warning("[pool] on_result error: %s", exc)
        results[expr] = res

    # Queue luôn giữ đầy max_concurrent (trừ khi đã hết pending).
    # Submit initial batch lên max để khởi động.
    initial_count = min(len(alphas), max_concurrent)
    for expr in list(alphas)[:initial_count]:
        try:
            job = adapter.submit_simulation(expr, make_sim_config())
            queue.add(expr, job.get("progress_url", f"pool://{batch_id}/{expr[:16]}"))
            pending.discard(expr)
        except Exception as exc:
            logger.warning("[pool] Initial submit error %s: %s", expr[:40], exc)
            persist(
                expr,
                _ClientResult(alpha_id=None, status="ERROR", error_message=str(exc)),
            )

    # Poll loop — mỗi vòng chỉ poll ĐÚNG 1 simulation đầu tiên trong queue.
    # Khi done → submit 1 alpha mới ngay lập tức → queue giữ full = max.
    poll_count = 0
    while pending or not queue.is_full():
        poll_count += 1

        # Nếu queue chưa đầy → submit thêm ngay (không cần poll)
        while pending and not queue.is_full():
            expr = list(pending)[0]
            pending.discard(expr)
            try:
                job = adapter.submit_simulation(expr, make_sim_config())
                queue.add(expr, job.get("progress_url", f"pool://{batch_id}/{expr[:16]}"))
                logger.debug("[pool] Slot filled: %s (queue=%d/%d)",
                             expr[:30], len(queue), max_concurrent)
            except Exception as exc:
                logger.warning("[pool] Submit error %s: %s", expr[:40], exc)
                persist(
                    expr,
                    _ClientResult(alpha_id=None, status="ERROR", error_message=str(exc)),
                )

        # Nếu queue đã trống → đã xong hết
        if queue.is_full() and not pending:
            break

        # Chờ đủ min_wait rồi poll simulation cũ nhất (FIFO)
        ready = queue.get_pending(min_wait_time=min_wait_time)
        if not ready:
            time.sleep(polling_interval)
            continue

        sim = ready[0]  # Chỉ poll cái cũ nhất
        try:
            status_resp = adapter.get_simulation_status(sim.simulation_id)
            if status_resp.get("status") == "completed":
                full = adapter.get_simulation_result(sim.simulation_id)
                client_res = _ClientResult(
                    alpha_id=status_resp.get("alpha_id"),
                    status="COMPLETE",
                    metrics=full.get("metrics", {}),
                )
                queue.remove(sim.simulation_id)
                persist(sim.expression, client_res)
                logger.debug("[pool] Done: %s → immediately filling slot (queue=%d/%d)",
                             sim.expression[:30], len(queue), max_concurrent)
            else:
                queue.update_status(sim.simulation_id, "SUBMITTED")
        except Exception as exc:
            logger.warning("[pool] Poll error %s: %s", sim.expression[:40], exc)
            queue.remove(sim.simulation_id)
            persist(
                sim.expression,
                _ClientResult(alpha_id=None, status="ERROR", error_message=str(exc)),
            )

        if progress_callback:
            progress_callback(len(results), len(alphas))

    logger.info("[pool] Resolved %d alphas (poll_count=%d).", len(alphas), poll_count)
    return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _shift_iso(timestamp: str, seconds: float) -> str:
    try:
        return (datetime.fromisoformat(timestamp) + timedelta(seconds=seconds)).isoformat()
    except (ValueError, TypeError):
        return timestamp
