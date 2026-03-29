from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from uuid import uuid4

from adapters.brain_api_adapter import ApiEndpointConfig, BrainApiAdapter
from adapters.brain_manual_adapter import BrainManualAdapter
from adapters.simulation_adapter import SimulationAdapter
from core.config import AppConfig, BrainConfig
from core.logging import get_logger
from generator.engine import AlphaCandidate
from services.models import (
    BrainSimulationBatch,
    CommandEnvironment,
    SimulationJob,
    SimulationResult,
)
from storage.models import (
    BrainResultRecord,
    ManualImportRecord,
    SubmissionBatchRecord,
    SubmissionRecord,
)
from storage.repository import SQLiteRepository

TERMINAL_STATUSES = {"completed", "failed", "rejected", "timeout"}


class BrainService:
    def __init__(
        self,
        repository: SQLiteRepository,
        brain_config: BrainConfig,
        adapter: SimulationAdapter | None = None,
    ) -> None:
        self.repository = repository
        self.brain_config = brain_config
        self.adapter = adapter or self._build_adapter(brain_config)

    def simulate_candidates(
        self,
        candidates: list[AlphaCandidate],
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        round_index: int = 0,
        batch_size: int | None = None,
    ) -> BrainSimulationBatch:
        jobs_batch = self.submit_candidates(
            candidates,
            config=config,
            environment=environment,
            round_index=round_index,
            batch_size=batch_size,
        )
        return self.poll_batch(
            jobs_batch,
            config=config,
            environment=environment,
        )

    def submit_candidates(
        self,
        candidates: list[AlphaCandidate],
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        round_index: int = 0,
        batch_size: int | None = None,
    ) -> BrainSimulationBatch:
        logger = get_logger(__name__, run_id=environment.context.run_id, stage="brain-submit")
        selected = list(candidates[: (batch_size or self.brain_config.batch_size)])
        batch_id = f"brain-{environment.context.run_id[:8]}-r{round_index:02d}-{uuid4().hex[:8]}"
        sim_config = self.build_simulation_config(
            config=config,
            environment=environment,
            round_index=round_index,
            batch_id=batch_id,
            candidates=selected,
        )
        expressions = [candidate.expression for candidate in selected]
        submission_payloads = self.adapter.batch_submit(expressions, sim_config)
        timestamp = datetime.now(UTC).isoformat()
        export_path = None
        jobs: list[SimulationJob] = []
        submission_records: list[SubmissionRecord] = []
        snapshot_json = json.dumps(sim_config, sort_keys=True)

        for candidate, payload in zip(selected, submission_payloads, strict=True):
            export_path = str(payload.get("export_path") or export_path or "")
            status = self.normalize_status(payload.get("status"))
            job = SimulationJob(
                job_id=str(payload["job_id"]),
                candidate_id=candidate.alpha_id,
                expression=candidate.expression,
                backend=self.brain_config.backend,
                status=status,
                submitted_at=str(payload.get("submitted_at") or timestamp),
                sim_config_snapshot=sim_config,
                run_id=environment.context.run_id,
                batch_id=batch_id,
                round_index=round_index,
                export_path=str(payload.get("export_path")) if payload.get("export_path") else None,
                raw_submission=dict(payload.get("raw_submission") or payload),
                error_message=payload.get("error_message"),
            )
            jobs.append(job)
            submission_records.append(
                SubmissionRecord(
                    job_id=job.job_id,
                    batch_id=batch_id,
                    run_id=environment.context.run_id,
                    round_index=round_index,
                    candidate_id=job.candidate_id,
                    expression=job.expression,
                    backend=job.backend,
                    status=job.status,
                    sim_config_snapshot=snapshot_json,
                    submitted_at=job.submitted_at,
                    updated_at=timestamp,
                    completed_at=timestamp if job.status in TERMINAL_STATUSES else None,
                    export_path=job.export_path,
                    raw_submission_json=json.dumps(job.raw_submission, sort_keys=True),
                    error_message=job.error_message,
                )
            )

        batch_status = "manual_pending" if all(job.status == "manual_pending" for job in jobs) else "submitted"
        self.repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id=batch_id,
                run_id=environment.context.run_id,
                round_index=round_index,
                backend=self.brain_config.backend,
                status=batch_status,
                candidate_count=len(jobs),
                sim_config_snapshot=snapshot_json,
                export_path=export_path,
                notes_json=json.dumps({"backend": self.brain_config.backend}, sort_keys=True),
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        self.repository.submissions.upsert_submissions(submission_records)
        logger.info("Submitted %s candidates to BRAIN backend=%s batch=%s", len(jobs), self.brain_config.backend, batch_id)
        return BrainSimulationBatch(
            batch_id=batch_id,
            backend=self.brain_config.backend,
            status=batch_status,
            jobs=tuple(jobs),
            results=(),
            export_path=export_path,
        )

    def poll_batch(
        self,
        batch: BrainSimulationBatch,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
    ) -> BrainSimulationBatch:
        logger = get_logger(__name__, run_id=environment.context.run_id, stage="brain-poll")
        jobs = list(batch.jobs)
        if not jobs:
            return batch
        if all(job.status in TERMINAL_STATUSES or job.status == "manual_pending" for job in jobs):
            if batch.status == "manual_pending":
                return batch
            terminal_results: list[SimulationResult] = []
            created_at = datetime.now(UTC).isoformat()
            for job in jobs:
                if job.status == "manual_pending":
                    continue
                result_payload = self.adapter.get_simulation_result(job.job_id)
                result = self.normalize_result(
                    job=job,
                    payload=result_payload,
                    sim_config=job.sim_config_snapshot,
                )
                terminal_results.append(result)
                self.repository.submissions.update_submission_status(
                    job.job_id,
                    status=result.status,
                    updated_at=created_at,
                    completed_at=result.simulated_at,
                    error_message=result.rejection_reason,
                )
            if terminal_results:
                self.repository.brain_results.save_results(
                    [
                        self.to_result_record(result=result, job=self._job_by_id(jobs, result.job_id), created_at=created_at)
                        for result in terminal_results
                    ]
                )
            self.repository.submissions.update_batch_status(
                batch.batch_id,
                status="completed",
                updated_at=created_at,
            )
            return BrainSimulationBatch(
                batch_id=batch.batch_id,
                backend=batch.backend,
                status="completed",
                jobs=batch.jobs,
                results=tuple(terminal_results),
                export_path=batch.export_path,
            )

        active_jobs = {job.job_id: job for job in jobs if job.status not in TERMINAL_STATUSES and job.status != "manual_pending"}
        if not active_jobs:
            return batch

        deadline = time.monotonic() + float(config.loop.timeout_seconds)
        retry_counts: dict[str, int] = {job_id: 0 for job_id in active_jobs}
        results: dict[str, SimulationResult] = {}
        while active_jobs and time.monotonic() < deadline:
            for job_id, job in list(active_jobs.items()):
                try:
                    status_payload = self.adapter.get_simulation_status(job_id)
                    status = self.normalize_status(status_payload.get("status"))
                except Exception as exc:  # noqa: BLE001
                    retry_counts[job_id] += 1
                    if retry_counts[job_id] > self.brain_config.max_retries:
                        status = "failed"
                        status_payload = {"job_id": job_id, "status": "failed", "error_message": str(exc)}
                    else:
                        continue

                now = datetime.now(UTC).isoformat()
                self.repository.submissions.update_submission_status(
                    job_id,
                    status=status,
                    updated_at=now,
                    completed_at=now if status in TERMINAL_STATUSES else None,
                    error_message=status_payload.get("error_message"),
                )
                if status in TERMINAL_STATUSES:
                    result_payload = self.adapter.get_simulation_result(job_id)
                    result = self.normalize_result(
                        job=job,
                        payload=result_payload,
                        sim_config=job.sim_config_snapshot,
                    )
                    self.repository.brain_results.save_results(
                        [self.to_result_record(result=result, job=job, created_at=now)]
                    )
                    results[job_id] = result
                    del active_jobs[job_id]
            if active_jobs:
                time.sleep(float(config.loop.poll_interval_seconds))

        if active_jobs:
            timeout_at = datetime.now(UTC).isoformat()
            timeout_results: list[BrainResultRecord] = []
            for job_id, job in active_jobs.items():
                self.repository.submissions.update_submission_status(
                    job_id,
                    status="timeout",
                    updated_at=timeout_at,
                    completed_at=timeout_at,
                )
                timed_out = SimulationResult(
                    expression=job.expression,
                    job_id=job_id,
                    status="timeout",
                    region=str(job.sim_config_snapshot.get("region", "")),
                    universe=str(job.sim_config_snapshot.get("universe", "")),
                    delay=int(job.sim_config_snapshot.get("delay", 1)),
                    neutralization=str(job.sim_config_snapshot.get("neutralization", "")),
                    decay=int(job.sim_config_snapshot.get("decay", 0)),
                    metrics={name: None for name in ("sharpe", "fitness", "turnover", "drawdown", "returns", "margin")},
                    submission_eligible=None,
                    rejection_reason="poll_timeout",
                    raw_result={},
                    simulated_at=timeout_at,
                    candidate_id=job.candidate_id,
                    batch_id=job.batch_id,
                    run_id=job.run_id,
                    round_index=job.round_index,
                    backend=job.backend,
                )
                results[job_id] = timed_out
                timeout_results.append(self.to_result_record(result=timed_out, job=job, created_at=timeout_at))
            self.repository.brain_results.save_results(timeout_results)
            logger.warning("Timed out waiting for %s BRAIN jobs in batch %s", len(active_jobs), batch.batch_id)

        final_results = [results[job.job_id] for job in jobs if job.job_id in results]
        final_status = "completed" if final_results else batch.status
        self.repository.submissions.update_batch_status(
            batch.batch_id,
            status=final_status,
            updated_at=datetime.now(UTC).isoformat(),
        )
        return BrainSimulationBatch(
            batch_id=batch.batch_id,
            backend=batch.backend,
            status=final_status,
            jobs=batch.jobs,
            results=tuple(final_results),
            export_path=batch.export_path,
        )

    def import_manual_results(
        self,
        path: str,
        *,
        run_id: str,
    ) -> list[SimulationResult]:
        if not isinstance(self.adapter, BrainManualAdapter):
            raise TypeError("Manual result import is only available for the BrainManualAdapter backend.")

        rows = self.adapter.import_manual_results(path)
        timestamp = datetime.now(UTC).isoformat()
        records: list[BrainResultRecord] = []
        results: list[SimulationResult] = []
        seen_batch_ids: set[str] = set()

        for row in rows:
            submission = self._resolve_submission(run_id=run_id, row=row)
            if submission is None:
                raise ValueError(
                    f"Could not map manual result to an existing submission. "
                    f"job_id='{row.get('job_id')}' candidate_id='{row.get('candidate_id')}'."
                )
            sim_config = json.loads(submission.sim_config_snapshot)
            job = SimulationJob(
                job_id=submission.job_id,
                candidate_id=submission.candidate_id,
                expression=submission.expression,
                backend=submission.backend,
                status=submission.status,
                submitted_at=submission.submitted_at,
                sim_config_snapshot=sim_config,
                run_id=submission.run_id,
                batch_id=submission.batch_id,
                round_index=submission.round_index,
                export_path=submission.export_path,
                raw_submission=json.loads(submission.raw_submission_json or "{}"),
                error_message=submission.error_message,
            )
            result = self.normalize_result(job=job, payload=row, sim_config=sim_config)
            self.repository.submissions.update_submission_status(
                job.job_id,
                status=result.status,
                updated_at=timestamp,
                completed_at=result.simulated_at,
                error_message=result.rejection_reason,
            )
            records.append(self.to_result_record(result=result, job=job, created_at=timestamp))
            results.append(result)
            seen_batch_ids.add(job.batch_id)

        if records:
            self.repository.brain_results.save_results(records)
        for batch_id in seen_batch_ids:
            self.repository.submissions.update_batch_status(batch_id, status="completed", updated_at=timestamp)
            self.repository.submissions.save_manual_import(
                ManualImportRecord(
                    import_id=f"import-{uuid4().hex[:12]}",
                    run_id=run_id,
                    batch_id=batch_id,
                    source_path=str(path),
                    imported_count=sum(1 for item in results if item.batch_id == batch_id),
                    created_at=timestamp,
                )
            )
        return results

    def build_simulation_config(
        self,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        round_index: int,
        batch_id: str,
        candidates: list[AlphaCandidate],
    ) -> dict[str, object]:
        candidate_payloads = [
            {
                "job_id": f"{batch_id}-{index:04d}",
                "candidate_id": candidate.alpha_id,
                "expression": candidate.expression,
                "template_name": candidate.template_name,
                "fields_used": list(candidate.fields_used),
                "operators_used": list(candidate.operators_used),
                "generation_mode": candidate.generation_mode,
                "generation_metadata": candidate.generation_metadata,
                "run_id": environment.context.run_id,
                "round_index": round_index,
            }
            for index, candidate in enumerate(candidates, start=1)
        ]
        return {
            "backend": self.brain_config.backend,
            "region": self.brain_config.region,
            "universe": self.brain_config.universe,
            "delay": self.brain_config.delay,
            "neutralization": self.brain_config.neutralization,
            "decay": self.brain_config.decay,
            "truncation": self.brain_config.truncation,
            "pasteurization": self.brain_config.pasteurization,
            "unit_handling": self.brain_config.unit_handling,
            "nan_handling": self.brain_config.nan_handling,
            "instrument_type": "EQUITY",
            "simulation_type": "REGULAR",
            "language": "FASTEXPR",
            "visualization": False,
            "test_period": "P1Y6M",
            "manual_export_dir": self.brain_config.manual_export_dir,
            "batch_id": batch_id,
            "run_id": environment.context.run_id,
            "round_index": round_index,
            "candidate_payloads": candidate_payloads,
            "config_profile": config.runtime.profile_name,
        }

    def normalize_result(
        self,
        *,
        job: SimulationJob,
        payload: dict,
        sim_config: dict[str, object],
    ) -> SimulationResult:
        raw_result = payload.get("raw_result") if isinstance(payload.get("raw_result"), dict) else payload
        metrics_source = raw_result.get("metrics") if isinstance(raw_result.get("metrics"), dict) else payload.get("metrics")
        metrics_source = metrics_source if isinstance(metrics_source, dict) else raw_result
        metrics = {
            "sharpe": _optional_float(metrics_source.get("sharpe")),
            "fitness": _optional_float(metrics_source.get("fitness")),
            "turnover": _optional_float(metrics_source.get("turnover")),
            "drawdown": _optional_float(metrics_source.get("drawdown") or metrics_source.get("max_drawdown")),
            "returns": _optional_float(metrics_source.get("returns") or metrics_source.get("return")),
            "margin": _optional_float(metrics_source.get("margin")),
        }
        submission_eligible = payload.get("submission_eligible")
        if submission_eligible is None and isinstance(raw_result, dict):
            submission_eligible = raw_result.get("submission_eligible")
        return SimulationResult(
            expression=str(payload.get("expression") or job.expression),
            job_id=job.job_id,
            status=self.normalize_status(payload.get("status")),
            region=str(payload.get("region") or raw_result.get("region") or sim_config.get("region") or ""),
            universe=str(payload.get("universe") or raw_result.get("universe") or sim_config.get("universe") or ""),
            delay=int(payload.get("delay") or raw_result.get("delay") or sim_config.get("delay") or 1),
            neutralization=str(
                payload.get("neutralization")
                or raw_result.get("neutralization")
                or sim_config.get("neutralization")
                or ""
            ),
            decay=int(payload.get("decay") or raw_result.get("decay") or sim_config.get("decay") or 0),
            metrics=metrics,
            submission_eligible=_optional_bool(submission_eligible),
            rejection_reason=str(
                payload.get("rejection_reason")
                or raw_result.get("rejection_reason")
                or payload.get("error_message")
                or ""
            )
            or None,
            raw_result=dict(raw_result),
            simulated_at=str(payload.get("simulated_at") or raw_result.get("simulated_at") or datetime.now(UTC).isoformat()),
            candidate_id=job.candidate_id,
            batch_id=job.batch_id,
            run_id=job.run_id,
            round_index=job.round_index,
            backend=job.backend,
            metric_source="external_brain",
        )

    def to_result_record(
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
            raw_result_json=json.dumps(result.raw_result, sort_keys=True),
            metric_source=result.metric_source,
            simulated_at=result.simulated_at,
            created_at=created_at,
        )

    @staticmethod
    def normalize_status(value: object) -> str:
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
            "failed": "failed",
            "error": "failed",
            "rejected": "rejected",
            "reject": "rejected",
            "timeout": "timeout",
            "manual_pending": "manual_pending",
            "manual": "manual_pending",
        }
        return mapping.get(normalized, normalized)

    def _build_adapter(self, brain_config: BrainConfig) -> SimulationAdapter:
        if brain_config.backend == "manual":
            return BrainManualAdapter(export_root=brain_config.manual_export_dir)
        return BrainApiAdapter(
            base_url=brain_config.api_base_url or "https://api.worldquantbrain.com",
            auth_env=brain_config.api_auth_env,
            email_env=brain_config.email_env,
            password_env=brain_config.password_env,
            session_path=brain_config.session_path,
            auth_expiry_seconds=brain_config.auth_expiry_seconds,
            open_browser_for_persona=brain_config.open_browser_for_persona,
            endpoints=ApiEndpointConfig(),
            max_retries=brain_config.max_retries,
            rate_limit_per_minute=brain_config.rate_limit_per_minute,
        )

    def _resolve_submission(self, *, run_id: str, row: dict) -> SubmissionRecord | None:
        job_id = str(row.get("job_id") or "")
        if job_id:
            found = self.repository.submissions.get_submission(job_id)
            if found is not None:
                return found
        candidate_id = str(row.get("candidate_id") or "")
        batch_id = str(row.get("batch_id") or "")
        submissions = self.repository.submissions.list_submissions(
            run_id=run_id,
            batch_id=batch_id or None,
        )
        for submission in submissions:
            if candidate_id and submission.candidate_id == candidate_id:
                return submission
        return None

    @staticmethod
    def _job_by_id(jobs: list[SimulationJob], job_id: str) -> SimulationJob:
        for job in jobs:
            if job.job_id == job_id:
                return job
        raise KeyError(job_id)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_bool(value: object) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None
