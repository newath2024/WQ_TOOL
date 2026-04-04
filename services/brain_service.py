from __future__ import annotations

import json
import random
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from adapters.brain_api_adapter import ApiEndpointConfig, BrainApiAdapter, BiometricsThrottled, PersonaVerificationRequired
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
PENDING_STATUSES = {"submitted", "running"}


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
        sim_config_override: dict[str, object] | None = None,
        note_overrides: dict[str, object] | None = None,
    ) -> BrainSimulationBatch:
        logger = get_logger(__name__, run_id=environment.context.run_id, stage="brain-submit")
        selected = list(candidates[: (batch_size or self.brain_config.batch_size)])
        if not selected:
            return BrainSimulationBatch(
                batch_id=f"brain-{environment.context.run_id[:8]}-empty",
                backend=self.brain_config.backend,
                status="empty",
                jobs=(),
                results=(),
            )

        batch_id = f"brain-{environment.context.run_id[:8]}-r{round_index:02d}-{uuid4().hex[:8]}"
        sim_config = (
            self._restore_simulation_config(
                sim_config=sim_config_override,
                environment=environment,
                round_index=round_index,
                batch_id=batch_id,
                candidates=selected,
                config_profile=config.runtime.profile_name,
            )
            if sim_config_override is not None
            else self.build_simulation_config(
                config=config,
                environment=environment,
                round_index=round_index,
                batch_id=batch_id,
                candidates=selected,
            )
        )
        timestamp = datetime.now(UTC).isoformat()
        snapshot_json = json.dumps(sim_config, sort_keys=True)
        notes_payload = {
            "backend": self.brain_config.backend,
            "candidate_ids": [candidate.alpha_id for candidate in selected],
        }
        if note_overrides:
            notes_payload.update(note_overrides)
        self.repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id=batch_id,
                run_id=environment.context.run_id,
                round_index=round_index,
                backend=self.brain_config.backend,
                status="submitting",
                candidate_count=len(selected),
                sim_config_snapshot=snapshot_json,
                export_path=None,
                notes_json=json.dumps(notes_payload, sort_keys=True),
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        export_path: str | None = None
        jobs: list[SimulationJob] = []
        try:
            payloads_by_candidate = {
                str(item.get("candidate_id")): item for item in sim_config.get("candidate_payloads", []) if item.get("candidate_id")
            }
            for candidate in selected:
                single_config = dict(sim_config)
                candidate_payload = payloads_by_candidate.get(candidate.alpha_id)
                if candidate_payload is not None:
                    single_config["candidate_payloads"] = [candidate_payload]
                payload = self.adapter.submit_simulation(candidate.expression, single_config)
                job = self._job_from_payload(
                    candidate=candidate,
                    payload=payload,
                    sim_config=single_config,
                    run_id=environment.context.run_id,
                    batch_id=batch_id,
                    round_index=round_index,
                    fallback_timestamp=timestamp,
                )
                jobs.append(job)
                export_path = str(payload.get("export_path") or export_path or "") or export_path
                self.repository.submissions.upsert_submissions(
                    [
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
                            updated_at=job.submitted_at,
                            completed_at=job.submitted_at if job.status in TERMINAL_STATUSES else None,
                            export_path=job.export_path,
                            raw_submission_json=json.dumps(job.raw_submission, sort_keys=True),
                            error_message=job.error_message,
                        )
                    ]
                )
                self.repository.submissions.update_batch_status(
                    batch_id,
                    status="submitting",
                    updated_at=job.submitted_at,
                    export_path=export_path,
                )
        except Exception as exc:
            error_at = datetime.now(UTC).isoformat()
            failed_status = "failed" if not jobs else "submitting"
            self.repository.submissions.update_batch_status(
                batch_id,
                status=failed_status,
                updated_at=error_at,
                export_path=export_path,
                service_status_reason=f"submission_failed:{type(exc).__name__}",
            )
            logger.exception("Batch submission interrupted for batch=%s", batch_id)
            raise

        batch_status = "manual_pending" if jobs and all(job.status == "manual_pending" for job in jobs) else "submitted"
        completed_at = datetime.now(UTC).isoformat()
        self.repository.submissions.update_batch_status(
            batch_id,
            status=batch_status,
            updated_at=completed_at,
            export_path=export_path,
            service_status_reason=None,
        )
        logger.info("Submitted %s candidates to BRAIN backend=%s batch=%s", len(jobs), self.brain_config.backend, batch_id)
        loaded = self.load_batch(batch_id)
        return loaded or BrainSimulationBatch(
            batch_id=batch_id,
            backend=self.brain_config.backend,
            status=batch_status,
            jobs=tuple(jobs),
            results=(),
            export_path=export_path,
        )

    def _restore_simulation_config(
        self,
        *,
        sim_config: dict[str, object],
        environment: CommandEnvironment,
        round_index: int,
        batch_id: str,
        candidates: list[AlphaCandidate],
        config_profile: str,
    ) -> dict[str, object]:
        restored = dict(sim_config)
        existing_payloads = restored.get("candidate_payloads")
        payloads_by_candidate = {
            str(item.get("candidate_id")): dict(item)
            for item in (existing_payloads if isinstance(existing_payloads, list) else [])
            if isinstance(item, dict) and item.get("candidate_id")
        }
        restored["backend"] = self.brain_config.backend
        restored["manual_export_dir"] = self.brain_config.manual_export_dir
        restored["batch_id"] = batch_id
        restored["run_id"] = environment.context.run_id
        restored["round_index"] = round_index
        restored["config_profile"] = config_profile
        restored["candidate_payloads"] = [
            {
                **payloads_by_candidate.get(candidate.alpha_id, {}),
                "job_id": f"{batch_id}-{index:04d}",
                "candidate_id": candidate.alpha_id,
                "expression": candidate.expression,
                "template_name": payloads_by_candidate.get(candidate.alpha_id, {}).get("template_name")
                or candidate.template_name,
                "fields_used": list(payloads_by_candidate.get(candidate.alpha_id, {}).get("fields_used") or candidate.fields_used),
                "operators_used": list(
                    payloads_by_candidate.get(candidate.alpha_id, {}).get("operators_used") or candidate.operators_used
                ),
                "generation_mode": str(
                    payloads_by_candidate.get(candidate.alpha_id, {}).get("generation_mode") or candidate.generation_mode
                ),
                "generation_metadata": payloads_by_candidate.get(candidate.alpha_id, {}).get("generation_metadata")
                if isinstance(payloads_by_candidate.get(candidate.alpha_id, {}).get("generation_metadata"), dict)
                else candidate.generation_metadata,
                "run_id": environment.context.run_id,
                "round_index": round_index,
            }
            for index, candidate in enumerate(candidates, start=1)
        ]
        return restored

    def poll_batch(
        self,
        batch: BrainSimulationBatch,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
    ) -> BrainSimulationBatch:
        deadline = time.monotonic() + float(config.loop.timeout_seconds)
        current = self.load_batch(batch.batch_id) or batch
        all_results: dict[str, SimulationResult] = {}
        while time.monotonic() < deadline:
            current = self.poll_batch_once(
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

        timeout_results = self.timeout_pending_batch_jobs(
            current.batch_id,
            reason="poll_timeout",
            updated_at=datetime.now(UTC).isoformat(),
        )
        for result in timeout_results:
            all_results[result.job_id] = result
        refreshed = self.load_batch(current.batch_id) or current
        return BrainSimulationBatch(
            batch_id=refreshed.batch_id,
            backend=refreshed.backend,
            status=refreshed.status,
            jobs=refreshed.jobs,
            results=tuple(all_results.values()),
            export_path=refreshed.export_path,
        )

    def poll_batch_once(
        self,
        batch_id: str,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        stuck_job_after_seconds: int | None = None,
    ) -> BrainSimulationBatch:
        batch = self.load_batch(batch_id)
        if batch is None:
            raise ValueError(f"Unknown submission batch: {batch_id}")
        logger = get_logger(
            __name__,
            run_id=environment.context.run_id,
            stage="brain-poll",
            batch_id=batch_id,
        )
        if batch.status == "manual_pending":
            return batch

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

            elapsed_seconds = _seconds_since(submission.submitted_at, now)
            if self.brain_config.timeout_seconds > 0 and elapsed_seconds >= float(self.brain_config.timeout_seconds):
                logger.warning("Marking job %s as timeout after %.1fs", job.job_id, elapsed_seconds)
                results.append(self._timeout_job(job, updated_at=now, reason="poll_timeout"))
                continue

            try:
                status_payload = self.adapter.get_simulation_status(job.job_id)
                status = self.normalize_status(status_payload.get("status"))
                retry_after = _optional_float(status_payload.get("retry_after"))
            except Exception as exc:  # noqa: BLE001
                is_persona_wait = isinstance(exc, PersonaVerificationRequired) or (isinstance(exc, RuntimeError) and "Persona" in str(exc))
                if isinstance(exc, BiometricsThrottled) or is_persona_wait:
                    delay = exc.retry_after_seconds if isinstance(exc, BiometricsThrottled) and exc.retry_after_seconds else 60
                    next_poll_after = _shift_iso(now, delay)
                    self.repository.submissions.update_submission_runtime(
                        job.job_id,
                        updated_at=now,
                        retry_count=submission.retry_count,
                        last_polled_at=now,
                        next_poll_after=next_poll_after,
                        service_failure_reason=str(exc),
                    )
                    logger.warning("Job %s polling delayed by auth: %s", job.job_id, type(exc).__name__)
                    continue

                retry_count = submission.retry_count + 1
                if retry_count > self.brain_config.max_retries:
                    logger.warning("Job %s exceeded retry budget; marking failed", job.job_id)
                    results.append(self._failed_job(job, updated_at=now, error_message=str(exc)))
                    continue
                next_poll_after = _shift_iso(now, _backoff_seconds(retry_count))
                self.repository.submissions.update_submission_runtime(
                    job.job_id,
                    updated_at=now,
                    retry_count=retry_count,
                    last_polled_at=now,
                    next_poll_after=next_poll_after,
                    service_failure_reason=str(exc),
                    error_message=str(exc),
                )
                logger.warning("Transient poll failure for job %s; retry=%s", job.job_id, retry_count)
                continue

            stuck_since = submission.stuck_since
            if (
                stuck_job_after_seconds
                and elapsed_seconds >= float(stuck_job_after_seconds)
                and not stuck_since
            ):
                stuck_since = now
                logger.warning("Detected stuck job %s after %.1fs", job.job_id, elapsed_seconds)

            if status in TERMINAL_STATUSES:
                results.append(self._finalize_terminal_job(job, status=status, status_payload=status_payload, updated_at=now))
                continue

            next_poll_after = _shift_iso(
                now,
                retry_after if retry_after is not None else config.service.poll_interval_seconds,
            )
            self.repository.submissions.update_submission_runtime(
                job.job_id,
                updated_at=now,
                status=status,
                retry_count=submission.retry_count,
                last_polled_at=now,
                next_poll_after=next_poll_after,
                stuck_since=stuck_since,
                service_failure_reason="stuck_job_detected" if stuck_since else None,
            )

        refreshed = self.load_batch(batch_id) or batch
        if refreshed.jobs and all(job.status in TERMINAL_STATUSES or job.status == "manual_pending" for job in refreshed.jobs):
            batch_status = "manual_pending" if all(job.status == "manual_pending" for job in refreshed.jobs) else "completed"
        elif any(job.status in PENDING_STATUSES for job in refreshed.jobs):
            batch_status = "running"
        else:
            batch_status = refreshed.status

        self.repository.submissions.update_batch_status(
            batch_id,
            status=batch_status,
            updated_at=now,
            last_polled_at=now,
        )
        refreshed = self.load_batch(batch_id) or refreshed
        return BrainSimulationBatch(
            batch_id=refreshed.batch_id,
            backend=refreshed.backend,
            status=batch_status,
            jobs=refreshed.jobs,
            results=tuple(results),
            export_path=refreshed.export_path,
        )

    def timeout_pending_batch_jobs(self, batch_id: str, *, reason: str, updated_at: str) -> list[SimulationResult]:
        batch = self.load_batch(batch_id)
        if batch is None:
            return []
        results: list[SimulationResult] = []
        for job in batch.jobs:
            if job.status in PENDING_STATUSES:
                results.append(self._timeout_job(job, updated_at=updated_at, reason=reason))
        if results:
            self.repository.submissions.update_batch_status(
                batch_id,
                status="completed",
                updated_at=updated_at,
                last_polled_at=updated_at,
            )
        return results

    def load_batch(self, batch_id: str) -> BrainSimulationBatch | None:
        batch_record = self.repository.submissions.get_batch(batch_id)
        if batch_record is None:
            return None
        submission_records = self.repository.submissions.list_submissions(
            run_id=batch_record.run_id,
            batch_id=batch_id,
        )
        jobs = tuple(self._job_from_submission_record(record) for record in submission_records)
        return BrainSimulationBatch(
            batch_id=batch_record.batch_id,
            backend=batch_record.backend,
            status=batch_record.status,
            jobs=jobs,
            results=(),
            export_path=batch_record.export_path,
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
            self.repository.submissions.update_submission_runtime(
                job.job_id,
                status=result.status,
                updated_at=timestamp,
                completed_at=result.simulated_at,
                error_message=result.rejection_reason,
                last_polled_at=timestamp,
                next_poll_after=None,
                service_failure_reason=None,
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
        selected_profile = self._select_simulation_profile()
        region = selected_profile.region if selected_profile is not None else self.brain_config.region
        universe = selected_profile.universe if selected_profile is not None else self.brain_config.universe
        delay = selected_profile.delay if selected_profile is not None else self.brain_config.delay
        neutralization = (
            selected_profile.neutralization
            if selected_profile is not None
            else self.brain_config.neutralization
        )
        decay = selected_profile.decay if selected_profile is not None else self.brain_config.decay
        truncation = selected_profile.truncation if selected_profile is not None else self.brain_config.truncation
        sim_config = {
            "backend": self.brain_config.backend,
            "region": region,
            "universe": universe,
            "delay": delay,
            "neutralization": neutralization,
            "decay": decay,
            "truncation": truncation,
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
        if selected_profile is not None:
            sim_config["simulation_profile"] = selected_profile.name
        return sim_config

    def _select_simulation_profile(self):
        profiles = list(self.brain_config.simulation_profiles)
        if not profiles:
            return None
        weights = [max(float(profile.weight), 0.0) for profile in profiles]
        if sum(weights) <= 0:
            return random.choice(profiles)
        return random.choices(profiles, weights=weights, k=1)[0]

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

    def _job_from_payload(
        self,
        *,
        candidate: AlphaCandidate,
        payload: dict,
        sim_config: dict[str, object],
        run_id: str,
        batch_id: str,
        round_index: int,
        fallback_timestamp: str,
    ) -> SimulationJob:
        status = self.normalize_status(payload.get("status"))
        submitted_at = str(payload.get("submitted_at") or fallback_timestamp)
        return SimulationJob(
            job_id=str(payload["job_id"]),
            candidate_id=candidate.alpha_id,
            expression=candidate.expression,
            backend=self.brain_config.backend,
            status=status,
            submitted_at=submitted_at,
            sim_config_snapshot=sim_config,
            run_id=run_id,
            batch_id=batch_id,
            round_index=round_index,
            export_path=str(payload.get("export_path")) if payload.get("export_path") else None,
            raw_submission=dict(payload.get("raw_submission") or payload),
            error_message=payload.get("error_message"),
        )

    def _job_from_submission_record(self, record: SubmissionRecord) -> SimulationJob:
        return SimulationJob(
            job_id=record.job_id,
            candidate_id=record.candidate_id,
            expression=record.expression,
            backend=record.backend,
            status=record.status,
            submitted_at=record.submitted_at,
            sim_config_snapshot=json.loads(record.sim_config_snapshot or "{}"),
            run_id=record.run_id,
            batch_id=record.batch_id,
            round_index=record.round_index,
            export_path=record.export_path,
            raw_submission=json.loads(record.raw_submission_json or "{}"),
            error_message=record.error_message,
        )

    def _finalize_terminal_job(
        self,
        job: SimulationJob,
        *,
        status: str,
        status_payload: dict,
        updated_at: str,
    ) -> SimulationResult:
        if status in {"completed", "rejected"}:
            try:
                result_payload = self.adapter.get_simulation_result(job.job_id)
            except Exception as exc:  # noqa: BLE001
                result_payload = {
                    "job_id": job.job_id,
                    "status": status,
                    "error_message": str(exc),
                    "raw_result": status_payload.get("raw_status") or status_payload,
                }
        else:
            result_payload = {
                "job_id": job.job_id,
                "status": status,
                "error_message": status_payload.get("error_message"),
                "raw_result": status_payload.get("raw_status") or status_payload,
            }
        result = self.normalize_result(job=job, payload=result_payload, sim_config=job.sim_config_snapshot)
        self.repository.submissions.update_submission_runtime(
            job.job_id,
            status=result.status,
            updated_at=updated_at,
            completed_at=result.simulated_at,
            error_message=result.rejection_reason,
            last_polled_at=updated_at,
            next_poll_after=None,
            service_failure_reason=None,
        )
        self.repository.brain_results.save_results([self.to_result_record(result=result, job=job, created_at=updated_at)])
        return result

    def _failed_job(self, job: SimulationJob, *, updated_at: str, error_message: str) -> SimulationResult:
        result = self.normalize_result(
            job=job,
            payload={
                "job_id": job.job_id,
                "status": "failed",
                "error_message": error_message,
                "raw_result": {},
                "simulated_at": updated_at,
            },
            sim_config=job.sim_config_snapshot,
        )
        self.repository.submissions.update_submission_runtime(
            job.job_id,
            status="failed",
            updated_at=updated_at,
            completed_at=updated_at,
            error_message=error_message,
            retry_count=self.brain_config.max_retries,
            last_polled_at=updated_at,
            next_poll_after=None,
            service_failure_reason=error_message,
        )
        self.repository.brain_results.save_results([self.to_result_record(result=result, job=job, created_at=updated_at)])
        return result

    def _timeout_job(self, job: SimulationJob, *, updated_at: str, reason: str) -> SimulationResult:
        result = SimulationResult(
            expression=job.expression,
            job_id=job.job_id,
            status="timeout",
            region=str(job.sim_config_snapshot.get("region", "")),
            universe=str(job.sim_config_snapshot.get("universe", "")),
            delay=int(job.sim_config_snapshot.get("delay", 1)),
            neutralization=str(job.sim_config_snapshot.get("neutralization", "")),
            decay=int(job.sim_config_snapshot.get("decay", 0)),
            metrics={name: None for name in ("sharpe", "fitness", "turnover", "drawdown", "returns", "margin")},
            submission_eligible=None,
            rejection_reason=reason,
            raw_result={},
            simulated_at=updated_at,
            candidate_id=job.candidate_id,
            batch_id=job.batch_id,
            run_id=job.run_id,
            round_index=job.round_index,
            backend=job.backend,
        )
        self.repository.submissions.update_submission_runtime(
            job.job_id,
            status="timeout",
            updated_at=updated_at,
            completed_at=updated_at,
            error_message=reason,
            last_polled_at=updated_at,
            next_poll_after=None,
            service_failure_reason=reason,
        )
        self.repository.brain_results.save_results([self.to_result_record(result=result, job=job, created_at=updated_at)])
        return result

    def _build_adapter(self, brain_config: BrainConfig) -> SimulationAdapter:
        if brain_config.backend == "manual":
            return BrainManualAdapter(export_root=brain_config.manual_export_dir)
        return BrainApiAdapter(
            base_url=brain_config.api_base_url or "https://api.worldquantbrain.com",
            auth_env=brain_config.api_auth_env,
            email_env=brain_config.email_env,
            password_env=brain_config.password_env,
            credentials_file=brain_config.credentials_file,
            session_path=brain_config.session_path,
            auth_expiry_seconds=brain_config.auth_expiry_seconds,
            open_browser_for_persona=brain_config.open_browser_for_persona,
            persona_poll_interval_seconds=brain_config.persona_poll_interval_seconds,
            persona_timeout_seconds=brain_config.persona_timeout_seconds,
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
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _seconds_since(started_at: str, ended_at: str) -> float:
    start_value = datetime.fromisoformat(started_at)
    end_value = datetime.fromisoformat(ended_at)
    return max((end_value - start_value).total_seconds(), 0.0)


def _shift_iso(timestamp: str, seconds: float) -> str:
    return (datetime.fromisoformat(timestamp) + timedelta(seconds=float(seconds))).isoformat()


def _backoff_seconds(retry_count: int) -> float:
    return float(min(2**retry_count, 300))
