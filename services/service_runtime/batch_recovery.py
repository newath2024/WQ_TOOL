from __future__ import annotations

import json
from datetime import UTC, datetime

from services.evaluation_service import alpha_candidate_from_record
from domain.brain import BrainResultRecord
from domain.candidate import AlphaCandidate
from storage.models import (
    SubmissionBatchRecord,
    SubmissionRecord,
)

class BatchRecovery:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def _recover_submitting_batches(
        self,
        *,
        service_name: str,
        run_id: str,
        now: str,
        allow_resubmit: bool = True,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        recovered: list[str] = []
        failed: list[str] = []
        resubmitted: list[str] = []
        quarantined: list[str] = []
        policy = self.config.service.ambiguous_submission_policy
        statuses: tuple[str, ...] = ("submitting", "paused_quarantine") if policy in {"fail", "resubmit"} else ("submitting",)
        batches = self.repository.submissions.list_batches_by_status(run_id=run_id, statuses=statuses)
        submissions_by_batch = {
            batch.batch_id: self.repository.submissions.list_submissions(run_id=run_id, batch_id=batch.batch_id)
            for batch in batches
        }
        superseded_batch_ids = self._resubmitted_source_batch_ids(run_id=run_id)
        latest_resubmit_batch_id = None
        if policy == "resubmit" and allow_resubmit:
            latest_resubmit_batch_id = self._latest_resubmittable_ambiguous_batch_id(
                batches=batches,
                submissions_by_batch=submissions_by_batch,
                superseded_batch_ids=superseded_batch_ids,
            )
        for batch in batches:
            if batch.status == "paused_quarantine" and not self._is_ambiguous_batch(batch):
                quarantined.append(batch.batch_id)
                continue
            submissions = submissions_by_batch[batch.batch_id]
            should_recover = bool(
                submissions
                and (
                    policy == "resubmit"
                    or (batch.candidate_count > 0 and len(submissions) >= batch.candidate_count)
                )
            )
            if should_recover:
                recovered_status = "manual_pending" if all(
                    submission.status == "manual_pending" for submission in submissions
                ) else "submitted"
                self.repository.submissions.update_batch_status(
                    batch.batch_id,
                    status=recovered_status,
                    updated_at=now,
                    service_status_reason=None,
                )
                recovered.append(batch.batch_id)
                continue
            if policy == "resubmit" and batch.batch_id in superseded_batch_ids:
                self._mark_ambiguous_batch_failed(
                    batch=batch,
                    submissions=submissions,
                    updated_at=now,
                    reason="ambiguous_submission_superseded",
                )
                failed.append(batch.batch_id)
                continue
            if policy == "resubmit" and not allow_resubmit:
                continue
            if (
                policy == "resubmit"
                and latest_resubmit_batch_id is not None
                and batch.batch_id != latest_resubmit_batch_id
            ):
                self._mark_ambiguous_batch_failed(
                    batch=batch,
                    submissions=submissions,
                    updated_at=now,
                    reason="ambiguous_submission_stale",
                )
                failed.append(batch.batch_id)
                continue
            if policy == "resubmit":
                resubmitted_batch_id = self._resubmit_ambiguous_batch(
                    run_id=run_id,
                    batch=batch,
                    submissions=submissions,
                )
                if resubmitted_batch_id:
                    resubmitted.append(resubmitted_batch_id)
                else:
                    failed.append(batch.batch_id)
                continue
            if policy == "quarantine":
                self.repository.submissions.update_batch_status(
                    batch.batch_id,
                    status="paused_quarantine",
                    updated_at=now,
                    service_status_reason="ambiguous_submission",
                    quarantined_at=now,
                )
                self.repository.service_runtime.update_state(
                    service_name,
                    status="paused_quarantine",
                    active_batch_id=batch.batch_id,
                    updated_at=now,
                )
                quarantined.append(batch.batch_id)
                continue
            self._mark_ambiguous_batch_failed(
                batch=batch,
                submissions=submissions,
                updated_at=now,
                reason="ambiguous_submission_assumed_failed",
            )
            failed.append(batch.batch_id)
        return recovered, failed, resubmitted, quarantined

    @staticmethod
    def _is_ambiguous_batch(batch: SubmissionBatchRecord) -> bool:
        reason = str(batch.service_status_reason or "")
        return batch.status == "submitting" or reason.startswith("ambiguous_submission")

    def _resubmit_ambiguous_batch(
        self,
        *,
        run_id: str,
        batch: SubmissionBatchRecord,
        submissions: list[SubmissionRecord],
    ) -> str | None:
        candidate_ids = self._candidate_ids_for_batch(batch=batch, submissions=submissions)
        loaded_candidates, skipped_candidates = self._load_candidates_for_batch(
            run_id=run_id,
            batch=batch,
            submissions=submissions,
        )
        active_submissions = self._active_submissions_by_candidate(run_id=run_id, excluding_batch_id=batch.batch_id)
        terminal_results = self._terminal_results_by_candidate(run_id=run_id)
        candidate_failure_reasons = {
            str(item["candidate_id"]): str(item["reason"])
            for item in skipped_candidates
            if item.get("candidate_id") and item.get("reason")
        }
        resubmittable_candidates: list[AlphaCandidate] = []
        for candidate_id in candidate_ids:
            if candidate_id in candidate_failure_reasons:
                continue
            active_submission = active_submissions.get(candidate_id)
            if active_submission is not None:
                skipped_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "ambiguous_replay_guard_active_submission",
                        "blocking_batch_id": active_submission.batch_id,
                        "blocking_job_id": active_submission.job_id,
                        "blocking_status": active_submission.status,
                    }
                )
                candidate_failure_reasons[candidate_id] = "ambiguous_replay_guard_active_submission"
                continue
            terminal_result = terminal_results.get(candidate_id)
            if terminal_result is not None:
                skipped_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "ambiguous_replay_guard_terminal_result",
                        "blocking_batch_id": terminal_result.batch_id,
                        "blocking_job_id": terminal_result.job_id,
                        "blocking_status": terminal_result.status,
                    }
                )
                candidate_failure_reasons[candidate_id] = "ambiguous_replay_guard_terminal_result"
                continue
            candidate = loaded_candidates.get(candidate_id)
            if candidate is None:
                skipped_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "ambiguous_submission_missing_candidate",
                    }
                )
                candidate_failure_reasons[candidate_id] = "ambiguous_submission_missing_candidate"
                continue
            resubmittable_candidates.append(candidate)
        recovery_notes = self._ambiguous_recovery_notes(
            source_candidate_ids=candidate_ids,
            skipped_candidates=skipped_candidates,
        )
        updated_at = datetime.now(UTC).isoformat()
        if not resubmittable_candidates:
            self._mark_ambiguous_batch_failed(
                batch=batch,
                submissions=submissions,
                updated_at=updated_at,
                reason="ambiguous_submission_no_resubmittable_candidates",
                note_overrides=recovery_notes,
                candidate_failure_reasons=candidate_failure_reasons,
            )
            return None
        sim_config_override = self._decode_json_object(batch.sim_config_snapshot)
        resubmitted_batch = self.brain_service.submit_candidates(
            resubmittable_candidates,
            config=self.config,
            environment=self.environment,
            round_index=batch.round_index,
            batch_size=len(resubmittable_candidates),
            sim_config_override=sim_config_override,
            note_overrides={
                "resubmitted_from_batch_id": batch.batch_id,
                **recovery_notes,
            },
        )
        self._mark_ambiguous_batch_failed(
            batch=batch,
            submissions=submissions,
            updated_at=updated_at,
            reason=f"ambiguous_submission_resubmitted:{resubmitted_batch.batch_id}",
            note_overrides={
                **recovery_notes,
                "resubmitted_batch_id": resubmitted_batch.batch_id,
            },
            candidate_failure_reasons=candidate_failure_reasons,
        )
        return resubmitted_batch.batch_id

    def _mark_ambiguous_batch_failed(
        self,
        *,
        batch: SubmissionBatchRecord,
        submissions: list[SubmissionRecord],
        updated_at: str,
        reason: str,
        note_overrides: dict[str, object] | None = None,
        candidate_failure_reasons: dict[str, str] | None = None,
    ) -> None:
        for submission in submissions:
            if submission.status not in {"submitted", "running", "manual_pending"}:
                continue
            submission_reason = (
                candidate_failure_reasons.get(submission.candidate_id, reason)
                if candidate_failure_reasons is not None
                else reason
            )
            self.repository.submissions.update_submission_runtime(
                submission.job_id,
                status="failed",
                updated_at=updated_at,
                completed_at=updated_at,
                error_message=submission_reason,
                last_polled_at=updated_at,
                next_poll_after=None,
                stuck_since=None,
                service_failure_reason=submission_reason,
            )
        notes_json = self._merged_batch_notes(batch=batch, note_overrides=note_overrides)
        update_kwargs: dict[str, object] = {
            "status": "failed",
            "updated_at": updated_at,
            "service_status_reason": reason,
            "quarantined_at": None,
        }
        if notes_json is not None:
            update_kwargs["notes_json"] = notes_json
        self.repository.submissions.update_batch_status(
            batch.batch_id,
            **update_kwargs,
        )

    def _load_candidates_for_batch(
        self,
        *,
        run_id: str,
        batch: SubmissionBatchRecord,
        submissions: list[SubmissionRecord],
    ) -> tuple[dict[str, AlphaCandidate], list[dict[str, str]]]:
        candidate_ids = self._candidate_ids_for_batch(batch=batch, submissions=submissions)
        if not candidate_ids:
            return {}, []
        payloads_by_candidate = self._batch_payloads_by_candidate(batch)
        submissions_by_candidate = {submission.candidate_id: submission for submission in submissions if submission.candidate_id}
        parent_refs_map = self.repository.get_parent_refs(run_id)
        records_by_id = {
            record.alpha_id: alpha_candidate_from_record(record, parent_refs=parent_refs_map.get(record.alpha_id))
            for record in self.repository.list_alpha_records(run_id)
            if record.alpha_id in set(candidate_ids)
        }
        candidates: dict[str, AlphaCandidate] = {}
        skipped: list[dict[str, str]] = []
        for candidate_id in candidate_ids:
            candidate = records_by_id.get(candidate_id)
            if candidate is None:
                payload = payloads_by_candidate.get(candidate_id, {})
                submission = submissions_by_candidate.get(candidate_id)
                expression = str(payload.get("expression") or (submission.expression if submission else "")).strip()
                if not expression:
                    skipped.append(
                        {
                            "candidate_id": candidate_id,
                            "reason": "ambiguous_submission_missing_candidate",
                        }
                    )
                    continue
                generation_metadata = payload.get("generation_metadata")
                candidate = AlphaCandidate(
                    alpha_id=candidate_id,
                    expression=expression,
                    normalized_expression=expression,
                    generation_mode=str(payload.get("generation_mode") or "recovered"),
                    parent_ids=(),
                    complexity=int(payload.get("complexity") or 0),
                    created_at=batch.created_at,
                    template_name=str(payload.get("template_name") or ""),
                    fields_used=tuple(str(item) for item in (payload.get("fields_used") or ()) if str(item)),
                    operators_used=tuple(str(item) for item in (payload.get("operators_used") or ()) if str(item)),
                    depth=int(payload.get("depth") or 0),
                    generation_metadata=generation_metadata if isinstance(generation_metadata, dict) else {},
                )
            candidates[candidate_id] = candidate
        return candidates, skipped

    def _candidate_ids_for_batch(
        self,
        *,
        batch: SubmissionBatchRecord,
        submissions: list[SubmissionRecord],
    ) -> list[str]:
        candidate_ids: list[str] = []
        for payload in self._decode_json_object(batch.sim_config_snapshot).get("candidate_payloads", []):
            if isinstance(payload, dict) and payload.get("candidate_id"):
                candidate_ids.append(str(payload["candidate_id"]))
        if not candidate_ids:
            for candidate_id in self._decode_json_object(batch.notes_json).get("candidate_ids", []):
                if candidate_id:
                    candidate_ids.append(str(candidate_id))
        if not candidate_ids:
            candidate_ids.extend(submission.candidate_id for submission in submissions if submission.candidate_id)
        return list(dict.fromkeys(candidate_ids))

    def _batch_payloads_by_candidate(self, batch: SubmissionBatchRecord) -> dict[str, dict[str, object]]:
        payloads = self._decode_json_object(batch.sim_config_snapshot).get("candidate_payloads", [])
        if not isinstance(payloads, list):
            return {}
        return {
            str(item.get("candidate_id")): dict(item)
            for item in payloads
            if isinstance(item, dict) and item.get("candidate_id")
        }

    def _active_submissions_by_candidate(
        self,
        *,
        run_id: str,
        excluding_batch_id: str,
    ) -> dict[str, SubmissionRecord]:
        active: dict[str, SubmissionRecord] = {}
        for submission in self.repository.submissions.list_pending_submissions(run_id):
            if not submission.candidate_id or submission.batch_id == excluding_batch_id:
                continue
            active.setdefault(submission.candidate_id, submission)
        return active

    def _terminal_results_by_candidate(self, *, run_id: str) -> dict[str, BrainResultRecord]:
        terminal_statuses = {"completed", "failed", "rejected", "timeout"}
        return {
            result.candidate_id: result
            for result in self.repository.brain_results.list_latest_results_by_candidate(run_id)
            if result.candidate_id and result.status in terminal_statuses
        }

    @staticmethod
    def _ambiguous_recovery_notes(
        *,
        source_candidate_ids: list[str],
        skipped_candidates: list[dict[str, str]],
    ) -> dict[str, object]:
        notes: dict[str, object] = {
            "recovery_source_candidate_ids": list(source_candidate_ids),
        }
        if skipped_candidates:
            notes["recovery_skipped_candidates"] = [dict(item) for item in skipped_candidates]
        return notes

    def _merged_batch_notes(
        self,
        *,
        batch: SubmissionBatchRecord,
        note_overrides: dict[str, object] | None,
    ) -> str | None:
        if note_overrides is None:
            return None
        notes = self._decode_json_object(batch.notes_json)
        notes.update(note_overrides)
        return json.dumps(notes, sort_keys=True)

    def _resubmitted_source_batch_ids(self, *, run_id: str) -> set[str]:
        source_batch_ids: set[str] = set()
        for batch in self.repository.submissions.list_batches(run_id):
            resubmitted_from = str(
                self._decode_json_object(batch.notes_json).get("resubmitted_from_batch_id") or ""
            ).strip()
            if resubmitted_from:
                source_batch_ids.add(resubmitted_from)
        return source_batch_ids

    def _latest_resubmittable_ambiguous_batch_id(
        self,
        *,
        batches: list[SubmissionBatchRecord],
        submissions_by_batch: dict[str, list[SubmissionRecord]],
        superseded_batch_ids: set[str],
    ) -> str | None:
        candidates: list[SubmissionBatchRecord] = []
        for batch in batches:
            if not self._is_ambiguous_batch(batch):
                continue
            if batch.batch_id in superseded_batch_ids:
                continue
            submissions = submissions_by_batch.get(batch.batch_id, [])
            if batch.candidate_count > 0 and len(submissions) >= batch.candidate_count:
                continue
            candidates.append(batch)
        if not candidates:
            return None
        latest = max(candidates, key=self._batch_recency_key)
        return latest.batch_id

    @staticmethod
    def _batch_recency_key(batch: SubmissionBatchRecord) -> tuple[datetime, datetime, int, str]:
        baseline = datetime.min.replace(tzinfo=UTC)

        def _parse(timestamp: str | None) -> datetime:
            if not timestamp:
                return baseline
            try:
                return datetime.fromisoformat(timestamp)
            except ValueError:
                return baseline

        return (_parse(batch.created_at), _parse(batch.updated_at), int(batch.round_index), batch.batch_id)


