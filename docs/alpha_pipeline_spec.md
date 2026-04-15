# Alpha Pipeline Execution Spec

This document is the normative execution spec for alpha lifecycle, service orchestration, timeout handling, BRAIN interaction, recovery, and observability in `WQ Tool`.

`docs/project_context.md`, `docs/pipeline.md`, and `docs/architecture.md` remain useful onboarding and explanatory documents, but they are not normative for orchestration, timeout, or recovery behavior.

If current code differs from this document, this document defines intended behavior and the mismatch must be treated as implementation debt unless the text explicitly labels a rule as current-state only.

## 1. Alpha Lifecycle State Machine

### 1.1 Canonical alpha states

The alpha lifecycle uses exactly these conceptual states:

- `GENERATED`
- `VALIDATED`
- `SELECTED`
- `SUBMITTED`
- `RUNNING`
- `COMPLETED`
- `FAILED`
- `TIMEOUT`

These are conceptual lifecycle states for a candidate alpha. They are not identical to all persisted runtime values.

### 1.2 State definitions

| State | Description | Entry condition | Exit condition | Allowed transitions | Forbidden transitions |
| --- | --- | --- | --- | --- | --- |
| `GENERATED` | Candidate expression has been created but not yet accepted by validation. | Generator emits a candidate expression and metadata. | Candidate either passes validation or is rejected before validation completes. | `VALIDATED`, `FAILED` | `SELECTED`, `SUBMITTED`, `RUNNING`, `COMPLETED`, `TIMEOUT` |
| `VALIDATED` | Candidate passed syntax, nesting, field, and compatibility checks required for downstream selection. | Validation succeeds for the generated candidate. | Candidate is either selected for simulation or dropped by downstream policy. | `SELECTED`, `FAILED` | `GENERATED`, `SUBMITTED`, `RUNNING`, `COMPLETED`, `TIMEOUT` |
| `SELECTED` | Candidate has passed pre-simulation selection and is eligible to be submitted to BRAIN. | Candidate is included in the selected simulation set for a batch. | Candidate is either submitted or cannot be submitted due to submission-stage failure. | `SUBMITTED`, `FAILED` | `GENERATED`, `VALIDATED`, `RUNNING`, `COMPLETED`, `TIMEOUT` |
| `SUBMITTED` | A simulation job has been accepted by the backend and persisted as pending. | Backend returns a job identifier or manual backend creates a tracked pending job. | Job either starts running, completes directly, fails, or times out locally. | `RUNNING`, `COMPLETED`, `FAILED`, `TIMEOUT` | `GENERATED`, `VALIDATED`, `SELECTED` |
| `RUNNING` | The backend reports the job as active and not terminal. | Poll returns an active status for a submitted job. | Job reaches a terminal backend state or local timeout state. | `COMPLETED`, `FAILED`, `TIMEOUT` | `GENERATED`, `VALIDATED`, `SELECTED`, `SUBMITTED` |
| `COMPLETED` | Simulation finished successfully and normalized result data is persisted. | Backend returns a completed terminal state and result normalization succeeds. | Terminal state. | None | Any non-terminal state |
| `FAILED` | Candidate or job reached an unrecoverable failure state. | Validation, submission, polling, or backend processing fails and the failure is not recoverable in-line. | Terminal state. | None | Any non-terminal state |
| `TIMEOUT` | Local orchestration declares the job terminal because its timeout deadline has expired. | `timeout_deadline_at` is reached and the job is still pending. | Terminal state. | None | Any non-terminal state |

### 1.3 Lifecycle rules

- Terminal states are `COMPLETED`, `FAILED`, and `TIMEOUT`.
- A terminal state must never transition back into an active lifecycle state.
- A candidate that never reaches `SUBMITTED` may still conceptually end in `FAILED`.
- Validation-stage failures are conceptual lifecycle failures even when the repo does not persist a `submissions` row for that candidate.

### 1.4 Alpha lifecycle vs service/runtime state

The following are orchestration or runtime states, not alpha lifecycle states:

- `manual_pending`
- `paused_quarantine`
- `waiting_persona`
- `waiting_persona_confirmation`
- `auth_throttled`
- `auth_unavailable`
- `cooldown`

Rules:

- Alpha lifecycle state answers "where is this candidate/job in the simulation lifecycle?"
- Runtime state answers "what is the service process currently doing or waiting on?"
- Runtime state must not be used as a substitute for lifecycle state in learning, timeout, or result accounting.

### 1.5 Persistence mapping

| Concept | Primary persisted fields | Notes |
| --- | --- | --- |
| Alpha lifecycle state | `submissions.status`, `brain_results.status` | `submissions.status` is the job-level operational source of truth; `brain_results.status` records normalized terminal outcomes. |
| Batch orchestration state | `submission_batches.status` | Batch states include orchestration-only values such as `submitting`, `manual_pending`, and `paused_quarantine`. |
| Service/runtime state | `service_runtime.status` | Runtime state tracks service control flow, auth waits, cooldown, and paused conditions. |
| Timeout source of truth | `submissions.timeout_deadline_at` | Local timeout must be computed from this field, not from raw `submitted_at` alone. |

Current-state mappings:

- `SUBMITTED` maps to `submissions.status = submitted`
- `RUNNING` maps to `submissions.status = running`
- `COMPLETED` maps to `submissions.status = completed` and `brain_results.status = completed`
- `FAILED` maps to `submissions.status in {failed, rejected}` and matching `brain_results.status`
- `TIMEOUT` maps to `submissions.status = timeout` and `brain_results.status = timeout`

## 2. Service Orchestration Contract

### 2.1 Tick contract

The service loop is a strict polling-first orchestrator.

Hard rules:

- If pending jobs or pending batches exist at tick start, the tick is `poll-only`.
- A tick that starts with pending work must not prepare a new batch.
- A tick that starts with pending work must not submit a new batch.
- New batch submission is allowed only when the tick starts from an idle state with no pending work.
- Polling must take precedence over candidate generation, validation, selection, and batch preparation.

Pending work is defined operationally as batch or job records that are still active, including:

- `submission_batches.status in {submitting, submitted, running}`
- `submissions.status in {submitted, running}`

### 2.2 Tick order

The expected tick order is:

1. Probe or resume auth state.
2. Apply auth wait handling or cooldown handling when required.
3. Recover interrupted `submitting` batches when safe.
4. Stop on `paused_quarantine` if unresolved ambiguous submission state exists.
5. If pending work exists, poll pending batches and return.
6. If no pending work exists, reconcile completed batches and learning side effects.
7. If still idle and not blocked by stop or cooldown, prepare and submit one new batch.

### 2.3 Scheduling rules

- Polling must never be blocked by batch preparation.
- Polling must never be blocked by candidate generation.
- Polling must never be blocked by validation or pre-simulation selection.
- A poll tick may reconcile completed batches after polling if pending count becomes zero.
- A submit tick may prepare at most one new batch.
- A submit tick must not burst multiple new batches to backfill slots.

### 2.4 Concurrency and slot filling

Submission capacity is bounded by:

- `service.max_pending_jobs`
- the runtime-learned safe cap derived from backend concurrency behavior

The effective pending cap is:

- `min(service.max_pending_jobs, learned_safe_cap)` when a learned cap exists
- otherwise `service.max_pending_jobs`

Slot filling strategy:

- Slot filling is conservative.
- Slot filling is only applied when the tick starts idle.
- Batch size is capped by available slots, configured simulation batch size, and selected candidate count.
- Recent failure rate may reduce submission count further.
- Learned backend caps are runtime safety limits, not targets to exceed.

### 2.5 Forbidden orchestration behavior

The service must never:

- poll and prepare a new batch in the same tick when the tick started with pending work
- block polling on synchronous batch preparation
- treat "freed capacity after polling" as permission to submit in that same tick
- burst new work because the backend briefly reports free slots

## 3. Timeout Policy

### 3.1 Timeout source of truth

Local timeout is defined by `submissions.timeout_deadline_at`.

Hard rules:

- Timeout must never be computed from raw `submitted_at` alone after submission has been persisted.
- Every pending submission must have a timeout deadline.
- Backfilled rows must be assigned a deadline if legacy rows exist without one.

### 3.2 Initial deadline

Initial deadline calculation:

- At submission time, set `timeout_deadline_at = submitted_at + brain.timeout_seconds`.
- If `brain.timeout_seconds <= 0`, timeout is disabled and no deadline is required.

### 3.3 Deadline extension rules

Deadline extensions are required whenever the service intentionally defers useful polling.

Deadline must be extended for:

- `PersonaVerificationRequired`
- `BiometricsThrottled`
- backend `retry_after`
- transient poll retry backoff
- service states that intentionally sleep while pending jobs exist:
  - `waiting_persona_confirmation`
  - `waiting_persona`
  - `auth_throttled`
  - `auth_unavailable`

Extension rules:

- Poll-level deferrals extend the targeted job deadline by the same defer amount.
- Service-level auth waits extend all pending deadlines for the run by `next_sleep_seconds`.
- A deadline extension must be persisted before the tick returns control to the scheduler.

### 3.4 Timeout enforcement

Timeout enforcement occurs only when:

- the job is still pending
- `timeout_deadline_at` exists
- current time is at or after the deadline

When enforced:

- the job transitions to `TIMEOUT`
- rejection reason must be recorded as `poll_timeout`
- the timeout is a local orchestration terminal state, not a backend-declared terminal result

### 3.5 Persona verification and service pause behavior

- Persona verification waits must not cause false local timeouts.
- Auth throttling waits must not cause false local timeouts.
- Service-managed sleeps with explicit defer duration must extend deadlines.
- Submission cooldown after `ConcurrentSimulationLimitExceeded` is not a polling pause; existing pending work must continue to be polled.
- `paused_quarantine` is not a timeout extension mechanism by itself; it is an orchestration block on ambiguous submission state.
- Operator-stopped or crashed processes are outside automatic extension scope; stale pending jobs after such events should be handled by normal polling if still within deadline, or by explicit recovery if deadlines are no longer trustworthy.

### 3.6 Required protections

The timeout model must explicitly prevent:

- poll starvation caused by internal preparation work
- false timeout caused by service auth waits
- false timeout caused by local retry or defer logic

## 4. BRAIN Interaction Model

### 4.1 Submission model

The submission model is batch-oriented and job-persistent.

Rules:

- One service submit tick creates at most one batch.
- A batch is persisted before external submission begins.
- Each submitted job is persisted individually with its own job identifier and timeout deadline.
- Manual backend batches may remain in `manual_pending`; API backend batches transition into active polling states.

### 4.2 Partial submissions

If submission is interrupted after some jobs were accepted:

- the batch record must reflect actual submitted count, not only planned count
- `submission_batches.candidate_count` must equal actual submitted jobs
- `notes_json` must include:
  - `planned_candidate_count`
  - `submitted_candidate_count`
  - `submission_interrupted = true`
- batch snapshot payload must be trimmed to the candidates actually submitted
- candidates that were not submitted are not members of that persisted batch and may be reconsidered later

### 4.3 Polling cadence

Polling rules:

- Jobs with `next_poll_after > now` are skipped until eligible.
- Otherwise, pending jobs are polled directly against the backend.
- Normal running-job cadence uses configured poll interval unless backend supplies `retry_after`.
- Polling cadence must remain independent from batch preparation time.

### 4.4 Retry strategy

Transient backend or transport errors:

- increment `retry_count`
- set `next_poll_after` using bounded backoff
- extend `timeout_deadline_at` by the same defer amount
- mark job `FAILED` when retry budget is exhausted

Auth-related deferrals:

- do not consume the job into `FAILED`
- do not convert directly to `TIMEOUT`
- must update poll metadata and deadline extension fields

### 4.5 Concurrent simulation limit handling

On `ConcurrentSimulationLimitExceeded`:

- stop the current submit attempt
- persist any jobs already accepted
- place the service into `cooldown`
- derive and persist a runtime `learned_safe_cap`
- do not immediately retry back to configured hard cap in the same service run

The learned safe cap is a conservative runtime limit, not a permanent config change.

## 5. Failure Taxonomy

Failures are grouped into validation failures, submission/backend failures, and poll/runtime failures.

| Failure type | Typical cause | Detection stage | Handling strategy |
| --- | --- | --- | --- |
| `parse_failed` | Expression cannot be parsed into the supported language/AST. | Generation or validation | Mark conceptual lifecycle as `FAILED`; reject before selection or submission. |
| `invalid_nesting` | Illegal operator nesting, invalid time-series vs cross-sectional composition, or structural rule break. | Validation | Reject before selection; do not submit; record validation reason where available. |
| `invalid_field` | Field is not allowed, not runtime-available, not supported for the active profile, or disallowed by validator rules. | Validation | Reject before selection; do not submit; track as validation-stage failure. |
| `unit_mismatch` | Expression combines units or wrappers in a way forbidden by validation rules. | Validation | Reject before selection; do not submit; record as validation failure. |
| `poll_timeout` | Local timeout deadline expires while the job is still pending. | Poll/runtime | Mark job `TIMEOUT`; persist local terminal result; treat as orchestration-side terminal state. |
| `auth_blocked` | Persona verification, biometrics throttling, or auth unavailability prevents useful backend progress. | Session management or polling | Defer polling, extend deadlines, surface runtime wait state; do not convert directly to `FAILED` or `TIMEOUT` while within managed defer rules. |
| `backend_error` | Submit, status, or result call fails due to backend, transport, or incompatible backend response. | Submission or polling | Retry when transient; mark `FAILED` when unrecoverable or retry budget is exhausted. |

Rules:

- Validation failures are pre-submission failures.
- Submission/backend failures are backend interaction failures.
- Poll/runtime failures occur after a job exists and has entered the tracked pending set.
- `poll_timeout` must never be conflated with backend-declared failure or rejection.

## 6. Recovery Mechanism

### 6.1 Supported recovery modes

Recovery supports:

- recover by `batch_id`
- recover by one or more `job_id` values

Recovery is for stale pending jobs that need a direct backend status check without applying normal local timeout gating first.

### 6.2 Recovery rules

Hard rules:

- Recovery may bypass local timeout checks.
- Recovery must not bypass auth readiness checks.
- Recovery must operate only on pending jobs in `submitted` or `running`.
- Recovery must not mutate unrelated jobs.
- Recovery must reuse the normal finalize and persistence path so that `submissions`, `submission_batches`, and `brain_results` remain internally consistent.

### 6.3 Recovery workflow

Expected recovery flow:

1. Resolve target jobs by `batch_id` or `job_id`.
2. Fail fast if target jobs do not exist or are not pending.
3. Probe auth state without requesting a new interactive login.
4. If auth is not ready, exit without mutating target job state.
5. Poll the backend directly while ignoring local timeout deadlines for those target jobs.
6. If remote status is terminal, finalize using the same normalization and persistence path as normal polling.
7. If remote status is still pending, refresh runtime poll metadata only.
8. Recompute batch status from actual job states after recovery completes.

### 6.4 When recovery should be used

Use recovery when:

- a process stopped or stalled and pending jobs may still be alive remotely
- persona or auth interruption made local timeout state untrustworthy for a specific stale batch
- an operator needs a one-off truth check on pending backend jobs before allowing normal service flow to continue

Do not use recovery when:

- the job is already terminal locally
- the batch is an ambiguous submission replay problem rather than a stale pending job problem
- auth is not ready
- normal service polling can resume safely without bypassing timeout logic

## 7. Learning Loop

### 7.1 Source of truth

External BRAIN outcomes are the source of truth for closed-loop learning and parent selection.

Local evaluation may assist screening, but it must not replace external outcome learning for closed-loop behavior.

### 7.2 Data collected for learning

The learning loop should use:

- terminal result status
- normalized metrics such as `fitness`, `sharpe`, `turnover`, `drawdown`, `returns`, and `margin`
- `submission_eligible`
- rejection reason or timeout reason
- expression structure, fields used, operators used, motif, mutation mode, and generation metadata
- region, regime, and batch context
- lineage and parent relationships

### 7.3 Feedback targets

Learning affects:

- operator preference
- field preference
- expression pattern preference
- mutation and exploration strategy
- parent selection for later rounds

### 7.4 Feedback injection points

Feedback is injected after normalized external outcomes are persisted, including:

- normal API polling completion
- manual result import completion
- completed batch reconciliation in the service loop

Rules:

- Learning must consume normalized terminal outcomes, not raw transport payloads alone.
- Completed and failed terminal outcomes are both informative.
- Timeouts and rejections must remain available as negative learning signals.

## 8. Anti-Patterns

The system must never:

- poll and submit in the same tick when the tick started with pending work
- block polling with batch preparation, candidate generation, validation, or selection
- compute timeout from `submitted_at` alone after a persisted deadline exists
- allow auth waits to create false local timeouts
- treat runtime wait states as lifecycle result states
- persist a partial submission as though all planned jobs were accepted
- recover non-pending jobs through the stale-job recovery path
- bypass auth readiness in recovery
- collapse batch concurrency behavior into uncontrolled burst submission after a temporary free slot
- lose the distinction between backend-declared failure and local `poll_timeout`
- allow invalid cross-sectional/time-series composition that violates validator nesting rules

## 9. Observability and Metrics

### 9.1 Required metrics

The system must expose or derive at least these metrics:

- `batch_duration`
- `poll_latency`
- `timeout_rate`
- `submission_rate`
- `alpha/hour`

Supporting metrics should include:

- `poll_pending_ms`
- `prepare_batch_ms`
- `submit_batch_ms`
- pending job count
- completed count
- failed count
- cooldown occurrences
- auth wait occurrences

### 9.2 Metric intent

| Metric | Definition | Bottleneck signal |
| --- | --- | --- |
| `batch_duration` | Time from batch creation to batch terminal completion. | High values indicate backend latency, poll delay, or stalled jobs. |
| `poll_latency` | Time spent polling pending work or delay between poll opportunities and actual poll execution. | Rising values indicate poll starvation or auth-induced deferral. |
| `timeout_rate` | `timeout terminal jobs / total terminal jobs` over a window. | High values indicate broken polling cadence, broken timeout model, or backend degradation. |
| `submission_rate` | Submitted jobs per hour. | Low values with low pending count indicate underfilled idle capacity or selection starvation. |
| `alpha/hour` | Completed external results per hour. | Low values despite high submission attempts indicate backend latency, timeout inflation, or stalled orchestration. |

### 9.3 Metric interpretation rules

- High `prepare_batch_ms` with stable `poll_pending_ms` is acceptable only when the service is idle before submission.
- High `prepare_batch_ms` combined with rising `timeout_rate` indicates forbidden coupling between prep and polling.
- High `submission_rate` with low `alpha_per_hour` indicates backend latency, timeout problems, or backend acceptance without useful completion.
- High auth wait counts with low timeout rate indicates deadline extension behavior is working as intended.
- High auth wait counts with high timeout rate indicates broken defer handling or incorrect deadline management.
