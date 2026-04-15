from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from adapters.brain_api_adapter import ConcurrentSimulationLimitExceeded
from adapters.brain_manual_adapter import BrainManualAdapter
from adapters.simulation_adapter import SimulationAdapter
from core.config import SimulationProfile, load_config
from core.run_context import RunContext
from data.field_registry import FieldRegistry, FieldSpec
from evaluation.critic import AlphaDiagnosis
from generator.engine import AlphaCandidate
from memory.case_memory import CaseMemoryService
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot
import services.brain_service as brain_service_module
from services.brain_service import BrainService
from services.candidate_selection_service import CandidateSelectionService
from services.closed_loop_service import ClosedLoopService
from services.models import CommandEnvironment, SimulationJob, SimulationResult
from services.runtime_service import init_run
from storage.models import BrainResultRecord, SubmissionBatchRecord, SubmissionRecord
from storage.repository import SQLiteRepository
from tests.conftest import write_sample_csv, write_sample_metadata_csv


class FakeCompletedAdapter(SimulationAdapter):
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.status_calls: Counter[str] = Counter()

    def submit_simulation(self, expression: str, sim_config: dict) -> dict:
        return self.batch_submit([expression], sim_config)[0]

    def batch_submit(self, expressions: list[str], sim_config: dict) -> list[dict]:
        submissions = []
        payloads = list(sim_config["candidate_payloads"])
        for expression, payload in zip(expressions, payloads, strict=True):
            job_id = payload["job_id"]
            self.jobs[job_id] = {"expression": expression}
            submissions.append(
                {
                    "job_id": job_id,
                    "expression": expression,
                    "status": "submitted",
                    "raw_submission": {"queued": True},
                }
            )
        return submissions

    def get_simulation_status(self, job_id: str) -> dict:
        self.status_calls[job_id] += 1
        return {"job_id": job_id, "status": "completed"}

    def get_simulation_result(self, job_id: str) -> dict:
        return {
            "job_id": job_id,
            "status": "completed",
            "raw_result": {
                "metrics": {
                    "sharpe": 1.35,
                    "fitness": 1.10,
                    "turnover": 0.45,
                    "drawdown": 0.12,
                    "returns": 0.08,
                    "margin": 0.05,
                },
                "submission_eligible": True,
            },
        }


class FlakyAdapter(FakeCompletedAdapter):
    def __init__(self, fail_times: int) -> None:
        super().__init__()
        self.fail_times = fail_times

    def get_simulation_status(self, job_id: str) -> dict:
        self.status_calls[job_id] += 1
        if self.status_calls[job_id] <= self.fail_times:
            raise RuntimeError("temporary transport error")
        return {"job_id": job_id, "status": "completed"}


class NeverCompletesAdapter(FakeCompletedAdapter):
    def get_simulation_status(self, job_id: str) -> dict:
        self.status_calls[job_id] += 1
        return {"job_id": job_id, "status": "running"}


class SubmitFailsAdapter(FakeCompletedAdapter):
    def submit_simulation(self, expression: str, sim_config: dict) -> dict:
        raise RuntimeError('BRAIN API request failed with status 400: {"settings":{"nanHandling":["\\"FALSE\\" is not a valid choice."]}}')


class PartialSubmitLimitAdapter(FakeCompletedAdapter):
    def __init__(self, *, fail_on_call: int) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call
        self.submit_attempts = 0

    def submit_simulation(self, expression: str, sim_config: dict) -> dict:
        self.submit_attempts += 1
        if self.submit_attempts >= self.fail_on_call:
            raise ConcurrentSimulationLimitExceeded("CONCURRENT_SIMULATION_LIMIT_EXCEEDED")
        return super().submit_simulation(expression, sim_config)


def test_manual_adapter_export_and_import_round_trip(tmp_path: Path) -> None:
    adapter = BrainManualAdapter(export_root=tmp_path)
    sim_config = {
        "batch_id": "manual-batch-1",
        "manual_export_dir": str(tmp_path),
        "candidate_payloads": [
            {
                "job_id": "job-1",
                "candidate_id": "alpha-1",
                "expression": "rank(close)",
                "template_name": "momentum",
                "fields_used": ["close"],
                "operators_used": ["rank"],
                "generation_mode": "template",
                "generation_metadata": {"template_name": "momentum"},
                "run_id": "run-1",
                "round_index": 1,
            }
        ],
    }

    submissions = adapter.batch_submit(["rank(close)"], sim_config)

    assert submissions[0]["status"] == "manual_pending"
    export_path = Path(submissions[0]["export_path"])
    assert export_path.exists()
    exported_rows = list(csv.DictReader(export_path.open("r", encoding="utf-8")))
    assert exported_rows[0]["candidate_id"] == "alpha-1"
    assert exported_rows[0]["template_name"] == "momentum"

    result_path = tmp_path / "manual_results.csv"
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "job_id",
                "candidate_id",
                "batch_id",
                "expression",
                "status",
                "sharpe",
                "fitness",
                "turnover",
                "drawdown",
                "returns",
                "margin",
                "submission_eligible",
                "rejection_reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "job_id": "job-1",
                "candidate_id": "alpha-1",
                "batch_id": "manual-batch-1",
                "expression": "rank(close)",
                "status": "completed",
                "sharpe": "1.1",
                "fitness": "0.9",
                "turnover": "0.4",
                "drawdown": "0.2",
                "returns": "0.07",
                "margin": "0.04",
                "submission_eligible": "true",
                "rejection_reason": "",
            }
        )

    imported = adapter.import_manual_results(result_path)

    assert imported[0]["job_id"] == "job-1"
    assert imported[0]["metrics"]["fitness"] == 0.9
    assert imported[0]["submission_eligible"] is True


def test_brain_service_normalizes_missing_metrics_safely(tmp_path: Path) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        service = BrainService(repository, load_config("config/dev.yaml").brain, adapter=FakeCompletedAdapter())
        job = SimulationJob(
            job_id="job-1",
            candidate_id="alpha-1",
            expression="rank(close)",
            backend="api",
            status="submitted",
            submitted_at="2026-01-01T00:00:00+00:00",
            sim_config_snapshot={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "sector", "decay": 0},
            run_id="run-1",
            batch_id="batch-1",
        )
        result = service.normalize_result(
            job=job,
            payload={"job_id": "job-1", "status": "completed", "raw_result": {"submission_eligible": False}},
            sim_config=job.sim_config_snapshot,
        )
    finally:
        repository.close()

    assert result.status == "completed"
    assert result.metrics["sharpe"] is None
    assert result.submission_eligible is False
    assert result.region == "USA"


def test_brain_service_build_simulation_config_selects_weighted_profile(monkeypatch) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        config.brain.simulation_profiles = [
            SimulationProfile(
                name="stable",
                region="USA",
                universe="TOP1000",
                delay=1,
                neutralization="SUBINDUSTRY",
                decay=3,
                truncation=0.01,
                weight=0.6,
            ),
            SimulationProfile(
                name="aggressive_short",
                region="USA",
                universe="TOP500",
                delay=1,
                neutralization="SUBINDUSTRY",
                decay=1,
                truncation=0.02,
                weight=0.4,
            ),
        ]
        environment = _init_environment(repository, config, "run-brain-sim")
        service = BrainService(repository, config.brain, adapter=FakeCompletedAdapter())
        captured: dict[str, list[float]] = {}

        def fake_choices(population, *, weights, k):
            assert k == 1
            captured["weights"] = list(weights)
            return [population[1]]

        monkeypatch.setattr(brain_service_module.random, "choices", fake_choices)

        sim_config = service.build_simulation_config(
            config=config,
            environment=environment,
            round_index=1,
            batch_id="batch-1",
            candidates=[_candidate("alpha-1", "rank(close)")],
        )
    finally:
        repository.close()

    assert captured["weights"] == [0.6, 0.4]
    assert sim_config["simulation_profile"] == "aggressive_short"
    assert sim_config["region"] == "USA"
    assert sim_config["universe"] == "TOP500"
    assert sim_config["delay"] == 1
    assert sim_config["neutralization"] == "SUBINDUSTRY"
    assert sim_config["decay"] == 1
    assert sim_config["truncation"] == 0.02


def test_brain_service_build_simulation_config_falls_back_when_profiles_empty() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        config.storage.path = ":memory:"
        config.brain.simulation_profiles = []
        environment = _init_environment(repository, config, "run-brain-sim")
        service = BrainService(repository, config.brain, adapter=FakeCompletedAdapter())

        sim_config = service.build_simulation_config(
            config=config,
            environment=environment,
            round_index=1,
            batch_id="batch-1",
            candidates=[_candidate("alpha-1", "rank(close)")],
        )
    finally:
        repository.close()

    assert "simulation_profile" not in sim_config
    assert sim_config["region"] == config.brain.region
    assert sim_config["universe"] == config.brain.universe
    assert sim_config["delay"] == config.brain.delay
    assert sim_config["neutralization"] == config.brain.neutralization
    assert sim_config["decay"] == config.brain.decay
    assert sim_config["truncation"] == config.brain.truncation


def test_submission_and_result_stores_persist_traceability(tmp_path: Path) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        repository.upsert_run(
            run_id="run-1",
            seed=7,
            config_path="config/dev.yaml",
            config_snapshot="{}",
            status="running",
            started_at="2026-01-01T00:00:00+00:00",
        )
        repository.submissions.upsert_batch(
            SubmissionBatchRecord(
                batch_id="batch-1",
                run_id="run-1",
                round_index=1,
                backend="manual",
                status="manual_pending",
                candidate_count=1,
                sim_config_snapshot="{}",
                export_path=str(tmp_path / "batch.csv"),
                notes_json="{}",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
            )
        )
        repository.submissions.upsert_submissions(
            [
                SubmissionRecord(
                    job_id="job-1",
                    batch_id="batch-1",
                    run_id="run-1",
                    round_index=1,
                    candidate_id="alpha-1",
                    expression="rank(close)",
                    backend="manual",
                    status="manual_pending",
                    sim_config_snapshot="{}",
                    submitted_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:00:00+00:00",
                    completed_at=None,
                    export_path=str(tmp_path / "batch.csv"),
                    raw_submission_json=json.dumps({"queued": True}),
                    error_message=None,
                )
            ]
        )
        repository.brain_results.save_results(
            [
                BrainResultRecord(
                    job_id="job-1",
                    run_id="run-1",
                    round_index=1,
                    batch_id="batch-1",
                    candidate_id="alpha-1",
                    expression="rank(close)",
                    status="completed",
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                    neutralization="sector",
                    decay=0,
                    sharpe=1.2,
                    fitness=0.9,
                    turnover=0.5,
                    drawdown=0.2,
                    returns=0.07,
                    margin=0.04,
                    submission_eligible=True,
                    rejection_reason=None,
                    raw_result_json="{}",
                    metric_source="external_brain",
                    simulated_at="2026-01-01T00:05:00+00:00",
                    created_at="2026-01-01T00:05:00+00:00",
                )
            ]
        )
        submission = repository.submissions.get_submission("job-1")
        result = repository.brain_results.get_result("job-1")
    finally:
        repository.close()

    assert submission is not None
    assert result is not None
    assert submission.batch_id == "batch-1"
    assert result.submission_eligible is True
    assert result.fitness == 0.9


def test_candidate_selection_enforces_template_diversity() -> None:
    selector = CandidateSelectionService()
    field_registry = FieldRegistry(
        fields={
            "close": FieldSpec(
                name="close",
                dataset="runtime",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="price",
                runtime_available=True,
                category_weight=1.0,
                field_score=1.0,
            ),
            "volume": FieldSpec(
                name="volume",
                dataset="runtime",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="volume",
                runtime_available=True,
                category_weight=0.9,
                field_score=0.9,
            ),
        }
    )
    candidates = [
        _candidate("alpha-1", "rank(ts_mean(close, 5))", template_name="momentum", fields_used=("close",)),
        _candidate("alpha-2", "zscore(close - ts_mean(close, 5))", template_name="momentum", fields_used=("close",)),
        _candidate("alpha-3", "rank(ts_std(volume, 5))", template_name="volatility", fields_used=("volume",)),
    ]

    selected, archived = selector.select_for_simulation(
        candidates,
        snapshot=PatternMemorySnapshot(regime_key="regime"),
        field_registry=field_registry,
        batch_size=2,
        min_pattern_support=1,
        rejection_filters=[],
    )

    assert len(selected) == 2
    assert {item.candidate.template_name for item in selected} == {"momentum", "volatility"}
    assert any(item.archive_reason == "template_diversity_cap" for item in archived)


def test_pre_sim_pipeline_applies_conservative_pre_screen_before_scoring() -> None:
    selector = CandidateSelectionService()
    field_registry = FieldRegistry(
        fields={
            "close": FieldSpec(
                name="close",
                dataset="runtime",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="price",
                runtime_available=True,
                category_weight=1.0,
                field_score=1.0,
            ),
            "volume": FieldSpec(
                name="volume",
                dataset="runtime",
                field_type="matrix",
                coverage=1.0,
                alpha_usage_count=0,
                category="volume",
                runtime_available=True,
                category_weight=0.9,
                field_score=0.9,
            ),
        }
    )
    eligible = _candidate(
        "alpha-soft-field",
        "rank(ts_mean(close, 5))",
        template_name="momentum",
        operators_used=("rank", "ts_mean"),
    )
    rejected = replace(
        _candidate(
            "alpha-pre-screen",
            "rank(ts_mean(volume, 5))",
            template_name="momentum",
            fields_used=("volume",),
            operators_used=("rank", "ts_mean"),
        ),
        complexity=1,
    )

    result = selector.run_pre_sim_pipeline(
        [eligible, rejected],
        snapshot=PatternMemorySnapshot(regime_key="regime"),
        field_registry=field_registry,
        batch_size=2,
        min_pattern_support=1,
        rejection_filters=[],
    )

    assert [item.candidate.alpha_id for item in result.selected] == ["alpha-soft-field"]
    assert len(result.archived) == 1
    archived = result.archived[0]
    assert archived.candidate.alpha_id == "alpha-pre-screen"
    assert archived.archive_reason == "pre_screen_low_complexity"
    assert "pre_screen_low_field_diversity" in archived.reason_codes
    assert result.stage_metrics["kept_after_pre_screen"] == 1
    assert result.stage_metrics["rejected_by_pre_screen"] == 1
    assert result.stage_metrics["warned_low_field_diversity"] == 2
    assert result.stage_metrics["rejected_pre_screen_low_complexity"] == 1
    pre_screen_decision = next(
        decision for decision in result.selection_decisions if decision.alpha_id == "alpha-pre-screen"
    )
    assert pre_screen_decision.selected is False
    assert pre_screen_decision.rank is None
    assert pre_screen_decision.reason_codes == archived.reason_codes


def test_brain_service_retries_then_completes(tmp_path: Path) -> None:
    config = load_config("config/dev.yaml")
    config.storage.path = ":memory:"
    config.brain.max_retries = 2
    config.loop.poll_interval_seconds = 1
    config.loop.timeout_seconds = 5
    repository = SQLiteRepository(":memory:")
    try:
        environment = _init_environment(repository, config, "run-brain-sim")
        service = BrainService(repository, config.brain, adapter=FlakyAdapter(fail_times=1))
        candidate = _candidate("alpha-1", "rank(close)")
        batch = service.simulate_candidates([candidate], config=config, environment=environment, round_index=1, batch_size=1)
    finally:
        repository.close()

    assert batch.completed_count == 1
    assert batch.results[0].status == "completed"


def test_brain_service_marks_timeout_when_jobs_never_finish(tmp_path: Path) -> None:
    config = load_config("config/dev.yaml")
    config.storage.path = ":memory:"
    config.loop.poll_interval_seconds = 1
    config.loop.timeout_seconds = 1
    repository = SQLiteRepository(":memory:")
    try:
        environment = _init_environment(repository, config, "run-brain-sim")
        service = BrainService(repository, config.brain, adapter=NeverCompletesAdapter())
        candidate = _candidate("alpha-1", "rank(close)")
        batch = service.simulate_candidates([candidate], config=config, environment=environment, round_index=1, batch_size=1)
    finally:
        repository.close()

    assert batch.results[0].status == "timeout"
    assert batch.results[0].rejection_reason == "poll_timeout_live"


def test_brain_service_marks_timeout_after_downtime_when_deadline_is_already_stale(tmp_path: Path) -> None:
    config = load_config("config/dev.yaml")
    config.storage.path = ":memory:"
    config.service.poll_interval_seconds = 5
    repository = SQLiteRepository(":memory:")
    try:
        environment = _init_environment(repository, config, "run-brain-timeout-downtime")
        service = BrainService(repository, config.brain, adapter=NeverCompletesAdapter())
        candidate = _candidate("alpha-1", "rank(close)")
        batch = service.submit_candidates([candidate], config=config, environment=environment, round_index=1, batch_size=1)
        submission = repository.submissions.get_submission(batch.jobs[0].job_id)
        assert submission is not None
        repository.submissions.update_submission_runtime(
            submission.job_id,
            updated_at=submission.updated_at,
            next_poll_after="2000-01-01T00:00:00+00:00",
            timeout_deadline_at="2000-01-01T00:00:30+00:00",
        )

        refreshed = service.poll_batch_once(batch.batch_id, config=config, environment=environment)
    finally:
        repository.close()

    assert len(refreshed.results) == 1
    assert refreshed.results[0].status == "timeout"
    assert refreshed.results[0].rejection_reason == "poll_timeout_after_downtime"


def test_brain_service_recover_jobs_ignores_local_timeout_deadline_and_finalizes_terminal_results(tmp_path: Path) -> None:
    config = load_config("config/dev.yaml")
    config.storage.path = ":memory:"
    repository = SQLiteRepository(":memory:")
    try:
        environment = _init_environment(repository, config, "run-brain-sim")
        service = BrainService(repository, config.brain, adapter=FakeCompletedAdapter())
        candidate = _candidate("alpha-1", "rank(close)")
        batch = service.submit_candidates([candidate], config=config, environment=environment, round_index=1, batch_size=1)
        submission = repository.submissions.get_submission(batch.jobs[0].job_id)
        assert submission is not None
        assert submission.timeout_deadline_at is not None
        repository.submissions.update_submission_runtime(
            submission.job_id,
            updated_at=submission.updated_at,
            timeout_deadline_at="2000-01-01T00:00:00+00:00",
        )

        recovered = service.recover_jobs([submission.job_id], config=config, environment=environment)
        refreshed_batch = repository.submissions.get_batch(batch.batch_id)
        results = repository.brain_results.list_results(run_id=environment.context.run_id)
    finally:
        repository.close()

    assert recovered[0].status == "completed"
    assert recovered[0].timeout_deadline_at is None
    assert refreshed_batch is not None
    assert refreshed_batch.status == "completed"
    assert len(results) == 1


def test_brain_service_recover_jobs_keeps_remote_running_jobs_pending(tmp_path: Path) -> None:
    config = load_config("config/dev.yaml")
    config.storage.path = ":memory:"
    repository = SQLiteRepository(":memory:")
    try:
        environment = _init_environment(repository, config, "run-brain-sim")
        service = BrainService(repository, config.brain, adapter=NeverCompletesAdapter())
        candidate = _candidate("alpha-1", "rank(close)")
        batch = service.submit_candidates([candidate], config=config, environment=environment, round_index=1, batch_size=1)
        submission = repository.submissions.get_submission(batch.jobs[0].job_id)
        assert submission is not None
        repository.submissions.update_submission_runtime(
            submission.job_id,
            updated_at=submission.updated_at,
            timeout_deadline_at="2000-01-01T00:00:00+00:00",
        )

        recovered = service.recover_jobs([submission.job_id], config=config, environment=environment)
        refreshed = repository.submissions.get_submission(submission.job_id)
        results = repository.brain_results.list_results(run_id=environment.context.run_id)
    finally:
        repository.close()

    assert recovered[0].status == "running"
    assert refreshed is not None
    assert refreshed.status == "running"
    assert results == []


def test_brain_service_marks_batch_failed_when_submit_fails_before_any_job(tmp_path: Path) -> None:
    config = load_config("config/dev.yaml")
    config.storage.path = ":memory:"
    repository = SQLiteRepository(":memory:")
    try:
        environment = _init_environment(repository, config, "run-brain-sim")
        service = BrainService(repository, config.brain, adapter=SubmitFailsAdapter())
        candidate = _candidate("alpha-1", "rank(close)")
        try:
            service.submit_candidates([candidate], config=config, environment=environment, round_index=1, batch_size=1)
        except RuntimeError as exc:
            assert "nanHandling" in str(exc)
        else:
            raise AssertionError("Expected submit failure")
        batch = repository.submissions.get_latest_batch(environment.context.run_id)
    finally:
        repository.close()

    assert batch is not None
    assert batch.status == "failed"
    assert batch.service_status_reason == "submission_failed:RuntimeError"


def test_brain_service_records_actual_submitted_count_when_limit_interrupts_batch(tmp_path: Path) -> None:
    config = load_config("config/dev.yaml")
    config.storage.path = ":memory:"
    repository = SQLiteRepository(":memory:")
    try:
        environment = _init_environment(repository, config, "run-brain-partial-submit")
        service = BrainService(repository, config.brain, adapter=PartialSubmitLimitAdapter(fail_on_call=3))
        candidates = [
            _candidate("alpha-1", "rank(close)"),
            _candidate("alpha-2", "rank(open)"),
            _candidate("alpha-3", "rank(volume)"),
        ]
        try:
            service.submit_candidates(candidates, config=config, environment=environment, round_index=1, batch_size=3)
        except ConcurrentSimulationLimitExceeded as exc:
            assert exc.detail == "CONCURRENT_SIMULATION_LIMIT_EXCEEDED"
        else:
            raise AssertionError("Expected concurrent simulation limit")
        batch = repository.submissions.get_latest_batch(environment.context.run_id)
        assert batch is not None
        submissions = repository.submissions.list_submissions(
            run_id=environment.context.run_id,
            batch_id=batch.batch_id,
        )
    finally:
        repository.close()

    assert batch.status == "submitting"
    assert batch.candidate_count == 2
    assert batch.service_status_reason == "submission_failed:ConcurrentSimulationLimitExceeded"
    assert len(submissions) == 2
    notes = json.loads(batch.notes_json)
    assert notes["candidate_ids"] == ["alpha-1", "alpha-2"]
    assert notes["planned_candidate_count"] == 3
    assert notes["submitted_candidate_count"] == 2
    assert notes["submission_interrupted"] is True
    snapshot = json.loads(batch.sim_config_snapshot)
    assert [payload["candidate_id"] for payload in snapshot["candidate_payloads"]] == ["alpha-1", "alpha-2"]


def test_closed_loop_service_runs_with_mocked_adapter(tmp_path: Path) -> None:
    data_path = tmp_path / "daily_ohlcv.csv"
    metadata_path = tmp_path / "symbol_metadata.csv"
    write_sample_csv(data_path)
    write_sample_metadata_csv(metadata_path)

    config = load_config("config/dev.yaml")
    config.data.path = str(data_path)
    config.aux_data.group_path = str(metadata_path)
    config.aux_data.factor_path = str(metadata_path)
    config.aux_data.mask_path = str(metadata_path)
    config.storage.path = str(tmp_path / "closed_loop.sqlite3")
    config.runtime.log_level = "WARNING"
    config.loop.rounds = 2
    config.loop.generation_batch_size = 6
    config.loop.simulation_batch_size = 2
    config.loop.mutate_top_k = 1
    config.loop.max_children_per_parent = 2
    config.loop.poll_interval_seconds = 1
    config.loop.timeout_seconds = 5
    config.generation.template_count = 6
    config.generation.grammar_count = 0
    config.generation.mutation_count = 2

    repository = SQLiteRepository(config.storage.path)
    try:
        environment = _init_environment(repository, config, "run-closed-loop")
        service = ClosedLoopService(
            repository,
            brain_service=BrainService(repository, config.brain, adapter=FakeCompletedAdapter()),
        )
        summary = service.run(config=config, environment=environment)
        results = repository.brain_results.list_results(run_id=environment.context.run_id)
        history = repository.connection.execute(
            "SELECT COUNT(*) AS total FROM alpha_history WHERE metric_source = 'external_brain'"
        ).fetchone()["total"]
    finally:
        repository.close()

    assert summary.status == "completed"
    assert len(summary.rounds) >= 1
    assert len(results) >= 1
    assert history >= 1


def test_region_specific_memory_isolated_and_cold_region_can_fall_back_to_global_priors(tmp_path: Path) -> None:
    repository = SQLiteRepository(":memory:")
    memory_service = PatternMemoryService()
    case_memory_service = CaseMemoryService()
    try:
        usa_config = load_config("config/dev.yaml")
        usa_config.storage.path = ":memory:"
        usa_config.brain.region = "USA"
        eur_config = load_config("config/dev.yaml")
        eur_config.storage.path = ":memory:"
        eur_config.brain.region = "EUR"
        asi_config = load_config("config/dev.yaml")
        asi_config.storage.path = ":memory:"
        asi_config.brain.region = "ASI"

        usa_context = memory_service.build_learning_context("shared-dataset", usa_config)
        eur_context = memory_service.build_learning_context("shared-dataset", eur_config)
        asi_context = memory_service.build_learning_context("shared-dataset", asi_config)

        _persist_brain_learning_record(
            repository,
            config=usa_config,
            learning_context=usa_context,
            candidate=_candidate("alpha-usa", "rank(ts_mean(close, 5))", template_name="momentum"),
            outcome_region="USA",
        )
        _persist_brain_learning_record(
            repository,
            config=eur_config,
            learning_context=eur_context,
            candidate=_candidate("alpha-eur", "zscore(ts_delta(close, 2))", template_name="mean_reversion"),
            outcome_region="EUR",
        )

        usa_snapshot = repository.alpha_history.load_snapshot(
            regime_key=usa_context.regime_key,
            region=usa_context.region,
            global_regime_key=usa_context.global_regime_key,
            parent_pool_size=10,
            region_learning_config=usa_config.adaptive_generation.region_learning,
            pattern_decay=usa_config.adaptive_generation.pattern_decay,
            prior_weight=usa_config.adaptive_generation.critic_thresholds.score_prior_weight,
        )
        cold_snapshot = repository.alpha_history.load_snapshot(
            regime_key=asi_context.regime_key,
            region=asi_context.region,
            global_regime_key=asi_context.global_regime_key,
            parent_pool_size=10,
            region_learning_config=asi_config.adaptive_generation.region_learning,
            pattern_decay=asi_config.adaptive_generation.pattern_decay,
            prior_weight=asi_config.adaptive_generation.critic_thresholds.score_prior_weight,
        )
        cold_case_snapshot = repository.alpha_history.load_case_snapshot(
            asi_context.regime_key,
            region=asi_context.region,
            global_regime_key=asi_context.global_regime_key,
            region_learning_config=asi_config.adaptive_generation.region_learning,
        )

        usa_parent_ids = {parent.alpha_id for parent in usa_snapshot.top_parents}
        local_family_values = {pattern.pattern_value for pattern in usa_snapshot.ordered_patterns(scope="local", kind="family")}
        global_family_values = {pattern.pattern_value for pattern in usa_snapshot.ordered_patterns(scope="global", kind="family")}
        cold_score, _, signature, _ = memory_service.score_expression(
            "rank(ts_mean(close, 5))",
            cold_snapshot,
            min_pattern_support=1,
        )
        predicted = case_memory_service.predict_objectives(
            generation_metadata={"motif": "momentum", "mutation_mode": "exploit_local"},
            signature=signature,
            snapshot=cold_case_snapshot,
            novelty_score=0.5,
            diversity_score=0.5,
        )
    finally:
        repository.close()

    assert usa_parent_ids == {"alpha-usa"}
    assert usa_snapshot.global_sample_count == 1
    assert local_family_values
    assert global_family_values
    assert local_family_values != global_family_values
    assert not cold_snapshot.top_parents
    assert cold_snapshot.sample_count == 0
    assert cold_snapshot.global_sample_count == 2
    assert cold_snapshot.blend is not None and cold_snapshot.blend.global_weight == 1.0
    assert cold_case_snapshot.sample_count == 0
    assert cold_case_snapshot.global_sample_count == 2
    assert predicted.fitness > 0.0
    assert cold_score > 0.0


def _candidate(
    alpha_id: str,
    expression: str,
    *,
    template_name: str = "template",
    fields_used: tuple[str, ...] = ("close",),
    operators_used: tuple[str, ...] = ("rank",),
) -> AlphaCandidate:
    return AlphaCandidate(
        alpha_id=alpha_id,
        expression=expression,
        normalized_expression=expression,
        generation_mode="template",
        parent_ids=(),
        complexity=3,
        created_at="2026-01-01T00:00:00+00:00",
        template_name=template_name,
        fields_used=fields_used,
        operators_used=operators_used,
        depth=2,
        generation_metadata={},
    )


def _persist_brain_learning_record(
    repository: SQLiteRepository,
    *,
    config,
    learning_context,
    candidate: AlphaCandidate,
    outcome_region: str,
) -> None:
    environment = _init_environment(repository, config, "run-brain-sim")
    repository.save_dataset_summary(
        environment.context.run_id,
        summary={},
        dataset_fingerprint="shared-dataset",
        selected_timeframe=config.backtest.timeframe,
        regime_key=learning_context.regime_key,
        global_regime_key=learning_context.global_regime_key,
        region=learning_context.region,
    )
    structural_signature = PatternMemoryService().extract_signature(candidate.expression)
    result = SimulationResult(
        expression=candidate.expression,
        job_id=f"job-{candidate.alpha_id}",
        status="completed",
        region=outcome_region,
        universe="TOP3000",
        delay=1,
        neutralization="sector",
        decay=0,
        metrics={
            "sharpe": 1.2,
            "fitness": 1.1,
            "turnover": 0.4,
            "drawdown": 0.2,
            "returns": 0.08,
            "margin": 0.04,
        },
        submission_eligible=True,
        rejection_reason=None,
        raw_result={},
        simulated_at="2026-01-01T00:05:00+00:00",
        candidate_id=candidate.alpha_id,
        batch_id=f"batch-{candidate.alpha_id}",
        run_id=environment.context.run_id,
        round_index=1,
        backend="manual",
    )
    repository.alpha_history.persist_brain_outcomes(
        run_id=environment.context.run_id,
        regime_key=learning_context.regime_key,
        region=learning_context.region,
        global_regime_key=learning_context.global_regime_key,
        entries=[
            {
                "candidate": candidate,
                "result": result,
                "diagnosis": AlphaDiagnosis(success_tags=["selected_top_alpha"], fail_tags=[]),
                "structural_signature": structural_signature,
                "gene_ids": [],
                "outcome_score": 0.9,
                "behavioral_novelty_score": 0.7,
                "passed_filters": True,
                "selected": True,
                "metric_source": "external_brain",
            }
        ],
        pattern_decay=config.adaptive_generation.pattern_decay,
        prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
        created_at="2026-01-01T00:10:00+00:00",
    )


def _init_environment(repository: SQLiteRepository, config, command_name: str) -> CommandEnvironment:
    context = RunContext.create(seed=11, config_path="config/dev.yaml")
    environment = CommandEnvironment(
        config_path="config/dev.yaml",
        command_name=command_name,
        context=context,
    )
    init_run(repository, config, environment, status="running")
    return environment
