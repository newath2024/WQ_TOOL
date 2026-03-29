from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from adapters.brain_manual_adapter import BrainManualAdapter
from adapters.simulation_adapter import SimulationAdapter
from core.config import load_config
from core.run_context import RunContext
from data.field_registry import FieldRegistry, FieldSpec
from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemorySnapshot
from services.brain_service import BrainService
from services.candidate_selection_service import CandidateSelectionService
from services.closed_loop_service import ClosedLoopService
from services.models import CommandEnvironment, SimulationJob
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
    assert batch.results[0].rejection_reason == "poll_timeout"


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


def _init_environment(repository: SQLiteRepository, config, command_name: str) -> CommandEnvironment:
    context = RunContext.create(seed=11, config_path="config/dev.yaml")
    environment = CommandEnvironment(
        config_path="config/dev.yaml",
        command_name=command_name,
        context=context,
    )
    init_run(repository, config, environment, status="running")
    return environment
