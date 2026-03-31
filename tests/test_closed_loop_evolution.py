from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from adapters.simulation_adapter import SimulationAdapter
from core.config import load_config
from core.run_context import RunContext
from services.brain_service import BrainService
from services.closed_loop_service import ClosedLoopService
from services.models import CommandEnvironment
from services.runtime_service import init_run
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
                    "sharpe": 1.20,
                    "fitness": 1.05,
                    "turnover": 0.35,
                    "drawdown": 0.10,
                    "returns": 0.07,
                    "margin": 0.05,
                },
                "submission_eligible": True,
            },
        }


def test_closed_loop_persists_pre_sim_and_learning_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "market.csv"
    metadata_path = tmp_path / "metadata.csv"
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
        run_id = environment.context.run_id
        duplicate_rows = repository.get_duplicate_decision_summary(run_id)
        stage_rows = repository.get_stage_metrics(run_id)
        selection_rows = repository.list_selection_scores(run_id)
        regime_snapshot = repository.get_latest_regime_snapshot(run_id)
        mutation_outcomes = repository.list_mutation_outcomes()
    finally:
        repository.close()

    assert summary.status == "completed"
    assert duplicate_rows
    assert any(row["stage"] == "pre_sim" for row in stage_rows)
    assert any(row["score_stage"] == "pre_sim" for row in selection_rows)
    assert any(row["score_stage"] == "mutation_parent" for row in selection_rows)
    assert regime_snapshot is not None
    assert isinstance(mutation_outcomes, list)


def test_closed_loop_writes_progress_log_jsonl(tmp_path: Path) -> None:
    data_path = tmp_path / "market.csv"
    metadata_path = tmp_path / "metadata.csv"
    write_sample_csv(data_path)
    write_sample_metadata_csv(metadata_path)

    config = load_config("config/dev.yaml")
    config.data.path = str(data_path)
    config.aux_data.group_path = str(metadata_path)
    config.aux_data.factor_path = str(metadata_path)
    config.aux_data.mask_path = str(metadata_path)
    config.storage.path = str(tmp_path / "closed_loop_progress.sqlite3")
    config.runtime.log_level = "WARNING"
    config.runtime.progress_log_dir = str(tmp_path / "progress")
    config.loop.rounds = 1
    config.loop.generation_batch_size = 4
    config.loop.simulation_batch_size = 2
    config.loop.mutate_top_k = 1
    config.loop.max_children_per_parent = 1
    config.generation.template_count = 4
    config.generation.grammar_count = 0
    config.generation.mutation_count = 1

    repository = SQLiteRepository(config.storage.path)
    try:
        environment = _init_environment(repository, config, "run-closed-loop")
        service = ClosedLoopService(
            repository,
            brain_service=BrainService(repository, config.brain, adapter=FakeCompletedAdapter()),
        )
        summary = service.run(config=config, environment=environment)
        log_path = tmp_path / "progress" / f"{environment.context.run_id}.jsonl"
        progress_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    finally:
        repository.close()

    assert summary.progress_log_path == str(log_path)
    assert any(row["event"] == "closed_loop_run_started" for row in progress_rows)
    assert any(row["event"] == "closed_loop_round_completed" for row in progress_rows)
    assert progress_rows[-1]["event"] == "closed_loop_run_finished"


def _init_environment(repository: SQLiteRepository, config, command_name: str) -> CommandEnvironment:
    context = RunContext.create(seed=11, config_path="config/dev.yaml")
    environment = CommandEnvironment(
        config_path="config/dev.yaml",
        command_name=command_name,
        context=context,
    )
    init_run(repository, config, environment, status="running")
    return environment
