from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from adapters.brain_api_adapter import PersonaVerificationRequired
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


class FakePersonaAdapter(FakeCompletedAdapter):
    def __init__(
        self,
        persona_url: str = "https://persona.example/scan",
        *,
        persona_confirmation_supported: bool = True,
        persona_confirmation_plan: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self.persona_url = persona_url
        self.auth_calls = 0
        self.resume_calls = 0
        self.persona_notifications: list[str] = []
        self.persona_confirmation_supported = persona_confirmation_supported
        self.persona_confirmation_plan = list(persona_confirmation_plan or [])
        self.persona_confirmation_prompts: list[tuple[str, str]] = []
        self.persona_confirmation_polls: list[tuple[str, int | None]] = []
        self.persona_timeout_seconds = 1800
        self._authenticated = False

    def probe_authenticated_session(self) -> dict:
        if self._authenticated:
            return {"authenticated": True, "mode": "session_cookie", "session_path": "session.json"}
        return {"authenticated": False, "mode": "not_authenticated", "session_path": None}

    def ensure_authenticated(self, **kwargs) -> dict:
        del kwargs
        self.auth_calls += 1
        raise PersonaVerificationRequired(self.persona_url)

    def resume_persona_authentication(self, persona_url: str) -> dict:
        self.resume_calls += 1
        self._authenticated = True
        return {"status": "ready", "mode": "session_cookie", "session_path": "session.json", "persona_url": persona_url}

    def send_persona_notification(self, persona_url: str) -> bool:
        self.persona_notifications.append(persona_url)
        return True

    def supports_persona_confirmation(self) -> bool:
        return self.persona_confirmation_supported

    def send_persona_confirmation_prompt(self, *, prompt_token: str, service_name: str) -> bool:
        self.persona_confirmation_prompts.append((prompt_token, service_name))
        return self.persona_confirmation_supported

    def poll_persona_confirmation(self, *, prompt_token: str, last_update_id: int | None = None) -> dict:
        self.persona_confirmation_polls.append((prompt_token, last_update_id))
        item = (
            self.persona_confirmation_plan.pop(0)
            if self.persona_confirmation_plan
            else {"supported": self.persona_confirmation_supported, "approved": False, "declined": False}
        )
        payload = dict(item)
        payload.setdefault("supported", self.persona_confirmation_supported)
        payload.setdefault("approved", False)
        payload.setdefault("declined", False)
        payload.setdefault("last_update_id", last_update_id)
        return payload


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


def test_closed_loop_waits_for_telegram_confirmation_before_requesting_persona_link(tmp_path: Path) -> None:
    data_path = tmp_path / "market.csv"
    metadata_path = tmp_path / "metadata.csv"
    write_sample_csv(data_path)
    write_sample_metadata_csv(metadata_path)

    config = load_config("config/dev.yaml")
    config.data.path = str(data_path)
    config.aux_data.group_path = str(metadata_path)
    config.aux_data.factor_path = str(metadata_path)
    config.aux_data.mask_path = str(metadata_path)
    config.storage.path = str(tmp_path / "closed_loop_persona.sqlite3")
    config.runtime.log_level = "WARNING"
    config.loop.rounds = 1
    config.loop.generation_batch_size = 4
    config.loop.simulation_batch_size = 2
    config.loop.mutate_top_k = 1
    config.loop.max_children_per_parent = 1
    config.generation.template_count = 4
    config.generation.grammar_count = 0
    config.generation.mutation_count = 1
    config.service.persona_confirmation_required = True
    config.service.persona_confirmation_poll_interval_seconds = 0
    config.service.persona_confirmation_prompt_cooldown_seconds = 999999
    config.service.persona_retry_interval_seconds = 0

    adapter = FakePersonaAdapter(
        persona_confirmation_plan=[
            {"approved": False, "last_update_id": 101},
            {"approved": True, "last_update_id": 102},
        ]
    )
    repository = SQLiteRepository(config.storage.path)
    try:
        environment = _init_environment(repository, config, "run-closed-loop")
        service = ClosedLoopService(
            repository,
            brain_service=BrainService(repository, config.brain, adapter=adapter),
        )
        summary = service.run(config=config, environment=environment)
    finally:
        repository.close()

    assert summary.status == "completed"
    assert adapter.auth_calls == 1
    assert adapter.resume_calls == 1
    assert len(adapter.persona_confirmation_prompts) == 1
    assert adapter.persona_confirmation_prompts[0][1] == "run-closed-loop"
    assert len(adapter.persona_confirmation_polls) == 2
    assert adapter.persona_notifications == ["https://persona.example/scan"]


def test_closed_loop_calls_list_existing_only_once_per_round(tmp_path: Path, monkeypatch) -> None:
    data_path = tmp_path / "market.csv"
    metadata_path = tmp_path / "metadata.csv"
    write_sample_csv(data_path)
    write_sample_metadata_csv(metadata_path)

    config = load_config("config/dev.yaml")
    config.data.path = str(data_path)
    config.aux_data.group_path = str(metadata_path)
    config.aux_data.factor_path = str(metadata_path)
    config.aux_data.mask_path = str(metadata_path)
    config.storage.path = str(tmp_path / "closed_loop_calls.sqlite3")
    config.runtime.log_level = "WARNING"
    config.loop.rounds = 3
    config.loop.generation_batch_size = 4
    config.loop.simulation_batch_size = 2
    config.loop.mutate_top_k = 1
    config.loop.max_children_per_parent = 1
    config.generation.template_count = 4
    config.generation.grammar_count = 0
    config.generation.mutation_count = 1

    repository = SQLiteRepository(config.storage.path)
    try:
        call_count = 0
        original = repository.list_existing_normalized_expressions

        def counted(run_id: str) -> set[str]:
            nonlocal call_count
            call_count += 1
            return original(run_id)

        monkeypatch.setattr(repository, "list_existing_normalized_expressions", counted)

        environment = _init_environment(repository, config, "run-closed-loop")
        service = ClosedLoopService(
            repository,
            brain_service=BrainService(repository, config.brain, adapter=FakeCompletedAdapter()),
        )
        summary = service.run(config=config, environment=environment)
    finally:
        repository.close()

    assert summary.status == "completed"
    assert call_count == config.loop.rounds


def _init_environment(repository: SQLiteRepository, config, command_name: str) -> CommandEnvironment:
    context = RunContext.create(seed=11, config_path="config/dev.yaml")
    environment = CommandEnvironment(
        config_path="config/dev.yaml",
        command_name=command_name,
        context=context,
    )
    init_run(repository, config, environment, status="running")
    return environment
