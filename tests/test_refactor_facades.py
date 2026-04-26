from __future__ import annotations

import config
import core.config as core_config
import domain
from config.builders import _build_service_config
from config.loader import load_config
from config.models.runtime import AppConfig, RuntimeConfig
from config.models.service import ServiceConfig
from config.validators import _normalize_brain_enum
from domain.brain import BrainResultRecord
from domain.candidate import AlphaCandidate
from domain.metrics import ObjectiveVector
from domain.simulation import SimulationResult
from services.service_runtime import ServiceWorker as RuntimeServiceWorker
from services.service_runtime.worker import ServiceWorker as WorkerImplementation
from services.service_worker import ServiceWorker as LegacyServiceWorker
from storage.models import BrainResultRecord as StorageBrainResultRecord
from storage.repository import SQLiteRepository


def test_domain_package_reexports_domain_models() -> None:
    assert domain.AlphaCandidate is AlphaCandidate
    assert domain.BrainResultRecord is BrainResultRecord
    assert domain.ObjectiveVector is ObjectiveVector
    assert domain.SimulationResult is SimulationResult


def test_service_worker_facades_point_to_runtime_worker() -> None:
    assert LegacyServiceWorker is WorkerImplementation
    assert RuntimeServiceWorker is WorkerImplementation


def test_sqlite_repository_facade_delegates_to_child_repositories(monkeypatch) -> None:
    repository = SQLiteRepository(":memory:")
    try:
        run_sentinel = object()
        expressions_sentinel = {"rank(close)"}

        def fake_get_run(run_id: str) -> object:
            assert run_id == "run-1"
            return run_sentinel

        def fake_list_existing_normalized_expressions(run_id: str) -> set[str]:
            assert run_id == "run-1"
            return expressions_sentinel

        monkeypatch.setattr(repository.runs, "get_run", fake_get_run)
        monkeypatch.setattr(
            repository.alphas,
            "list_existing_normalized_expressions",
            fake_list_existing_normalized_expressions,
        )

        assert repository.get_run("run-1") is run_sentinel
        assert repository.list_existing_normalized_expressions("run-1") is expressions_sentinel
    finally:
        repository.close()


def test_core_config_keeps_reexport_compatibility() -> None:
    assert config.load_config is load_config
    assert core_config.load_config is load_config
    assert core_config.AppConfig is AppConfig
    assert core_config.RuntimeConfig is RuntimeConfig
    assert core_config.ServiceConfig is ServiceConfig
    assert core_config._build_service_config is _build_service_config
    assert core_config._normalize_brain_enum is _normalize_brain_enum
    assert StorageBrainResultRecord is BrainResultRecord

    for exported_name in (
        "load_config",
        "AppConfig",
        "RuntimeConfig",
        "ServiceConfig",
        "_build_service_config",
        "_normalize_brain_enum",
    ):
        assert exported_name in core_config.__all__
