from __future__ import annotations

from core.config import load_config
from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemorySnapshot
from services.brain_learning_service import BrainLearningService
from services.models import SimulationResult
from storage.repository import SQLiteRepository


def test_persist_results_skips_poll_timeout_after_downtime_learning_rows() -> None:
    repository = SQLiteRepository(":memory:")
    try:
        config = load_config("config/dev.yaml")
        service = BrainLearningService(repository)
        candidate = AlphaCandidate(
            alpha_id="alpha-timeout",
            expression="rank(close)",
            normalized_expression="rank(close)",
            generation_mode="template",
            parent_ids=(),
            complexity=2,
            created_at="2026-04-21T00:00:00+00:00",
            template_name="momentum",
            fields_used=("close",),
            operators_used=("rank",),
            depth=2,
            generation_metadata={"motif": "momentum"},
        )
        result = SimulationResult(
            expression="rank(close)",
            job_id="job-timeout",
            status="timeout",
            region="USA",
            universe="TOP3000",
            delay=1,
            neutralization="SECTOR",
            decay=0,
            metrics={},
            submission_eligible=None,
            rejection_reason="poll_timeout_after_downtime",
            raw_result={},
            simulated_at="2026-04-21T00:00:00+00:00",
            candidate_id="alpha-timeout",
            batch_id="batch-timeout",
            run_id="run-timeout",
            round_index=1,
        )

        service.persist_results(
            config=config,
            regime_key="regime",
            region="USA",
            global_regime_key="global-regime",
            snapshot=PatternMemorySnapshot(regime_key="regime"),
            candidates_by_id={"alpha-timeout": candidate},
            results=[result],
        )

        history_count = repository.connection.execute("SELECT COUNT(*) AS total FROM alpha_history").fetchone()["total"]
        mutation_count = repository.connection.execute("SELECT COUNT(*) AS total FROM mutation_outcomes").fetchone()["total"]
    finally:
        repository.close()

    assert history_count == 0
    assert mutation_count == 0
