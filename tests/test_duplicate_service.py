from __future__ import annotations

import json

from core.config import DuplicateConfig
from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemoryService
from services.duplicate_service import DuplicateService
from storage.repository import SQLiteRepository


def test_exact_same_run_duplicate_is_hard_blocked() -> None:
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
        existing = _candidate("alpha-existing", "rank(ts_mean(close, 5))")
        incoming = _candidate("alpha-new", "rank(ts_mean(close, 5))")
        repository.save_alpha_candidates("run-1", [existing])

        service = DuplicateService(repository, config=DuplicateConfig())
        result = service.filter_pre_sim(
            [incoming],
            run_id="run-1",
            round_index=1,
            legacy_regime_key="legacy",
            global_regime_key="global",
        )
    finally:
        repository.close()

    assert not result.kept_candidates
    assert len(result.blocked_candidates) == 1
    assert result.decisions[0].reason_code == "exact_same_run"


def test_near_structural_same_run_duplicate_is_blocked() -> None:
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
        candidates = [
            _candidate("alpha-1", "rank(ts_mean(close, 5))"),
            _candidate("alpha-2", "rank(ts_mean(close, 6))"),
        ]
        service = DuplicateService(
            repository,
            config=DuplicateConfig(same_run_structural_threshold=0.90),
        )
        result = service.filter_pre_sim(
            candidates,
            run_id="run-1",
            round_index=1,
            legacy_regime_key="legacy",
            global_regime_key="global",
        )
    finally:
        repository.close()

    assert len(result.kept_candidates) == 1
    assert len(result.blocked_candidates) == 1
    assert result.decisions[-1].reason_code == "near_structural_same_run"


def test_cross_run_duplicate_respects_global_scope() -> None:
    repository = SQLiteRepository(":memory:")
    memory_service = PatternMemoryService()
    try:
        _seed_alpha_history(
            repository,
            run_id="prior-run",
            alpha_id="alpha-prior",
            expression="rank(ts_mean(close, 5))",
            global_regime_key="global-a",
            memory_service=memory_service,
        )
        candidate = _candidate("alpha-new", "rank(ts_mean(close, 5))")
        service = DuplicateService(repository, config=DuplicateConfig())
        blocked = service.filter_pre_sim(
            [candidate],
            run_id="run-2",
            round_index=1,
            legacy_regime_key="legacy",
            global_regime_key="global-a",
        )
        kept = service.filter_pre_sim(
            [candidate],
            run_id="run-2",
            round_index=1,
            legacy_regime_key="legacy",
            global_regime_key="global-b",
        )
    finally:
        repository.close()

    assert blocked.decisions[0].reason_code == "exact_cross_run"
    assert len(blocked.blocked_candidates) == 1
    assert kept.decisions[0].decision == "kept"
    assert len(kept.kept_candidates) == 1


def test_save_alpha_candidates_persists_structural_signature() -> None:
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
        repository.save_alpha_candidates("run-1", [_candidate("alpha-1", "rank(ts_mean(close, 5))")])
        row = repository.connection.execute(
            "SELECT structural_signature_json FROM alphas WHERE run_id = ? AND alpha_id = ?",
            ("run-1", "alpha-1"),
        ).fetchone()
    finally:
        repository.close()

    assert row is not None
    assert json.loads(row["structural_signature_json"])["family_signature"]


def test_exact_same_run_duplicate_does_not_require_structural_window() -> None:
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
        existing = _candidate("alpha-existing", "rank(ts_mean(close, 5))")
        incoming = _candidate("alpha-new", "rank(ts_mean(close, 5))")
        repository.save_alpha_candidates("run-1", [existing])
        service = DuplicateService(
            repository,
            config=DuplicateConfig(same_run_structural_reference_limit=0),
        )

        result = service.filter_pre_sim(
            [incoming],
            run_id="run-1",
            round_index=1,
            legacy_regime_key="legacy",
            global_regime_key="global",
        )
    finally:
        repository.close()

    assert not result.kept_candidates
    assert result.decisions[0].reason_code == "exact_same_run"


def test_pre_sim_dedup_uses_targeted_repository_paths(monkeypatch) -> None:
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
        repository.save_alpha_candidates("run-1", [_candidate("alpha-existing", "rank(ts_mean(open, 5))")])

        def fail_full_scan(run_id: str):  # noqa: ARG001
            raise AssertionError("pre-sim dedup should not load every alpha record")

        calls: list[int] = []
        original_refs = repository.get_same_run_structural_references

        def track_refs(*, run_id: str, limit: int):
            calls.append(limit)
            return original_refs(run_id=run_id, limit=limit)

        monkeypatch.setattr(repository, "list_alpha_records", fail_full_scan)
        monkeypatch.setattr(repository, "get_same_run_structural_references", track_refs)
        service = DuplicateService(
            repository,
            config=DuplicateConfig(same_run_structural_reference_limit=1),
        )

        result = service.filter_pre_sim(
            [_candidate("alpha-new", "rank(ts_mean(close, 6))")],
            run_id="run-1",
            round_index=1,
            legacy_regime_key="legacy",
            global_regime_key="global",
        )
    finally:
        repository.close()

    assert result.decisions
    assert calls == [1]


def _candidate(alpha_id: str, expression: str) -> AlphaCandidate:
    return AlphaCandidate(
        alpha_id=alpha_id,
        expression=expression,
        normalized_expression=expression,
        generation_mode="template",
        parent_ids=(),
        complexity=3,
        created_at="2026-01-01T00:00:00+00:00",
        template_name="template",
        fields_used=("close",),
        operators_used=("rank", "ts_mean"),
        depth=2,
        generation_metadata={},
    )


def _seed_alpha_history(
    repository: SQLiteRepository,
    *,
    run_id: str,
    alpha_id: str,
    expression: str,
    global_regime_key: str,
    memory_service: PatternMemoryService,
) -> None:
    repository.upsert_run(
        run_id=run_id,
        seed=7,
        config_path="config/dev.yaml",
        config_snapshot="{}",
        status="running",
        started_at="2026-01-01T00:00:00+00:00",
    )
    signature = memory_service.extract_signature(expression, generation_metadata={})
    repository.connection.execute(
        """
        INSERT INTO alpha_history
        (run_id, alpha_id, region, regime_key, global_regime_key, market_regime_key, effective_regime_key,
         regime_label, regime_confidence, expression, normalized_expression, generation_mode, generation_metadata_json,
         parent_refs_json, structural_signature_json, gene_ids_json, train_metrics_json, validation_metrics_json,
         test_metrics_json, validation_signal_json, validation_returns_json, outcome_score, behavioral_novelty_score,
         passed_filters, selected, submission_pass_count, diagnosis_summary_json, rejection_reasons_json, metric_source, created_at)
        VALUES (?, ?, '', 'legacy', ?, '', 'legacy', 'unknown', 0.0, ?, ?, 'template', '{}', '[]', ?, '[]', '{}', '{}', '{}', '{}', '{}', 1.0, 0.5, 1, 1, 1, ?, '[]', 'external_brain', '2026-01-01T00:00:00+00:00')
        """,
        (
            run_id,
            alpha_id,
            global_regime_key,
            expression,
            expression,
            json.dumps(signature.to_dict(), sort_keys=True),
            json.dumps({"fail_tags": [], "success_tags": []}, sort_keys=True),
        ),
    )
    repository.connection.commit()
