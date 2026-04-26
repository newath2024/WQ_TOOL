from __future__ import annotations

import sqlite3


from storage.models import (
    MutationOutcomeRecord,
)


class MutationRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def save_mutation_outcomes(self, records: list[MutationOutcomeRecord]) -> None:
        for record in records:
            self.connection.execute(
                """
                INSERT INTO mutation_outcomes
                (run_id, child_alpha_id, parent_alpha_id, parent_run_id, mutation_mode, family_signature,
                 effective_regime_key, outcome_source, parent_post_sim_score, child_post_sim_score, outcome_delta,
                 parent_quality_score, child_quality_score, quality_delta, selected_for_simulation,
                 selected_for_mutation, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, child_alpha_id, parent_alpha_id, outcome_source) DO UPDATE SET
                    parent_run_id = excluded.parent_run_id,
                    mutation_mode = excluded.mutation_mode,
                    family_signature = excluded.family_signature,
                    effective_regime_key = excluded.effective_regime_key,
                    parent_post_sim_score = excluded.parent_post_sim_score,
                    child_post_sim_score = excluded.child_post_sim_score,
                    outcome_delta = excluded.outcome_delta,
                    parent_quality_score = excluded.parent_quality_score,
                    child_quality_score = excluded.child_quality_score,
                    quality_delta = excluded.quality_delta,
                    selected_for_simulation = excluded.selected_for_simulation,
                    selected_for_mutation = excluded.selected_for_mutation,
                    created_at = excluded.created_at
                """,
                (
                    record.run_id,
                    record.child_alpha_id,
                    record.parent_alpha_id,
                    record.parent_run_id,
                    record.mutation_mode,
                    record.family_signature,
                    record.effective_regime_key,
                    record.outcome_source,
                    record.parent_post_sim_score,
                    record.child_post_sim_score,
                    record.outcome_delta,
                    record.parent_quality_score,
                    record.child_quality_score,
                    record.quality_delta,
                    int(record.selected_for_simulation),
                    int(record.selected_for_mutation),
                    record.created_at,
                ),
            )
        self.connection.commit()

    def list_mutation_outcomes(
        self,
        *,
        effective_regime_key: str | None = None,
        family_signature: str | None = None,
    ) -> list[dict]:
        clauses = ["1 = 1"]
        params: list[object] = []
        if effective_regime_key:
            clauses.append("effective_regime_key = ?")
            params.append(effective_regime_key)
        if family_signature:
            clauses.append("family_signature = ?")
            params.append(family_signature)
        rows = self.connection.execute(
            f"""
            SELECT *
            FROM mutation_outcomes
            WHERE {" AND ".join(clauses)}
            ORDER BY created_at DESC
            """,
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_mutation_outcomes_with_motif(
        self,
        *,
        effective_regime_key: str | None = None,
        limit: int = 2000,
    ) -> list[dict]:
        """Return structural mutation outcomes enriched with the child alpha motif.

        ``child_motif`` is lifted from ``alpha_cases.motif`` so that
        ``MutationPolicy._motif_success_weights()`` can bias structural mutation
        toward historically high-success motifs without a raw SQL JOIN inside the
        policy layer.
        """
        clauses = ["mo.mutation_mode = 'structural'"]
        params: list[object] = []
        if effective_regime_key:
            clauses.append("mo.effective_regime_key = ?")
            params.append(effective_regime_key)
        params.append(limit)
        rows = self.connection.execute(
            f"""
            SELECT
                mo.mutation_mode,
                mo.family_signature,
                mo.effective_regime_key,
                mo.outcome_delta,
                mo.selected_for_simulation,
                mo.selected_for_mutation,
                mo.created_at,
                COALESCE(ac.motif, '') AS child_motif
            FROM mutation_outcomes mo
            LEFT JOIN alpha_cases ac
                ON ac.run_id  = mo.run_id
               AND ac.alpha_id = mo.child_alpha_id
            WHERE {" AND ".join(clauses)}
            ORDER BY mo.created_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]
