from __future__ import annotations

from abc import ABC, abstractmethod


class SimulationAdapter(ABC):
    @abstractmethod
    def submit_simulation(self, expression: str, sim_config: dict) -> dict:
        """Submit a single expression for external simulation."""

    @abstractmethod
    def get_simulation_status(self, job_id: str) -> dict:
        """Fetch the latest known job status."""

    @abstractmethod
    def get_simulation_result(self, job_id: str) -> dict:
        """Fetch the terminal result payload for a completed or rejected job."""

    def batch_submit(self, expressions: list[str], sim_config: dict) -> list[dict]:
        """Submit multiple expressions. Adapters may override this for native batching."""
        return [self.submit_simulation(expression, sim_config) for expression in expressions]
