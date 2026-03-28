from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import compute_performance_metrics
from backtest.neutralization import factor_neutralize
from backtest.simulation import resolve_simulation_profile
from core.config import BacktestConfig, SimulationConfig
from features.transforms import ResearchMatrices


def test_performance_metrics_computation() -> None:
    index = pd.date_range("2021-01-01", periods=4, freq="B")
    daily_returns = pd.Series([0.01, -0.005, 0.02, 0.0], index=index)
    turnover = pd.Series([0.2, 0.1, 0.15, 0.1], index=index)

    metrics = compute_performance_metrics(
        daily_returns=daily_returns,
        turnover=turnover,
        annualization_factor=252,
        turnover_penalty=0.1,
        drawdown_penalty=0.5,
    )

    assert metrics.observation_count == 4
    assert np.isclose(metrics.average_return, daily_returns.mean())
    assert np.isclose(metrics.turnover, turnover.mean())
    assert metrics.cumulative_return > 0


def test_simulation_delay_profiles_change_position_timing() -> None:
    index = pd.date_range("2021-01-01", periods=5, freq="B")
    close = pd.DataFrame(
        {"AAA": [100.0, 101.0, 102.0, 103.0, 104.0], "BBB": [100.0, 99.0, 98.0, 97.0, 96.0]},
        index=index,
    )
    signal = pd.DataFrame({"AAA": [1.0] * 5, "BBB": [0.0] * 5}, index=index)
    matrices = ResearchMatrices(numeric_fields={"close": close})
    backtest = BacktestConfig(
        timeframe="1d",
        mode="cross_sectional",
        portfolio_construction="long_only",
        selection_fraction=0.5,
        signal_delay=1,
        holding_period=3,
        volatility_scaling=False,
        transaction_cost_bps=0.0,
    )

    d0 = run_backtest(signal, matrices, backtest, SimulationConfig(delay_mode="d0"))
    d1 = run_backtest(signal, matrices, backtest, SimulationConfig(delay_mode="d1"))
    fast = run_backtest(signal, matrices, backtest, SimulationConfig(delay_mode="fast_d1"))

    assert resolve_simulation_profile(SimulationConfig(delay_mode="d0"), backtest).effective_signal_delay == 0
    assert resolve_simulation_profile(SimulationConfig(delay_mode="fast_d1"), backtest).effective_holding_period == 1
    assert d0.positions.iloc[0].abs().sum() > 0
    assert d1.positions.iloc[0].abs().sum() == 0
    assert d1.positions.iloc[1].abs().sum() > 0
    assert fast.positions.equals(fast.target_weights.shift(1).fillna(0.0))


def test_factor_neutralization_removes_linear_exposure() -> None:
    index = pd.date_range("2021-01-01", periods=2, freq="B")
    signal = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
        index=index,
        columns=["AAA", "BBB", "CCC", "DDD"],
    )
    beta = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        index=index,
        columns=signal.columns,
    )

    residual = factor_neutralize(signal, {"beta": beta})

    assert residual.abs().max().max() < 1e-8
