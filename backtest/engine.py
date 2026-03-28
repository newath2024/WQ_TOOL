from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtest.costs import compute_turnover, transaction_costs
from backtest.neutralization import apply_neutralization
from backtest.portfolio import apply_holding_period, cross_sectional_bucket_weights, symbol_level_weights
from backtest.simulation import (
    SimulationProfile,
    apply_signal_controls,
    apply_weight_clip,
    resolve_simulation_profile,
)
from core.config import BacktestConfig, SimulationConfig
from features.operators import safe_divide
from features.transforms import ResearchMatrices


@dataclass(slots=True)
class BacktestArtifacts:
    signal: pd.DataFrame
    neutralized_signal: pd.DataFrame
    target_weights: pd.DataFrame
    positions: pd.DataFrame
    forward_returns: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    cumulative_returns: pd.Series
    simulation_profile: dict
    subuniverse_returns: dict[str, pd.Series] = field(default_factory=dict)


def run_backtest(
    signal: pd.DataFrame,
    matrices: ResearchMatrices,
    backtest_config: BacktestConfig,
    simulation_config: SimulationConfig,
    universe_mask: pd.DataFrame | None = None,
) -> BacktestArtifacts:
    close_prices = matrices.numeric_fields["close"]
    profile = resolve_simulation_profile(simulation_config, backtest_config)
    working_signal = apply_signal_controls(signal.copy(), close_prices, profile)
    if universe_mask is not None:
        working_signal = working_signal.where(universe_mask)

    neutralized_signal = apply_neutralization(
        working_signal,
        mode=profile.neutralization,
        matrices=matrices,
        factor_columns=simulation_config.factor_columns,
    )
    if profile.secondary_neutralization:
        neutralized_signal = apply_neutralization(
            neutralized_signal,
            mode=profile.secondary_neutralization,
            matrices=matrices,
            factor_columns=simulation_config.factor_columns,
        )
    if universe_mask is not None:
        neutralized_signal = neutralized_signal.where(universe_mask)

    close_returns = close_prices.pct_change()
    portfolio_signal = neutralized_signal
    if backtest_config.volatility_scaling:
        rolling_vol = close_returns.rolling(
            window=backtest_config.volatility_lookback,
            min_periods=backtest_config.volatility_lookback,
        ).std(ddof=0)
        portfolio_signal = safe_divide(portfolio_signal, rolling_vol)

    if backtest_config.mode == "cross_sectional":
        target_weights = cross_sectional_bucket_weights(
            scores=portfolio_signal,
            selection_fraction=backtest_config.selection_fraction,
            construction=backtest_config.portfolio_construction,
        )
    else:
        target_weights = symbol_level_weights(
            scores=portfolio_signal,
            construction=backtest_config.portfolio_construction,
            rank_window=backtest_config.symbol_rank_window,
            upper_quantile=backtest_config.upper_quantile,
            lower_quantile=backtest_config.lower_quantile,
        )

    if universe_mask is not None:
        target_weights = target_weights.where(universe_mask, 0.0)
    target_weights = apply_weight_clip(target_weights, profile.weight_clip)
    delayed_weights = target_weights.shift(profile.effective_signal_delay).fillna(0.0)
    positions = apply_holding_period(delayed_weights, profile.effective_holding_period)
    if universe_mask is not None:
        positions = positions.where(universe_mask, 0.0)

    forward_returns = close_returns.shift(-1)
    gross_returns = positions.mul(forward_returns, fill_value=0.0).sum(axis=1)
    turnover = compute_turnover(positions)
    costs = transaction_costs(turnover, backtest_config.transaction_cost_bps)
    net_returns = gross_returns - costs
    cumulative_returns = (1.0 + net_returns.fillna(0.0)).cumprod() - 1.0

    return BacktestArtifacts(
        signal=working_signal,
        neutralized_signal=neutralized_signal,
        target_weights=target_weights,
        positions=positions,
        forward_returns=forward_returns,
        gross_returns=gross_returns,
        net_returns=net_returns,
        turnover=turnover,
        cumulative_returns=cumulative_returns,
        simulation_profile=profile.to_dict(),
    )
