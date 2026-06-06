"""Momentum-line forecasting formula library."""

from momentum_forecast.costs import TradingCost
from momentum_forecast.data import bps_to_log_cost, log_prices, log_returns, simple_to_log_return
from momentum_forecast.models import (
    ar1_expected_cumulative_return,
    ar1_expected_return_at_horizon,
    ar1_half_life,
    channel_survival_probability,
    estimated_drift_profit_probability,
    expected_channel_life,
    expected_regime_life,
    hazard_adjusted_drift_gain,
    known_drift_profit_probability,
    regime_survival_probability,
)
from momentum_forecast.signals import classify_momentum_strength, momentum_z_score
from momentum_forecast.types import MomentumForecast

__all__ = [
    "MomentumForecast",
    "TradingCost",
    "ar1_expected_cumulative_return",
    "ar1_expected_return_at_horizon",
    "ar1_half_life",
    "bps_to_log_cost",
    "channel_survival_probability",
    "classify_momentum_strength",
    "estimated_drift_profit_probability",
    "expected_channel_life",
    "expected_regime_life",
    "hazard_adjusted_drift_gain",
    "known_drift_profit_probability",
    "log_prices",
    "log_returns",
    "momentum_z_score",
    "regime_survival_probability",
    "simple_to_log_return",
]
