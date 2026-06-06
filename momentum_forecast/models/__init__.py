"""Forecasting model formulas."""

from momentum_forecast.models.ar1 import ar1_expected_cumulative_return, ar1_expected_return_at_horizon, ar1_half_life
from momentum_forecast.models.channel_life import channel_survival_probability, expected_channel_life
from momentum_forecast.models.estimated_drift import estimated_drift_profit_probability
from momentum_forecast.models.gaussian_drift import known_drift_profit_probability
from momentum_forecast.models.hazard import expected_regime_life, hazard_adjusted_drift_gain, regime_survival_probability

__all__ = [
    "ar1_expected_cumulative_return",
    "ar1_expected_return_at_horizon",
    "ar1_half_life",
    "channel_survival_probability",
    "estimated_drift_profit_probability",
    "expected_channel_life",
    "expected_regime_life",
    "hazard_adjusted_drift_gain",
    "known_drift_profit_probability",
    "regime_survival_probability",
]
