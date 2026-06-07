"""Empirical fitting utilities for momentum-forecast formula inputs."""

from momentum_forecast.fitting.ar1 import fit_ar1_phi
from momentum_forecast.fitting.channel import (
    fit_channel_band_from_residual_quantile,
    fit_channel_band_from_volatility,
    fit_line_residuals,
)
from momentum_forecast.fitting.compose import estimated_drift_probability_from_fits, fit_estimated_drift_inputs
from momentum_forecast.fitting.drift import fit_mean_drift, fit_ols_line_drift
from momentum_forecast.fitting.hazard import fit_constant_hazard, fit_constant_hazard_with_censoring
from momentum_forecast.fitting.rolling import rolling_ar1_phi, rolling_mean_drift, rolling_return_variance
from momentum_forecast.fitting.types import AR1Fit, ChannelFit, DriftFit, HazardFit, VolatilityFit
from momentum_forecast.fitting.volatility import fit_ewma_volatility, fit_return_variance, fit_return_volatility

__all__ = [
    "AR1Fit",
    "ChannelFit",
    "DriftFit",
    "HazardFit",
    "VolatilityFit",
    "estimated_drift_probability_from_fits",
    "fit_ar1_phi",
    "fit_channel_band_from_residual_quantile",
    "fit_channel_band_from_volatility",
    "fit_constant_hazard",
    "fit_constant_hazard_with_censoring",
    "fit_estimated_drift_inputs",
    "fit_ewma_volatility",
    "fit_line_residuals",
    "fit_mean_drift",
    "fit_ols_line_drift",
    "fit_return_variance",
    "fit_return_volatility",
    "rolling_ar1_phi",
    "rolling_mean_drift",
    "rolling_return_variance",
]
