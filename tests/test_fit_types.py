from dataclasses import FrozenInstanceError

import pytest

from momentum_forecast.fitting import AR1Fit, ChannelFit, DriftFit, HazardFit, VolatilityFit
from momentum_forecast.models import estimated_drift_profit_probability, known_drift_profit_probability


def test_fit_types_are_immutable_and_formula_compatible():
    drift = DriftFit(0.001, 0.0001, 10.0, 20, "sample_mean")
    vol = VolatilityFit(0.000004, 0.002, 20, "sample_variance")
    hazard = HazardFit(0.1, 10.0, 3, 1, "right_censored_mle")

    assert known_drift_profit_probability(drift.drift, vol.volatility, 5, 0.0) > 0.5
    assert estimated_drift_profit_probability(drift.drift, vol.variance, drift.lookback, 5, 0.0) > 0.5
    assert hazard.expected_life == pytest.approx(10.0)
    with pytest.raises(FrozenInstanceError):
        drift.drift = 0.0


def test_fit_types_validate_inputs():
    with pytest.raises(ValueError):
        DriftFit(float("nan"), None, None, 10, "sample_mean")
    with pytest.raises(ValueError):
        VolatilityFit(-1.0, 0.0, 10, "sample_variance")
    with pytest.raises(ValueError):
        AR1Fit(0.5, -1.0, None, None, None, 10)
    with pytest.raises(ValueError):
        ChannelFit(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10, "channel")
    with pytest.raises(ValueError):
        HazardFit(0.1, 10.0, 0, 1, "hazard")
