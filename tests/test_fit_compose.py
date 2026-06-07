import pytest

from momentum_forecast.fitting import (
    estimated_drift_probability_from_fits,
    fit_estimated_drift_inputs,
    fit_mean_drift,
    fit_return_variance,
)
from momentum_forecast.models import estimated_drift_profit_probability


def test_fit_estimated_drift_inputs_and_probability_match_direct_call():
    returns = [0.001, 0.002, -0.001, 0.003, 0.002]
    drift_fit, volatility_fit = fit_estimated_drift_inputs(returns)
    bridged = estimated_drift_probability_from_fits(drift_fit, volatility_fit, horizon=5, cost=0.0005)
    direct = estimated_drift_profit_probability(
        drift_fit.drift,
        volatility_fit.variance,
        drift_fit.lookback,
        horizon=5,
        cost=0.0005,
    )
    assert bridged == pytest.approx(direct)


def test_estimated_drift_probability_from_fits_rejects_mismatched_lookback():
    with pytest.raises(ValueError):
        estimated_drift_probability_from_fits(
            fit_mean_drift([1.0, 2.0]),
            fit_return_variance([1.0, 2.0, 3.0]),
            horizon=1,
            cost=0.0,
        )
