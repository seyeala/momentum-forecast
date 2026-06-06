import math

import pytest

from momentum_forecast.models import estimated_drift_profit_probability


def test_estimated_drift_probability_is_half_at_breakeven():
    assert estimated_drift_profit_probability(0.001, 0.0001, 10, 5, 0.005) == pytest.approx(0.5)
    assert estimated_drift_profit_probability(-0.001, 0.0001, 10, 5, 0.005, side="short") == pytest.approx(0.5)


def test_estimated_drift_includes_slope_uncertainty_penalty():
    mu_hat = 0.002
    q_hat = 0.0001
    lookback = 10
    horizon = 10
    cost = 0.001
    probability = estimated_drift_profit_probability(mu_hat, q_hat, lookback, horizon, cost)
    z = (mu_hat * horizon - cost) / math.sqrt(q_hat * horizon * (1 + horizon / lookback))
    expected = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    assert probability == pytest.approx(expected)


def test_estimated_drift_probability_penalizes_shorter_lookback():
    short_lookback = estimated_drift_profit_probability(0.002, 0.0001, 5, 10, 0.001)
    long_lookback = estimated_drift_profit_probability(0.002, 0.0001, 100, 10, 0.001)
    assert short_lookback < long_lookback


@pytest.mark.parametrize(
    "kwargs",
    [
        {"q_hat": 0.0},
        {"lookback": 0},
        {"horizon": 0.0},
        {"cost": -0.1},
        {"side": "flat"},
    ],
)
def test_estimated_drift_rejects_invalid_inputs(kwargs):
    base = dict(mu_hat=0.0, q_hat=0.0001, lookback=10, horizon=1.0, cost=0.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        estimated_drift_profit_probability(**base)
