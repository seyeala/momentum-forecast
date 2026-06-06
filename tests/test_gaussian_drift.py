import pytest

from momentum_forecast.models import known_drift_profit_probability


def test_known_drift_probability_is_half_at_breakeven():
    assert known_drift_profit_probability(0.001, 0.01, 10, 0.01) == pytest.approx(0.5)
    assert known_drift_profit_probability(-0.001, 0.01, 10, 0.01, side="short") == pytest.approx(0.5)


def test_known_drift_probability_increases_with_long_drift():
    low = known_drift_profit_probability(0.0, 0.01, 5, 0.001)
    high = known_drift_profit_probability(0.002, 0.01, 5, 0.001)
    assert high > low


def test_known_drift_probability_decreases_with_cost():
    cheap = known_drift_profit_probability(0.001, 0.01, 5, 0.0)
    expensive = known_drift_profit_probability(0.001, 0.01, 5, 0.01)
    assert expensive < cheap


@pytest.mark.parametrize(
    "kwargs",
    [
        {"volatility": 0.0},
        {"horizon": 0.0},
        {"cost": -0.1},
        {"side": "flat"},
    ],
)
def test_known_drift_rejects_invalid_inputs(kwargs):
    base = dict(drift=0.0, volatility=0.01, horizon=1.0, cost=0.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        known_drift_profit_probability(**base)
