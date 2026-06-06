import pytest

from momentum_forecast.models import channel_survival_probability, expected_channel_life


def test_expected_channel_life_formula():
    assert expected_channel_life(0.02, 0.01) == pytest.approx(4.0)


def test_channel_survival_probability_is_bounded_and_decreases_with_horizon():
    near = channel_survival_probability(1.0, 0.03, 0.01)
    far = channel_survival_probability(10.0, 0.03, 0.01)
    assert 0.0 <= far <= near <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"band": 0.0},
        {"residual_volatility": 0.0},
    ],
)
def test_expected_channel_life_rejects_invalid_inputs(kwargs):
    base = dict(band=0.01, residual_volatility=0.01)
    base.update(kwargs)
    with pytest.raises(ValueError):
        expected_channel_life(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0.0},
        {"band": 0.0},
        {"residual_volatility": 0.0},
        {"terms": 0},
    ],
)
def test_channel_survival_probability_rejects_invalid_inputs(kwargs):
    base = dict(horizon=1.0, band=0.01, residual_volatility=0.01)
    base.update(kwargs)
    with pytest.raises(ValueError):
        channel_survival_probability(**base)
