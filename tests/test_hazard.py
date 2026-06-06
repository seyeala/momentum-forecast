import pytest

from momentum_forecast.models import expected_regime_life, hazard_adjusted_drift_gain, regime_survival_probability


def test_regime_survival_probability_and_expected_life():
    assert regime_survival_probability(0.2, 0.0) == pytest.approx(1.0)
    assert regime_survival_probability(0.2, 5.0) == pytest.approx(0.36787944117144233)
    assert expected_regime_life(0.2) == pytest.approx(5.0)


def test_hazard_adjusted_drift_gain_small_hazard_limit():
    assert hazard_adjusted_drift_gain(0.001, 1e-9, 10.0) == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("func", "args"),
    [
        (regime_survival_probability, (0.0, 1.0)),
        (regime_survival_probability, (0.1, -1.0)),
        (expected_regime_life, (0.0,)),
        (hazard_adjusted_drift_gain, (0.001, 0.0, 1.0)),
        (hazard_adjusted_drift_gain, (0.001, 0.1, -1.0)),
    ],
)
def test_hazard_functions_reject_invalid_inputs(func, args):
    with pytest.raises(ValueError):
        func(*args)
