import pytest

from momentum_forecast.fitting import fit_ewma_volatility, fit_return_variance, fit_return_volatility
from tests.helpers import make_gaussian_returns


def test_fit_return_variance_formula():
    fit = fit_return_variance([1.0, 2.0, 3.0])
    assert fit.variance == pytest.approx(1.0)
    assert fit.volatility == pytest.approx(1.0)
    assert fit.lookback == 3


def test_fit_return_volatility_recovers_synthetic_variance():
    returns = make_gaussian_returns(mu=0.0, sigma=0.02, n=30_000, seed=3)
    fit = fit_return_volatility(returns)
    assert fit.variance == pytest.approx(0.02**2, rel=0.04)
    assert fit.volatility == pytest.approx(0.02, rel=0.02)


def test_fit_ewma_volatility_hand_computable_sequence():
    fit = fit_ewma_volatility([0.01, 0.03], alpha=0.25)
    expected_variance = 0.25 * 0.03**2 + 0.75 * 0.01**2
    assert fit.variance == pytest.approx(expected_variance)
    assert "0.25" in fit.method


@pytest.mark.parametrize("values", [[], [0.1]])
def test_fit_return_variance_rejects_too_few_values(values):
    with pytest.raises(ValueError):
        fit_return_variance(values)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1])
def test_fit_ewma_volatility_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        fit_ewma_volatility([0.1], alpha)
