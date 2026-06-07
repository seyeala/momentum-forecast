import pytest

from momentum_forecast.fitting import fit_mean_drift, fit_ols_line_drift
from tests.helpers import make_gaussian_returns, make_linear_log_prices


def test_fit_mean_drift_formula():
    fit = fit_mean_drift([0.01, 0.03, 0.02])
    assert fit.drift == pytest.approx(0.02)
    assert fit.lookback == 3
    assert fit.standard_error == pytest.approx(0.005773502691896258)
    assert fit.t_stat == pytest.approx(fit.drift / fit.standard_error)


def test_fit_mean_drift_recovers_synthetic_mean():
    returns = make_gaussian_returns(mu=0.001, sigma=0.01, n=20_000, seed=1)
    fit = fit_mean_drift(returns)
    assert fit.drift == pytest.approx(0.001, abs=3 * 0.01 / (20_000**0.5))


def test_fit_ols_line_drift_recovers_synthetic_slope():
    prices = make_linear_log_prices(slope=0.002, sigma=0.01, n=500, seed=2)
    fit = fit_ols_line_drift(prices)
    assert fit.drift == pytest.approx(0.002, abs=0.00001)
    assert fit.standard_error is not None
    assert fit.t_stat is not None


@pytest.mark.parametrize("values", [[], [0.1]])
def test_fit_mean_drift_rejects_too_few_values(values):
    with pytest.raises(ValueError):
        fit_mean_drift(values)


@pytest.mark.parametrize("values", [[], [0.1], [0.1, float("nan")]])
def test_fit_ols_line_drift_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        fit_ols_line_drift(values)
