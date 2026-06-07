import math

import pytest

from momentum_forecast.fitting import (
    fit_channel_band_from_residual_quantile,
    fit_channel_band_from_volatility,
    fit_line_residuals,
)
from tests.helpers import make_linear_log_prices


def test_fit_line_residuals_perfect_line():
    fit = fit_line_residuals([1.0, 1.1, 1.2, 1.3])
    assert fit.slope == pytest.approx(0.1)
    assert fit.residual_variance == pytest.approx(0.0)
    assert fit.residual_volatility == pytest.approx(0.0)
    assert fit.band == pytest.approx(0.0)
    assert math.isinf(fit.expected_life)


def test_fit_line_residuals_noisy_line():
    prices = make_linear_log_prices(slope=0.002, sigma=0.01, n=500, seed=5)
    fit = fit_line_residuals(prices)
    assert fit.slope == pytest.approx(0.002, abs=0.00001)
    assert fit.residual_volatility > 0.0
    assert fit.band == pytest.approx(2 * fit.residual_volatility)
    assert fit.expected_life == pytest.approx(4.0)


def test_fit_channel_band_from_residual_quantile():
    assert fit_channel_band_from_residual_quantile([-3.0, -1.0, 2.0, 4.0], 0.5) == pytest.approx(2.0)
    assert fit_channel_band_from_residual_quantile([-3.0, -1.0, 2.0, 4.0], 0.95) == pytest.approx(4.0)


def test_fit_channel_band_from_volatility():
    assert fit_channel_band_from_volatility(0.2, multiplier=3.0) == pytest.approx(0.6)


@pytest.mark.parametrize("values", [[], [1.0], [1.0, 2.0], [1.0, float("nan"), 2.0]])
def test_fit_line_residuals_rejects_invalid_inputs(values):
    with pytest.raises(ValueError):
        fit_line_residuals(values)


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1])
def test_fit_channel_band_quantile_rejects_invalid_quantile(quantile):
    with pytest.raises(ValueError):
        fit_channel_band_from_residual_quantile([1.0], quantile)


@pytest.mark.parametrize("kwargs", [{"residual_volatility": 0.0}, {"multiplier": 0.0}])
def test_fit_channel_band_volatility_rejects_invalid_parameters(kwargs):
    base = dict(residual_volatility=0.1, multiplier=2.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        fit_channel_band_from_volatility(**base)
