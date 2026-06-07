import pytest

from momentum_forecast.fitting import fit_ar1_phi
from tests.helpers import make_ar1_returns


def test_fit_ar1_phi_formula():
    fit = fit_ar1_phi([1.0, 0.5, 0.25, 0.125])
    assert fit.phi == pytest.approx(0.5)
    assert fit.innovation_variance == pytest.approx(0.0)
    assert fit.half_life == pytest.approx(1.0)


def test_fit_ar1_phi_recovers_synthetic_phi():
    returns = make_ar1_returns(phi=0.7, innovation_sigma=0.01, n=20_000, seed=4)
    fit = fit_ar1_phi(returns)
    assert fit.phi == pytest.approx(0.7, abs=0.03)
    assert fit.standard_error is not None
    assert fit.t_stat is not None


@pytest.mark.parametrize("values", [[], [0.1], [0.1, 0.2]])
def test_fit_ar1_phi_rejects_too_few_values(values):
    with pytest.raises(ValueError):
        fit_ar1_phi(values)
