import pytest

from momentum_forecast.fitting import fit_constant_hazard, fit_constant_hazard_with_censoring
from tests.helpers import make_exponential_lifetimes


def test_fit_constant_hazard_formula():
    fit = fit_constant_hazard([4.0, 6.0])
    assert fit.hazard == pytest.approx(0.2)
    assert fit.expected_life == pytest.approx(5.0)
    assert fit.n_events == 2
    assert fit.n_censored == 0


def test_fit_constant_hazard_with_censoring_formula():
    fit = fit_constant_hazard_with_censoring([4.0, 6.0, 10.0], [True, False, True])
    assert fit.hazard == pytest.approx(2.0 / 20.0)
    assert fit.n_events == 2
    assert fit.n_censored == 1


def test_fit_constant_hazard_recovers_synthetic_hazard():
    lifetimes = make_exponential_lifetimes(hazard=0.2, n=20_000, seed=6)
    fit = fit_constant_hazard(lifetimes)
    assert fit.hazard == pytest.approx(0.2, rel=0.03)


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("nan")]])
def test_fit_constant_hazard_rejects_invalid_lifetimes(values):
    with pytest.raises(ValueError):
        fit_constant_hazard(values)


def test_fit_constant_hazard_with_censoring_rejects_invalid_observed():
    with pytest.raises(ValueError):
        fit_constant_hazard_with_censoring([1.0], [False])
    with pytest.raises(ValueError):
        fit_constant_hazard_with_censoring([1.0], [True, False])
    with pytest.raises(ValueError):
        fit_constant_hazard_with_censoring([1.0], [1])
