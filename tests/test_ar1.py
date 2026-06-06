import pytest

from momentum_forecast.models import ar1_expected_cumulative_return, ar1_expected_return_at_horizon, ar1_half_life


def test_ar1_expected_return_at_horizon():
    assert ar1_expected_return_at_horizon(0.01, 0.5, 3) == pytest.approx(0.00125)


def test_ar1_expected_cumulative_return_phi_zero():
    assert ar1_expected_cumulative_return(0.01, 0.0, 5) == pytest.approx(0.0)


def test_ar1_expected_cumulative_return_between_zero_and_one():
    assert ar1_expected_cumulative_return(0.01, 0.5, 3) == pytest.approx(0.00875)


def test_ar1_expected_cumulative_return_phi_one():
    assert ar1_expected_cumulative_return(0.01, 1.0, 5) == pytest.approx(0.05)


def test_ar1_expected_cumulative_return_phi_greater_than_one():
    assert ar1_expected_cumulative_return(0.01, 1.1, 2) == pytest.approx(0.0231)


def test_ar1_half_life():
    assert ar1_half_life(0.5) == pytest.approx(1.0)
    assert ar1_half_life(0.0) is None
    assert ar1_half_life(1.0) is None
    assert ar1_half_life(1.1) is None


@pytest.mark.parametrize(
    ("func", "args"),
    [
        (ar1_expected_return_at_horizon, (0.01, 0.5, 0)),
        (ar1_expected_cumulative_return, (0.01, 0.5, 0)),
    ],
)
def test_ar1_rejects_non_positive_horizon(func, args):
    with pytest.raises(ValueError):
        func(*args)
