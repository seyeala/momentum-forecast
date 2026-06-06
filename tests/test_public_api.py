def test_package_imports_and_public_exports():
    import momentum_forecast

    expected = {
        "MomentumForecast",
        "TradingCost",
        "known_drift_profit_probability",
        "estimated_drift_profit_probability",
        "momentum_z_score",
    }
    assert expected.issubset(set(momentum_forecast.__all__))
