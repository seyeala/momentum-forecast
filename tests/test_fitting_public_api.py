def test_fitting_imports_and_public_exports():
    import momentum_forecast.fitting as fitting

    expected = {
        "DriftFit",
        "VolatilityFit",
        "AR1Fit",
        "ChannelFit",
        "HazardFit",
        "fit_mean_drift",
        "fit_return_variance",
        "fit_ar1_phi",
        "fit_line_residuals",
        "fit_constant_hazard",
        "rolling_mean_drift",
    }
    assert expected.issubset(set(fitting.__all__))


def test_phase1_models_do_not_import_fitting_modules():
    import momentum_forecast.models.ar1 as ar1
    import momentum_forecast.models.channel_life as channel_life
    import momentum_forecast.models.estimated_drift as estimated_drift
    import momentum_forecast.models.gaussian_drift as gaussian_drift
    import momentum_forecast.models.hazard as hazard

    for module in [ar1, channel_life, estimated_drift, gaussian_drift, hazard]:
        assert "momentum_forecast.fitting" not in repr(module.__dict__)
