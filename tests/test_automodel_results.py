import numpy as np

from sr_traffic.fd import automodel_results
from sr_traffic.fd.automodel_registry import I80_PREDICTION_DIAGRAMS
from sr_traffic.fd.automodel_results import compute_errors, parse_args


def test_registry_covers_every_calibrated_i80_prediction_diagram():
    assert [diagram.key for diagram in I80_PREDICTION_DIAGRAMS] == [
        "greenshields",
        "idm",
        "weidmann",
        "triangular",
        "del_castillo",
    ]
    for diagram in I80_PREDICTION_DIAGRAMS:
        assert diagram.baseline_coefficients
        assert diagram.paper_correction_coefficients
        assert diagram.correction_coefficients
        assert np.all(np.isfinite(diagram.baseline_coefficients))
        assert np.all(np.isfinite(diagram.paper_correction_coefficients))
        assert np.all(np.isfinite(diagram.correction_coefficients))
        assert "flat_downwind" in diagram.expression


def test_compute_errors_uses_paper_relative_root_squared_error():
    true = np.asarray([3.0, 4.0])
    predicted = np.asarray([0.0, 4.0])

    rho_error, velocity_error, flow_error = compute_errors(
        true, true, true, predicted, predicted, predicted
    )

    assert rho_error == velocity_error == flow_error
    np.testing.assert_allclose(rho_error, 3.0 / 5.0)


def test_cli_defaults_to_the_frozen_benchmark():
    args = parse_args([])

    assert args.road_name == "I80"
    assert args.task == "prediction"
    assert not args.tables_only


def test_automodel_report_excludes_published_sr_models(monkeypatch):
    monkeypatch.setattr(
        automodel_results.diagrams,
        "define_flux_der",
        lambda _complex, flux: flux,
    )

    models, plot_names, corrected_names = automodel_results.make_model_functions(
        object(), {}
    )

    expected_names = []
    for diagram in I80_PREDICTION_DIAGRAMS:
        expected_names.extend((diagram.name, f"Automodel-{diagram.name}"))
    assert list(models) == expected_names
    assert plot_names == expected_names
    assert corrected_names == expected_names[1::2]
    assert all(not name.startswith("SR-") for name in models)
