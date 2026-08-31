import numpy as np
import jax.numpy as jnp
from dctkit import config
from dctkit.dec import cochain as C
from dctkit.mesh import util

from sr_traffic.fd import diagrams
from sr_traffic.utils.flat import define_flats

config()


IDM_COEFFICIENTS = (0.43936351, 0.93094344, 0.16251414, 0.61353022)


def test_inverse_idm_satisfies_equilibrium_relation():
    spacing = np.asarray([0.45, 0.6, 1.0, 2.0, 10.0])
    velocity = diagrams.inverse_IDM(spacing, *IDM_COEFFICIENTS)

    assert np.all(np.isfinite(velocity))
    np.testing.assert_allclose(
        diagrams.IDM_eq(spacing, velocity, *IDM_COEFFICIENTS), 0.0, atol=2.0e-5
    )


def test_idm_velocity_is_finite_nonnegative_and_nonincreasing_in_density():
    density = np.linspace(1.0e-6, 1.0, 400)
    velocity = np.asarray(diagrams.IDM_v(density, *IDM_COEFFICIENTS))

    assert np.all(np.isfinite(velocity))
    assert np.all(velocity >= 0.0)
    assert np.all(np.diff(velocity) <= 1.0e-7)
    assert velocity[-1] == 0.0


def test_selected_dec_corrections_return_finite_primal_zero_cochains():
    mesh, _ = util.generate_line_mesh(8, L=1.0)
    complex_ = util.build_complex_from_mesh(mesh, space_dim=1)
    complex_.get_hodge_star()
    complex_.get_primal_edge_vectors()
    complex_.get_dual_edge_vectors()
    zeros_p = C.CochainP0(complex_, jnp.zeros(complex_.num_nodes))
    zeros_d = C.CochainD0(complex_, jnp.zeros(complex_.num_nodes - 1))
    flat_right = define_flats(complex_, zeros_p, zeros_d)["flat_linear_right_P"]
    density = C.CochainP0(complex_, jnp.linspace(0.1, 0.8, complex_.num_nodes))

    corrections = (
        (diagrams.downwind_window3_correction, (0.1, 8.0)),
        (
            diagrams.exponential_downwind_window3_correction,
            (0.6, -10.0, -0.4),
        ),
        (
            diagrams.local_exponential_downwind_window3_correction,
            (0.5, -6.0, -10.0),
        ),
    )
    for correction, coefficients in corrections:
        multiplier = correction(density, flat_right, *coefficients)
        assert multiplier.dim == 0
        assert multiplier.is_primal
        assert multiplier.coeffs.shape == density.coeffs.shape
        assert np.all(np.isfinite(multiplier.coeffs))


def test_multiplicatively_corrected_flux_matches_explicit_product():
    mesh, _ = util.generate_line_mesh(8, L=1.0)
    complex_ = util.build_complex_from_mesh(mesh, space_dim=1)
    complex_.get_hodge_star()
    complex_.get_primal_edge_vectors()
    complex_.get_dual_edge_vectors()
    zeros_p = C.CochainP0(complex_, jnp.zeros(complex_.num_nodes))
    zeros_d = C.CochainD0(complex_, jnp.zeros(complex_.num_nodes - 1))
    flat_right = define_flats(complex_, zeros_p, zeros_d)["flat_linear_right_P"]
    density = C.CochainP0(complex_, jnp.linspace(0.1, 0.5, complex_.num_nodes))
    baseline_coefficients = (0.6, 0.9)
    correction_coefficients = (0.1, 8.0)

    actual = diagrams.multiplicatively_corrected_flux(
        density,
        diagrams.Greenshields_flux,
        baseline_coefficients,
        diagrams.downwind_window3_correction,
        flat_right,
        correction_coefficients,
    )
    baseline = diagrams.Greenshields_flux(density, *baseline_coefficients)
    multiplier = diagrams.downwind_window3_correction(
        density, flat_right, *correction_coefficients
    )
    expected = C.cochain_mul(baseline, multiplier)

    np.testing.assert_allclose(actual.coeffs, expected.coeffs)
