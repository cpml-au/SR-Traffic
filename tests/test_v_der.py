import jax.numpy as jnp
from jax import jacfwd
from sr_traffic.data.data import preprocess_data
from dctkit.dec import cochain as C
from dctkit import config

config()


def v_test(rho):
    rho_coeffs = rho.coeffs
    kernel = 3 * jnp.ones_like(rho_coeffs)
    kernel_coch = C.Cochain(rho.dim, rho.is_primal, rho.complex, kernel)
    v_rho_coeffs = (1 - rho_coeffs) * C.convolution(rho, kernel_coch, 3).coeffs
    return C.Cochain(rho.dim, rho.is_primal, rho.complex, v_rho_coeffs)


def test_v_der():
    data_info = preprocess_data("US80")
    S = data_info["S"]

    delta_x = (S.node_coords[1] - S.node_coords[0]).item()

    def v_wrap_P0(rho_coeffs):
        rho = C.CochainP0(S, rho_coeffs)
        return v_test(rho).coeffs.flatten()

    def v_wrap_D0(rho_coeffs):
        rho = C.CochainD0(S, rho_coeffs)
        return v_test(rho).coeffs.flatten()

    v_jac_P0 = jacfwd(v_wrap_P0)
    v_jac_D0 = jacfwd(v_wrap_D0)

    def v_der_P0_array(x):
        return jnp.diag(v_jac_P0(x.flatten()))

    def v_der_D0_array(x):
        return jnp.diag(v_jac_D0(x.flatten()))

    rho_coeffs_P0 = data_info["rhoP0"][:, 0]
    rhoP0 = C.CochainP0(S, rho_coeffs_P0)
    kernel_P0 = 3 * jnp.ones_like(rho_coeffs_P0)
    kernel_P0_coch = C.CochainP0(S, kernel_P0)
    rho_conv_kernel_P0 = C.convolution(rhoP0, kernel_P0_coch, 3).coeffs.flatten()

    v_der_P0 = v_der_P0_array(rho_coeffs_P0.reshape(-1, 1))
    v_der_P0_true = -rho_conv_kernel_P0 + (1 - rho_coeffs_P0) * 3 * delta_x

    rho_coeffs_D0 = data_info["density"][:, 0]
    rhoD0 = C.CochainD0(S, rho_coeffs_D0)
    kernel_D0 = 3 * jnp.ones_like(rho_coeffs_D0)
    kernel_D0_coch = C.CochainD0(S, kernel_D0)
    rho_conv_kernel_D0 = C.convolution(rhoD0, kernel_D0_coch, 3).coeffs.flatten()

    v_der_D0 = v_der_D0_array(rho_coeffs_D0.reshape(-1, 1))
    v_der_D0_true = -rho_conv_kernel_D0 + (1 - rho_coeffs_D0) * 3 * delta_x

    assert jnp.allclose(v_der_P0[1:-2], v_der_P0_true[1:-2], atol=1e-15)
    assert jnp.allclose(v_der_D0[1:-2], v_der_D0_true[1:-2], atol=1e-15)
