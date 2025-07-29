from dctkit.dec import cochain as C
import jax.numpy as jnp
from jax import lax, jit
from functools import partial


def compute_flux(S, flux, flux_der, rho_t_coeffs, flats):
    rho_t = C.CochainD0(S, rho_t_coeffs)

    # reconstruct values at cell boundaries
    rho_plus = C.star(flats["linear_right"](rho_t))
    rho_minus = C.star(flats["linear_left"](rho_t))

    flux_plus = flux(rho_plus)
    flux_minus = flux(rho_minus)

    # build Rusanov flux, for reference see e.g. Leveque page 233
    # or Toro page 329
    abs_flux_der_plus = C.abs(flux_der(rho_plus))
    abs_flux_der_minus = C.abs(flux_der(rho_minus))
    max_flux = C.maximum(abs_flux_der_plus, abs_flux_der_minus)

    max_term = C.scalar_mul(C.cochain_mul(max_flux, C.sub(rho_minus, rho_plus)), 0.5)

    flux = C.sub(C.scalar_mul(C.add(flux_plus, flux_minus), 0.5), max_term)
    return flux


def update_rho(S, flux, flux_der, delta_t, k, rho_t_coeffs, flats):
    rho_t = C.CochainD0(S, rho_t_coeffs)
    flux = compute_flux(S, flux, flux_der, rho_t_coeffs, flats)
    rho_tp1_no_diff = C.sub(rho_t, C.scalar_mul(C.star(C.coboundary(flux)), delta_t))
    rho_tp1_pred = C.sub(rho_tp1_no_diff, C.scalar_mul(C.laplacian(rho_t), k * delta_t))
    return rho_tp1_pred.coeffs


def body_fun(S, rho, flux, flux_der, delta_t, k, flats, rho_t_check, t: float):
    rho_t, check = rho_t_check
    rho_tp1 = update_rho(S, flux, flux_der, delta_t, k, rho_t, flats)
    rho_tp1 = rho_tp1.at[0].set(rho[0, t + 1])
    rho_tp1 = rho_tp1.at[-3:].set(rho[-3:, t + 1].reshape(-1, 1))

    def compute_step(rho_t):
        rho_tp1 = update_rho(S, flux, flux_der, delta_t, k, rho_t, flats)
        rho_tp1 = rho_tp1.at[0].set(rho[0, t + 1])
        rho_tp1 = rho_tp1.at[-3:].set(rho[-3:, t + 1].reshape(-1, 1))
        nan_check = jnp.sum(jnp.abs(rho_tp1) > 1) > 0
        return rho_tp1, nan_check

    def skip_step(val):
        return (val, True)

    rho_tp1_check = lax.cond(
        check, lambda _: skip_step(rho_t), lambda _: compute_step(rho_t), operand=None
    )

    # rho_tp1_check = (rho_tp1, check)

    # to compute velocity, first interpolate rho_tp1
    rho_tp1_P0 = C.star(flats["linear_left"](C.CochainD0(S, rho_tp1_check[0])))
    flux_tp1 = flux(rho_tp1_P0).coeffs
    v_tp1 = flux_tp1 / rho_tp1_P0.coeffs

    return rho_tp1_check, (rho_tp1_check[0], v_tp1[:-1], flux_tp1[:-1])


def main_loop(rho_0, single_iteration, num_t_points):
    init = (rho_0.reshape(-1, 1), False)
    _, rho_v_f = lax.scan(single_iteration, init, jnp.arange(num_t_points - 1))
    return rho_v_f


def godunov_solver(
    rho_0, S, rho_bnd_array, flux, flux_der, delta_t, k, flats, num_t_points
):
    single_iteration = partial(
        body_fun, S, rho_bnd_array, flux, flux_der, delta_t, k, flats
    )
    jitted_iteration = jit(single_iteration)
    rho_v_f = main_loop(rho_0, jitted_iteration, num_t_points)
    rho_1_T = rho_v_f[0][:, :, 0]
    v_1_T = rho_v_f[1][:, :, 0]
    f_1_T = rho_v_f[2][:, :, 0]

    # first interpolate rho_0, then compute velocity
    rho_0_P0 = C.star(flats["linear_left"](C.CochainD0(S, rho_0)))
    f_0 = flux(rho_0_P0).coeffs
    v_0 = f_0 / rho_0_P0.coeffs

    # insert initial values of v and f
    rho_computed = jnp.vstack([rho_0, rho_1_T]).T
    v_computed = jnp.vstack([v_0[:-1].flatten(), v_1_T]).T
    f_computed = jnp.vstack([f_0[:-1].flatten(), f_1_T]).T
    return rho_computed, v_computed, f_computed
