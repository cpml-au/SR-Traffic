import jax.numpy as jnp
import dctkit.dec.cochain as C
import numpy.typing as npt
from jax import vmap, grad, lax, jacfwd
from functools import partial


def Greenshields_flux(rho, v_max, rho_max):
    return C.Cochain(
        rho.dim,
        rho.is_primal,
        rho.complex,
        v_max * rho.coeffs * (1 - rho.coeffs / rho_max),
    )


def Greenberg_flux(rho, v_max, rho_max):
    return C.Cochain(
        rho.dim,
        rho.is_primal,
        rho.complex,
        v_max * rho.coeffs * jnp.log(rho_max / rho.coeffs),
    )


def Underwood_flux(rho, v_max, rho_max):
    return C.Cochain(
        rho.dim,
        rho.is_primal,
        rho.complex,
        v_max * rho.coeffs * jnp.exp(-rho.coeffs / rho_max),
    )


def Weidmann_v(rho, v_max, rho_max, lambda_w):
    return v_max * (1 - jnp.exp(-lambda_w * (1 / rho - 1 / rho_max)))


def Weidmann_flux(rho, v_max, rho_max, lambda_w):
    return C.Cochain(
        rho.dim,
        rho.is_primal,
        rho.complex,
        rho.coeffs * Weidmann_v(rho.coeffs, v_max, rho_max, lambda_w),
    )


def Greenshields_non_local_flux(rho, v_max, rho_max, eta, eta_window):
    conv = C.convolution(rho, eta, eta_window)
    f_0_conv = v_max * rho.coeffs * (1 - conv.coeffs / rho_max)
    return C.Cochain(rho.dim, rho.is_primal, rho.complex, f_0_conv)


def triangular_flux(rho, V_0, l_eff, T):
    rho_critic = 1 / (V_0 * T + l_eff)
    free_traffic_idx = rho.coeffs <= rho_critic
    congested_traffic_idx = (rho.coeffs > rho_critic) * (rho.coeffs <= 1 / l_eff)
    flux_interm = jnp.where(
        congested_traffic_idx,
        1 / T * (1 - rho.coeffs * l_eff),
        jnp.zeros_like(rho.coeffs),
    )
    flux_coeffs = jnp.where(free_traffic_idx, V_0 * rho.coeffs, flux_interm)
    return C.Cochain(rho.dim, rho.is_primal, rho.complex, flux_coeffs)


def Greenshields_v(rho, v_max, rho_max):
    return v_max * (1 - rho / rho_max)


def Underwood_v(rho, V_0, rho_jam):
    return V_0 * jnp.exp(-rho / rho_jam)


def triangular_v(rho, V_0, l_eff, T):
    rho_critic = 1 / (V_0 * T + l_eff)
    free_traffic_idx = rho <= rho_critic
    congested_traffic_idx = (rho > rho_critic) * (rho <= 1 / l_eff)
    flux_interm = jnp.where(
        congested_traffic_idx, 1 / T * (1 / rho - l_eff), jnp.zeros_like(rho)
    )
    v_coeffs = jnp.where(free_traffic_idx, V_0, flux_interm)
    return v_coeffs


def IDM_fn(v, s0, T, delta, v0):
    return (s0 + v * T) / jnp.sqrt(1 - (v / v0) ** delta)


def IDM_eq(s, v, s0, T, delta, v0):
    return 1 - (v / v0) ** delta - ((s0 + v * T) / s) ** 2


@partial(vmap, in_axes=(0, None, None, None, None))
def inverse_IDM(s_target, s0, T, delta, v0):
    def f(v):
        return IDM_eq(s_target, v, s0, T, delta, v0)

    der_f = grad(f)

    def body_fun(val):
        v, _ = val
        f_val = f(v)
        f_prime = der_f(v)
        v_next = v - f_val / f_prime
        err = jnp.abs(f_val)
        return (v_next, err)

    def cond_fun(val):
        _, err = val
        return err > 1e-6

    v0_guess = 0.5 * v0
    init = (v0_guess, jnp.inf)
    v_final, _ = lax.while_loop(cond_fun, body_fun, init)
    return v_final


def IDM_flux(rho, s0, T, delta, v0):
    rho_coeffs = rho.coeffs.ravel()
    s = 1 / rho_coeffs - 1
    v = inverse_IDM(s, s0, T, delta, v0)
    return C.Cochain(rho.dim, rho.is_primal, rho.complex, rho_coeffs * v)


def del_castillo_v(rho, C_jam, V_max, rho_max, theta):
    rho_norm = rho / rho_max
    a = V_max / C_jam
    v = (
        C_jam
        / rho_norm
        * (
            1
            + (a - 1) * rho_norm
            - ((a * rho_norm) ** theta + (1 - rho_norm) ** theta) ** (1 / theta)
        )
    )
    return v


def del_castillo_flux(rho, C_jam, V_max, rho_max, theta):
    v = del_castillo_v(rho.coeffs, C_jam, V_max, rho_max, theta)
    return C.Cochain(rho.dim, rho.is_primal, rho.complex, rho.coeffs * v)


def define_flux_der(S, flux):
    def flux_wrap(rho_coeffs, *args):
        rho = C.CochainP0(S, rho_coeffs)
        return flux(rho, *args).coeffs.flatten()

    der = jacfwd(flux_wrap)

    def der_auto(rho, *args):
        return C.CochainP0(rho.complex, jnp.diag(der(rho.coeffs.flatten(), *args)))

    return der_auto


# def extended_triangular(rho, u_max, rho_max, rho_c):
#     v_e = u_max * (1 - rho.coeffs / rho_max)
#     Q_c = rho_c * u_max * (1 - rho_c / rho_max)
#     w = Q_c / (rho_max - rho_c)


if __name__ == "__main__":
    # FIXME: this should become a test
    from jax import jit
    import time

    opt_idm = [0.95870305, 0.9524395, 0.68423716, 0.41707987]
    flux_idm_args = {
        "s0": opt_idm[0],
        "T": opt_idm[1],
        "delta": opt_idm[2],
        "v0": opt_idm[3],
    }
    s_targets = jnp.linspace(10, 100, 100)
    jit_inv = jit(inverse_IDM, static_argnums=(1, 2, 3, 4))
    tic = time.perf_counter()
    v = jit_inv(s_targets, opt_idm[0], opt_idm[1], opt_idm[2], opt_idm[3])
    s_comp = IDM_fn(v, **flux_idm_args)
    toc = time.perf_counter()
    print(s_comp, s_targets)
    print(s_comp - s_targets)
    print(toc - tic)
