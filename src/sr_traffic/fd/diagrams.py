import jax.numpy as jnp
import dctkit.dec.cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from jax import vmap, grad, lax, jacfwd
from functools import partial
import numpy.typing as npt
from typing import Callable


def Greenshields_flux(rho: C.Cochain, v_max: float, rho_max: float):
    return C.Cochain(
        rho.dim,
        rho.is_primal,
        rho.complex,
        v_max * rho.coeffs * (1 - rho.coeffs / rho_max),
    )


def Greenberg_flux(rho: C.Cochain, v_max: float, rho_max: float):
    return C.Cochain(
        rho.dim,
        rho.is_primal,
        rho.complex,
        v_max * rho.coeffs * jnp.log(rho_max / rho.coeffs),
    )


def Underwood_flux(rho: C.Cochain, v_max: float, rho_max: float):
    return C.Cochain(
        rho.dim,
        rho.is_primal,
        rho.complex,
        v_max * rho.coeffs * jnp.exp(-rho.coeffs / rho_max),
    )


def Weidmann_flux(rho: C.Cochain, v_max: float, rho_max: float, lambda_w: float):
    return C.Cochain(
        rho.dim,
        rho.is_primal,
        rho.complex,
        rho.coeffs * Weidmann_v(rho.coeffs, v_max, rho_max, lambda_w),
    )


def triangular_flux(rho: C.Cochain, V_0: float, l_eff: float, T: float):
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


def Greenshields_v(rho: npt.NDArray, v_max: float, rho_max: float):
    return v_max * (1 - rho / rho_max)


def Underwood_v(rho: npt.NDArray, v_max: float, rho_max: float):
    return v_max * jnp.exp(-rho / rho_max)


def Weidmann_v(rho: npt.NDArray, v_max: float, rho_max: float, lambda_w: float):
    return v_max * (1 - jnp.exp(-lambda_w * (1 / rho - 1 / rho_max)))


def triangular_v(rho: npt.NDArray, V_0: float, l_eff: float, T: float):
    rho_critic = 1 / (V_0 * T + l_eff)
    free_traffic_idx = rho <= rho_critic
    congested_traffic_idx = (rho > rho_critic) * (rho <= 1 / l_eff)
    flux_interm = jnp.where(
        congested_traffic_idx, 1 / T * (1 / rho - l_eff), jnp.zeros_like(rho)
    )
    v_coeffs = jnp.where(free_traffic_idx, V_0, flux_interm)
    return v_coeffs


def IDM_fn(v: npt.NDArray, s0: float, T: float, delta: float, v0: float):
    return (s0 + v * T) / jnp.sqrt(1 - (v / v0) ** delta)


def IDM_eq(
    s: npt.NDArray, v: npt.NDArray, s0: float, T: float, delta: float, v0: float
):
    return 1 - (v / v0) ** delta - ((s0 + v * T) / s) ** 2


@partial(vmap, in_axes=(0, None, None, None, None))
def inverse_IDM(s_target: npt.NDArray, s0: float, T: float, delta: float, v0: float):
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


def IDM_flux(rho: C.Cochain, s0: float, T: float, delta: float, v0: float):
    rho_coeffs = rho.coeffs.ravel()
    s = 1 / rho_coeffs - 1
    v = inverse_IDM(s, s0, T, delta, v0)
    return C.Cochain(rho.dim, rho.is_primal, rho.complex, rho_coeffs * v)


def del_castillo_v(
    rho: npt.NDArray, C_jam: float, V_max: float, rho_max: float, theta: float
):
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


def del_castillo_flux(
    rho: C.Cochain, C_jam: float, V_max: float, rho_max: float, theta: float
):
    v = del_castillo_v(rho.coeffs, C_jam, V_max, rho_max, theta)
    return C.Cochain(rho.dim, rho.is_primal, rho.complex, rho.coeffs * v)


def define_flux_der(S: SimplicialComplex, flux: Callable):
    def flux_wrap(rho_coeffs, *args):
        rho = C.CochainP0(S, rho_coeffs)
        return flux(rho, *args).coeffs.flatten()

    der = jacfwd(flux_wrap)

    def der_auto(rho, *args):
        return C.CochainP0(rho.complex, jnp.diag(der(rho.coeffs.flatten(), *args)))

    return der_auto
