import jax.numpy as jnp
import dctkit.dec.cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from jax import vmap, lax, jacfwd
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
    """Invert the equilibrium IDM spacing relation on ``0 <= v <= v0``.

    The previous unconstrained Newton iteration could step to negative velocity;
    fractional values of ``delta`` then produced NaNs.  On the physical interval
    the equilibrium equation is monotone, so fixed-iteration bisection is both
    robust and compatible with JAX transformations.
    """

    spacing = jnp.maximum(s_target, s0)

    def body_fun(_, bounds):
        lower, upper = bounds
        midpoint = 0.5 * (lower + upper)
        residual = IDM_eq(spacing, midpoint, s0, T, delta, v0)
        lower = jnp.where(residual > 0, midpoint, lower)
        upper = jnp.where(residual > 0, upper, midpoint)
        return lower, upper

    lower, upper = lax.fori_loop(0, 64, body_fun, (0.0, v0))
    velocity = 0.5 * (lower + upper)
    return jnp.where(s_target > s0, velocity, 0.0)


def IDM_v(rho: npt.NDArray, s0: float, T: float, delta: float, v0: float):
    """Evaluate the equilibrium IDM velocity as a function of density.

    ``inverse_IDM`` accepts the net vehicle spacing ``s`` rather than density.
    This wrapper performs the same ``s = 1 / rho - 1`` conversion used by the
    flux function and returns zero once the spacing reaches the minimum gap.
    Clipping density away from zero keeps the conversion finite.
    """

    rho_safe = jnp.maximum(rho, jnp.finfo(jnp.asarray(rho).dtype).eps)
    spacing = 1 / rho_safe - 1
    return inverse_IDM(spacing, s0, T, delta, v0)


def IDM_flux(rho: C.Cochain, s0: float, T: float, delta: float, v0: float):
    rho_coeffs = rho.coeffs.ravel()
    v = IDM_v(rho_coeffs, s0, T, delta, v0)
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


def downwind_window3_correction(
    rho: C.Cochain,
    flat_downwind: Callable,
    inner_slope: float,
    kernel_slope: float,
):
    """Return the selected three-point DEC multiplicative correction.

    The expression is
    ``1 + (delta flat_downwind exp(inner_slope*rho)) *_3 exp(kernel_slope*rho)``.
    It was selected on the first 60% of the I80 prediction interval and uses
    only primitives enabled in ``sr_traffic.yaml``.
    """

    ones = C.Cochain(rho.dim, rho.is_primal, rho.complex, jnp.ones_like(rho.coeffs))
    gradient = C.codifferential(flat_downwind(C.exp(C.scalar_mul(rho, inner_slope))))
    kernel = C.exp(C.scalar_mul(rho, kernel_slope))
    return C.add(ones, C.convolution(gradient, kernel, 3))


def exponential_downwind_window3_correction(
    rho: C.Cochain,
    flat_downwind: Callable,
    amplitude: float,
    inner_slope: float,
    kernel_slope: float,
):
    """Return a positive exponential envelope around the conv-3 response."""

    gradient = C.codifferential(flat_downwind(C.exp(C.scalar_mul(rho, inner_slope))))
    kernel = C.exp(C.scalar_mul(rho, kernel_slope))
    response = C.convolution(gradient, kernel, 3)
    return C.exp(C.scalar_mul(response, amplitude))


def local_exponential_downwind_window3_correction(
    rho: C.Cochain,
    flat_downwind: Callable,
    local_slope: float,
    inner_slope: float,
    kernel_slope: float,
):
    """Return a local exponential plus the selected conv-3 response."""

    local = C.exp(C.scalar_mul(rho, local_slope))
    gradient = C.codifferential(flat_downwind(C.exp(C.scalar_mul(rho, inner_slope))))
    kernel = C.exp(C.scalar_mul(rho, kernel_slope))
    return C.add(local, C.convolution(gradient, kernel, 3))


def multiplicatively_corrected_flux(
    rho: C.Cochain,
    baseline_flux: Callable,
    baseline_coefficients: npt.NDArray,
    correction: Callable,
    flat_downwind: Callable,
    correction_coefficients: npt.NDArray,
):
    """Apply a fitted DEC correction to any basic fundamental diagram."""

    baseline = baseline_flux(rho, *baseline_coefficients)
    multiplier = correction(rho, flat_downwind, *correction_coefficients)
    return C.cochain_mul(baseline, multiplier)


def define_flux_der(S: SimplicialComplex, flux: Callable):
    def flux_wrap(rho_coeffs, *args):
        rho = C.CochainP0(S, rho_coeffs)
        return flux(rho, *args).coeffs.flatten()

    der = jacfwd(flux_wrap)

    def der_auto(rho, *args):
        return C.CochainP0(rho.complex, jnp.diag(der(rho.coeffs.flatten(), *args)))

    return der_auto
