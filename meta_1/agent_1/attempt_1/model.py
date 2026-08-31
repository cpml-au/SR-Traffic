"""Amplitude-scaled version of the published upwind-gradient correction."""

from collections.abc import Callable, Mapping, Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C

from automodel.model import CorrectionSpec


def _ones_like(rho: C.Cochain) -> C.Cochain:
    """Construct the dimensionless identity correction on ``rho``'s complex."""

    return C.Cochain(
        rho.dim, rho.is_primal, rho.complex, jnp.ones_like(rho.coeffs)
    )


def amplitude_upwind_correction(
    rho: C.Cochain,
    flats: Mapping[str, Callable],
    params: Sequence[float] = (1.0, 2.5, -3.5),
) -> C.Cochain:
    """Return ``1 + a * (delta flat_left exp(b*rho) *_1 exp(c*rho))``."""

    amplitude, inner_slope, envelope_slope = params
    gradient = C.codifferential(
        flats["linear_left_P"](C.exp(C.scalar_mul(rho, inner_slope)))
    )
    envelope = C.exp(C.scalar_mul(rho, envelope_slope))
    nonlocal_term = C.convolution(gradient, envelope, 1)
    return C.add(
        _ones_like(rho), C.scalar_mul(nonlocal_term, amplitude)
    )


CORRECTION = CorrectionSpec(
    name="amplitude_upwind",
    expression=(
        "1 + a * ((delta flat_left exp(b*rho)) *_1 exp(c*rho))"
    ),
    function=amplitude_upwind_correction,
    default_params=(1.0, 2.5, -3.5),
    bounds=((0.0, 2.5), (0.0, 6.0), (-8.0, 0.0)),
    tree_nodes=17,
)
