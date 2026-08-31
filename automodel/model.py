"""Multiplicative correction models used by the automodel search.

Every correction has the signature ``correction(rho, flats, params)`` so that
model structure and coefficient optimization remain separate.  The returned
primal 0-cochain multiplies both the baseline flux and velocity.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from dctkit.dec import cochain as C

Correction = Callable[[C.Cochain, Mapping[str, Callable], Sequence[float]], C.Cochain]


@dataclass(frozen=True)
class CorrectionSpec:
    """Metadata needed to fit and compare one correction structure."""

    name: str
    expression: str
    function: Correction
    default_params: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]
    tree_nodes: int


def _ones_like(rho: C.Cochain) -> C.Cochain:
    return C.Cochain(rho.dim, rho.is_primal, rho.complex, jnp.ones_like(rho.coeffs))


def identity_correction(
    rho: C.Cochain,
    flats: Mapping[str, Callable],
    params: Sequence[float] = (),
) -> C.Cochain:
    """Return the unmodified fundamental diagram, ``g[rho] = 1``."""

    del flats, params
    return _ones_like(rho)


def constant_correction(
    rho: C.Cochain,
    flats: Mapping[str, Callable],
    params: Sequence[float] = (1.0,),
) -> C.Cochain:
    """Return the simplest fitted ansatz, ``g[rho] = c0``."""

    del flats
    return C.scalar_mul(_ones_like(rho), params[0])


def paper_prediction_correction(
    rho: C.Cochain,
    flats: Mapping[str, Callable],
    params: Sequence[float] = (2.72042969, -3.4958167993226743),
) -> C.Cochain:
    """Prediction correction from Eq. (4) of the SR-Traffic paper.

    ``g[rho] = 1 + (delta flat_upwind exp(c1 rho)) *_1 exp(c2 rho)``.
    In this repository ``flat_linear_left_P`` is the upwind primal flat.
    """

    gradient = C.codifferential(
        flats["linear_left_P"](C.exp(C.scalar_mul(rho, params[0])))
    )
    kernel = C.exp(C.scalar_mul(rho, params[1]))
    return C.add(_ones_like(rho), C.convolution(gradient, kernel, 1))


CORRECTIONS = {
    "identity": CorrectionSpec(
        name="identity",
        expression="1",
        function=identity_correction,
        default_params=(),
        bounds=(),
        tree_nodes=1,
    ),
    "constant": CorrectionSpec(
        name="constant",
        expression="c0",
        function=constant_correction,
        default_params=(1.0,),
        bounds=((0.5, 1.5),),
        tree_nodes=1,
    ),
    "paper_prediction": CorrectionSpec(
        name="paper_prediction",
        expression="1 + (delta flat_upwind exp(c1*rho)) *_1 exp(c2*rho)",
        function=paper_prediction_correction,
        default_params=paper_prediction_correction.__defaults__[-1],
        bounds=((-10.0, 10.0), (-10.0, 10.0)),
        tree_nodes=15,
    ),
}
