"""Positive local-density/right-gradient hybrid correction for meta 1."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from dctkit.dec import cochain as C


Correction = Callable[[C.Cochain, Mapping[str, Callable], Sequence[float]], C.Cochain]


@dataclass(frozen=True)
class CorrectionSpec:
    """Structure and coefficient-search metadata consumed by the harness."""

    name: str
    expression: str
    function: Correction
    default_params: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]
    tree_nodes: int


def correction(
    rho: C.Cochain,
    flats: Mapping[str, Callable],
    params: Sequence[float],
) -> C.Cochain:
    """Return exp(c0*rho + c1*(delta flat_right(rho) *_1 exp(c2*rho)))."""

    c0, c1, c2 = params
    local = C.scalar_mul(rho, c0)
    gradient = C.codifferential(flats["linear_right_P"](rho))
    kernel = C.exp(C.scalar_mul(rho, c2))
    nonlocal_response = C.scalar_mul(C.convolution(gradient, kernel, 1), c1)
    return C.exp(C.add(local, nonlocal_response))


CORRECTION = CorrectionSpec(
    name="positive_local_right_gradient",
    expression="exp(c0*rho + c1*((delta flat_right_P rho) *_1 exp(c2*rho)))",
    function=correction,
    default_params=(0.0, 0.0, 0.0),
    bounds=((-1.5, 0.5), (-4.0, 4.0), (-4.0, 4.0)),
    tree_nodes=15,
)
