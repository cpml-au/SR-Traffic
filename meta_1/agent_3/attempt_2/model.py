"""Positive local-density/squared-gradient hybrid correction for meta 1."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

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
    """Return exp(c0*rho + c1*square(delta flat_right(rho)))."""

    c0, c1 = params
    local = C.scalar_mul(rho, c0)
    gradient = C.codifferential(flats["linear_right_P"](rho))
    gradient_response = C.scalar_mul(C.square(gradient), c1)
    return C.exp(C.add(local, gradient_response))


CORRECTION = CorrectionSpec(
    name="positive_local_squared_right_gradient",
    expression="exp(c0*rho + c1*square(delta flat_right_P rho))",
    function=correction,
    default_params=(0.0, 0.0),
    bounds=((-0.5, 0.5), (-4.0, 4.0)),
    tree_nodes=11,
)
