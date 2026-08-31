"""Compact local-exponential/right-conv-3 hybrid for meta 2."""

from collections.abc import Mapping, Sequence

from dctkit.dec import cochain as C

from automodel.model import CorrectionSpec


def correction(
    rho: C.Cochain,
    flats: Mapping[str, object],
    params: Sequence[float] = (0.0, 0.0, 8.0),
) -> C.Cochain:
    """Return exp(c0*rho) + (delta flat_right exp(c1*rho) *_3 exp(c2*rho))."""

    local_slope, inner_slope, kernel_slope = params
    local = C.exp(C.scalar_mul(rho, local_slope))
    gradient = C.codifferential(
        flats["linear_right_P"](C.exp(C.scalar_mul(rho, inner_slope)))
    )
    kernel = C.exp(C.scalar_mul(rho, kernel_slope))
    response = C.convolution(gradient, kernel, 3)
    return C.add(local, response)


CORRECTION = CorrectionSpec(
    name="local_exponential_downwind_window3",
    expression=(
        "exp(c0*rho) + ((delta flat_right_P exp(c1*rho)) *_3 exp(c2*rho))"
    ),
    function=correction,
    default_params=(0.0, 0.0, 8.0),
    bounds=((-0.5, 0.5), (-10.0, 10.0), (-10.0, 10.0)),
    tree_nodes=18,
)
