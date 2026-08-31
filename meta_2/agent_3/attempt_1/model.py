"""Strictly positive envelope around the winning right/conv-3 response."""

from collections.abc import Mapping, Sequence

from dctkit.dec import cochain as C

from automodel.model import CorrectionSpec


def correction(
    rho: C.Cochain,
    flats: Mapping[str, object],
    params: Sequence[float] = (0.0, 0.0, 0.0),
) -> C.Cochain:
    """Return exp(a * ((delta flat_right exp(b*rho)) *_3 exp(c*rho)))."""

    amplitude, inner_slope, kernel_slope = params
    gradient = C.codifferential(
        flats["linear_right_P"](C.exp(C.scalar_mul(rho, inner_slope)))
    )
    kernel = C.exp(C.scalar_mul(rho, kernel_slope))
    response = C.convolution(gradient, kernel, 3)
    return C.exp(C.scalar_mul(response, amplitude))


CORRECTION = CorrectionSpec(
    name="exponential_downwind_window3",
    expression=(
        "exp(a * ((delta flat_right_P exp(b*rho)) *_3 exp(c*rho)))"
    ),
    function=correction,
    default_params=(0.0, 0.0, 0.0),
    bounds=((0.0, 2.5), (-10.0, 10.0), (-10.0, 10.0)),
    tree_nodes=16,
)
