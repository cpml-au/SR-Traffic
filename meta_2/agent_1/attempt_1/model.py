"""Amplitude-decoupled refinement of the downwind three-point winner."""

from collections.abc import Mapping, Sequence

from dctkit.dec import cochain as C

from automodel.model import CorrectionSpec, _ones_like


def amplitude_downwind_window3_correction(
    rho: C.Cochain,
    flats: Mapping[str, object],
    params: Sequence[float] = (0.0, 0.0, 0.0),
) -> C.Cochain:
    """Return ``1 + a * conv_3(delta(flat_right(exp(c1*rho))), exp(c2*rho))``."""

    amplitude, inner_slope, kernel_slope = params
    gradient = C.codifferential(
        flats["linear_right_P"](C.exp(C.scalar_mul(rho, inner_slope)))
    )
    kernel = C.exp(C.scalar_mul(rho, kernel_slope))
    response = C.convolution(gradient, kernel, 3)
    return C.add(_ones_like(rho), C.scalar_mul(response, amplitude))


CORRECTION = CorrectionSpec(
    name="amplitude_downwind_window3",
    expression=(
        "1 + a * ((delta flat_downwind/right exp(c1*rho)) *_3 exp(c2*rho))"
    ),
    function=amplitude_downwind_window3_correction,
    default_params=(0.0, 0.0, 0.0),
    bounds=((-2.0, 2.0), (-15.0, 15.0), (-15.0, 15.0)),
    tree_nodes=17,
)
