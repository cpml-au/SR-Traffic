"""Reduced two-parameter linear-gradient downwind conv-3 correction."""

from collections.abc import Mapping, Sequence

from dctkit.dec import cochain as C

from automodel.model import CorrectionSpec, _ones_like


def amplitude_linear_downwind_window3_correction(
    rho: C.Cochain,
    flats: Mapping[str, object],
    params: Sequence[float] = (0.0, 0.0),
) -> C.Cochain:
    """Return ``1 + a * conv_3(delta(flat_right(rho)), exp(c*rho))``."""

    amplitude, kernel_slope = params
    gradient = C.codifferential(flats["linear_right_P"](rho))
    kernel = C.exp(C.scalar_mul(rho, kernel_slope))
    response = C.convolution(gradient, kernel, 3)
    return C.add(_ones_like(rho), C.scalar_mul(response, amplitude))


CORRECTION = CorrectionSpec(
    name="amplitude_linear_downwind_window3",
    expression="1 + a * ((delta flat_downwind/right rho) *_3 exp(c*rho))",
    function=amplitude_linear_downwind_window3_correction,
    default_params=(0.0, 0.0),
    bounds=((-5.0, 5.0), (-15.0, 15.0)),
    tree_nodes=14,
)
