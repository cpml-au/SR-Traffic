"""Linear-gradient simplification of the meta-1 downwind conv-3 winner."""

from collections.abc import Mapping, Sequence

from dctkit.dec import cochain as C

from automodel.model import CorrectionSpec, _ones_like


def linear_downwind_window3_correction(
    rho: C.Cochain,
    flats: Mapping[str, object],
    params: Sequence[float] = (0.0, 0.0),
) -> C.Cochain:
    """Return ``1 + a*conv_3(delta(flat_right(rho)), exp(b*rho))``."""

    amplitude, kernel_slope = params
    gradient = C.codifferential(flats["linear_right_P"](rho))
    kernel = C.exp(C.scalar_mul(rho, kernel_slope))
    nonlocal_gradient = C.convolution(gradient, kernel, 3)
    return C.add(
        _ones_like(rho), C.scalar_mul(nonlocal_gradient, amplitude)
    )


CORRECTION = CorrectionSpec(
    name="linear_downwind_window3",
    expression=(
        "1 + a * ((delta flat_downwind/right rho) *_3 exp(b*rho))"
    ),
    function=linear_downwind_window3_correction,
    # a=0 makes the correction exactly one for any density field.
    default_params=(0.0, 0.0),
    bounds=((-10.0, 10.0), (-10.0, 10.0)),
    tree_nodes=14,
)
