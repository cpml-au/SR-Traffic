"""Downwind/right-flat, three-point DEC correction for automodel attempt 2."""

from collections.abc import Mapping, Sequence

from dctkit.dec import cochain as C

from automodel.model import CorrectionSpec, _ones_like


def downwind_window3_correction(
    rho: C.Cochain,
    flats: Mapping[str, object],
    params: Sequence[float] = (0.0, 0.0),
) -> C.Cochain:
    """Return ``1 + conv_3(delta(flat_right(exp(c1*rho))), exp(c2*rho))``."""

    gradient = C.codifferential(
        flats["linear_right_P"](C.exp(C.scalar_mul(rho, params[0])))
    )
    kernel = C.exp(C.scalar_mul(rho, params[1]))
    return C.add(_ones_like(rho), C.convolution(gradient, kernel, 3))


CORRECTION = CorrectionSpec(
    name="downwind_window3",
    expression=(
        "1 + (delta flat_downwind/right exp(c1*rho)) *_3 exp(c2*rho)"
    ),
    function=downwind_window3_correction,
    # This seed removes rho-dependence from both branches and passes the
    # corrected-velocity feasibility check for every baseline.
    default_params=(0.0, 0.0),
    bounds=((-10.0, 10.0), (-10.0, 10.0)),
    tree_nodes=15,
)
