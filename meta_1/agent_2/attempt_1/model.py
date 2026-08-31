"""Downwind/right-flat, one-point DEC correction for automodel attempt 1."""

from collections.abc import Mapping, Sequence

from dctkit.dec import cochain as C

from automodel.model import CorrectionSpec, _ones_like


def downwind_window1_correction(
    rho: C.Cochain,
    flats: Mapping[str, object],
    params: Sequence[float] = (2.7, -3.5),
) -> C.Cochain:
    """Return ``1 + conv_1(delta(flat_right(exp(c1*rho))), exp(c2*rho))``."""

    gradient = C.codifferential(
        flats["linear_right_P"](C.exp(C.scalar_mul(rho, params[0])))
    )
    kernel = C.exp(C.scalar_mul(rho, params[1]))
    return C.add(_ones_like(rho), C.convolution(gradient, kernel, 1))


CORRECTION = CorrectionSpec(
    name="downwind_window1",
    expression=(
        "1 + (delta flat_downwind/right exp(c1*rho)) *_1 exp(c2*rho)"
    ),
    function=downwind_window1_correction,
    default_params=(2.7, -3.5),
    bounds=((-10.0, 10.0), (-10.0, 10.0)),
    tree_nodes=15,
)
