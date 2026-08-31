"""Positive-envelope variant of the published upwind-gradient correction."""

from collections.abc import Callable, Mapping, Sequence

from dctkit.dec import cochain as C

from automodel.model import CorrectionSpec


def exponential_upwind_envelope(
    rho: C.Cochain,
    flats: Mapping[str, Callable],
    params: Sequence[float] = (1.0, -2.5, -3.5),
) -> C.Cochain:
    """Return ``exp(a * (delta flat_left exp(b*rho) *_1 exp(c*rho)))``."""

    amplitude, inner_slope, envelope_slope = params
    gradient = C.codifferential(
        flats["linear_left_P"](C.exp(C.scalar_mul(rho, inner_slope)))
    )
    envelope = C.exp(C.scalar_mul(rho, envelope_slope))
    nonlocal_term = C.convolution(gradient, envelope, 1)
    return C.exp(C.scalar_mul(nonlocal_term, amplitude))


CORRECTION = CorrectionSpec(
    name="exponential_upwind_envelope",
    expression=(
        "exp(a * ((delta flat_left exp(b*rho)) *_1 exp(c*rho)))"
    ),
    function=exponential_upwind_envelope,
    default_params=(1.0, -2.5, -3.5),
    bounds=((0.0, 2.5), (-8.0, 0.0), (-8.0, 0.0)),
    tree_nodes=16,
)
