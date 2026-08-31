"""Selected I80/prediction multiplicative corrections for every basic FD."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dctkit.dec import cochain as C

from automodel.model import CorrectionSpec
from sr_traffic.fd import diagrams
from sr_traffic.fd.automodel_registry import (
    COMMON_EXPRESSION,
    I80_PREDICTION_BY_KEY,
    LOCAL_EXPRESSION,
    POSITIVE_EXPRESSION,
)


def common_downwind_window3(
    rho: C.Cochain,
    flats: Mapping[str, object],
    params: Sequence[float],
) -> C.Cochain:
    return diagrams.downwind_window3_correction(rho, flats["linear_right_P"], *params)


def positive_downwind_window3(
    rho: C.Cochain,
    flats: Mapping[str, object],
    params: Sequence[float],
) -> C.Cochain:
    return diagrams.exponential_downwind_window3_correction(
        rho, flats["linear_right_P"], *params
    )


def local_downwind_window3(
    rho: C.Cochain,
    flats: Mapping[str, object],
    params: Sequence[float],
) -> C.Cochain:
    return diagrams.local_exponential_downwind_window3_correction(
        rho, flats["linear_right_P"], *params
    )


COMMON = CorrectionSpec(
    name="downwind_window3",
    expression=COMMON_EXPRESSION,
    function=common_downwind_window3,
    default_params=(0.0, 0.0),
    bounds=((-10.0, 10.0), (-10.0, 10.0)),
    tree_nodes=15,
)

POSITIVE = CorrectionSpec(
    name="exponential_downwind_window3",
    expression=POSITIVE_EXPRESSION,
    function=positive_downwind_window3,
    default_params=(0.0, 0.0, 0.0),
    bounds=((0.0, 2.5), (-10.0, 10.0), (-10.0, 10.0)),
    tree_nodes=16,
)

LOCAL = CorrectionSpec(
    name="local_exponential_downwind_window3",
    expression=LOCAL_EXPRESSION,
    function=local_downwind_window3,
    default_params=(0.0, 0.0, 8.0),
    bounds=((-0.5, 0.5), (-10.0, 10.0), (-10.0, 10.0)),
    tree_nodes=18,
)


@dataclass(frozen=True)
class SelectedCorrection:
    correction: CorrectionSpec
    params: tuple[float, ...]
    source: str


SELECTED_CORRECTIONS = {
    "greenshields": SelectedCorrection(
        LOCAL,
        I80_PREDICTION_BY_KEY["greenshields"].correction_coefficients,
        I80_PREDICTION_BY_KEY["greenshields"].source,
    ),
    "weidmann": SelectedCorrection(
        COMMON,
        I80_PREDICTION_BY_KEY["weidmann"].correction_coefficients,
        I80_PREDICTION_BY_KEY["weidmann"].source,
    ),
    "triangular": SelectedCorrection(
        COMMON,
        I80_PREDICTION_BY_KEY["triangular"].correction_coefficients,
        I80_PREDICTION_BY_KEY["triangular"].source,
    ),
    "idm": SelectedCorrection(
        POSITIVE,
        I80_PREDICTION_BY_KEY["idm"].correction_coefficients,
        I80_PREDICTION_BY_KEY["idm"].source,
    ),
    "del_castillo": SelectedCorrection(
        COMMON,
        I80_PREDICTION_BY_KEY["del_castillo"].correction_coefficients,
        I80_PREDICTION_BY_KEY["del_castillo"].source,
    ),
}
