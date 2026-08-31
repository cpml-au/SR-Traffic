"""Frozen Automodel corrections for the I80 prediction benchmark.

The registry lives in the installed ``sr_traffic`` package so simulation and
reporting code can share one authoritative set of expressions and coefficients.
Model selection and coefficient fitting remain in the top-level ``automodel``
experiment artifacts.
"""

from dataclasses import dataclass
from typing import Callable

from sr_traffic.fd import diagrams

PAPER_EXPRESSION = "1 + (delta flat_upwind exp(c1*rho)) *_1 exp(c2*rho)"
COMMON_EXPRESSION = "1 + (delta flat_downwind exp(c1*rho)) *_3 exp(c2*rho)"
POSITIVE_EXPRESSION = "exp(a * ((delta flat_downwind exp(b*rho)) *_3 exp(c*rho)))"
LOCAL_EXPRESSION = "exp(c0*rho) + (delta flat_downwind exp(c1*rho)) *_3 exp(c2*rho)"


@dataclass(frozen=True)
class AutomodelDiagram:
    """One calibrated baseline FD and its frozen multiplicative correction."""

    key: str
    name: str
    baseline_flux: Callable
    baseline_coefficients: tuple[float, ...]
    paper_correction_coefficients: tuple[float, float]
    correction: Callable
    correction_coefficients: tuple[float, ...]
    expression: str
    tree_nodes: int
    source: str


I80_PREDICTION_DIAGRAMS = (
    AutomodelDiagram(
        key="greenshields",
        name="Greenshields",
        baseline_flux=diagrams.Greenshields_flux,
        baseline_coefficients=(0.54673127, 0.55995123),
        paper_correction_coefficients=(2.8602150898540906, -5.644270147515965),
        correction=diagrams.local_exponential_downwind_window3_correction,
        correction_coefficients=(0.5, -5.957330755601453, -10.0),
        expression=LOCAL_EXPRESSION,
        tree_nodes=18,
        source="meta_2/agent_3/attempt_2",
    ),
    AutomodelDiagram(
        key="idm",
        name="IDM",
        baseline_flux=diagrams.IDM_flux,
        baseline_coefficients=(0.43936351, 0.93094344, 0.16251414, 0.61353022),
        paper_correction_coefficients=(2.56095629955761, -0.66842648814023597),
        correction=diagrams.exponential_downwind_window3_correction,
        correction_coefficients=(0.6140045959036415, -10.0, -0.4413702101694601),
        expression=POSITIVE_EXPRESSION,
        tree_nodes=16,
        source="meta_2/agent_3/attempt_1",
    ),
    AutomodelDiagram(
        key="weidmann",
        name="Weidmann",
        baseline_flux=diagrams.Weidmann_flux,
        baseline_coefficients=(0.63190729, 0.80612097, 0.24947817),
        paper_correction_coefficients=(2.1246640785666937, -4.186657672933578),
        correction=diagrams.downwind_window3_correction,
        correction_coefficients=(0.201754023475905, 8.151159515206118),
        expression=COMMON_EXPRESSION,
        tree_nodes=15,
        source="meta_1/agent_2/attempt_2",
    ),
    AutomodelDiagram(
        key="triangular",
        name="Triangular",
        baseline_flux=diagrams.triangular_flux,
        baseline_coefficients=(0.37013956, 1.48964708, 6.59672108),
        paper_correction_coefficients=(2.72042969, -3.4958167993226743),
        correction=diagrams.downwind_window3_correction,
        correction_coefficients=(0.13270747982141273, 8.70403766073787),
        expression=COMMON_EXPRESSION,
        tree_nodes=15,
        source="meta_1/agent_2/attempt_2",
    ),
    AutomodelDiagram(
        key="del_castillo",
        name="Del Castillo",
        baseline_flux=diagrams.del_castillo_flux,
        baseline_coefficients=(0.31807369, 0.46732741, 0.61532169, 2.60100492),
        paper_correction_coefficients=(2.1724725554243847, -0.09043064382541566),
        correction=diagrams.downwind_window3_correction,
        correction_coefficients=(0.13009550793468766, 9.519560935812226),
        expression=COMMON_EXPRESSION,
        tree_nodes=15,
        source="meta_1/agent_2/attempt_2",
    ),
)

I80_PREDICTION_BY_KEY = {diagram.key: diagram for diagram in I80_PREDICTION_DIAGRAMS}
