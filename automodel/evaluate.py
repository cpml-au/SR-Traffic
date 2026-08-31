"""Fit and evaluate multiplicative traffic-FD corrections on I80/prediction."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from dctkit import config
from dctkit.dec import cochain as C
from scipy.optimize import differential_evolution, minimize

import sr_traffic.utils.flat as tf_flat
from sr_traffic.data.data import build_dataset, preprocess_data
from sr_traffic.fd import diagrams
from sr_traffic.sr.utils import is_v_unfeasible, solve

from automodel.model import CORRECTIONS, CorrectionSpec

config()


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    flux: Callable
    velocity: Callable
    coefficients: tuple[float, ...]


@dataclass
class Metrics:
    data_error: float
    rho_rrmse: float
    velocity_rrmse: float
    flow_rrmse: float


BASELINES = {
    "greenshields": BaselineSpec(
        "Greenshields",
        diagrams.Greenshields_flux,
        diagrams.Greenshields_v,
        (0.54673127, 0.55995123),
    ),
    "weidmann": BaselineSpec(
        "Weidmann",
        diagrams.Weidmann_flux,
        diagrams.Weidmann_v,
        (0.63190729, 0.80612097, 0.24947817),
    ),
    "triangular": BaselineSpec(
        "Triangular",
        diagrams.triangular_flux,
        diagrams.triangular_v,
        (0.37013956, 1.48964708, 6.59672108),
    ),
    "idm": BaselineSpec(
        "IDM",
        diagrams.IDM_flux,
        diagrams.IDM_v,
        (0.43936351, 0.93094344, 0.16251414, 0.61353022),
    ),
    "del_castillo": BaselineSpec(
        "Del Castillo",
        diagrams.del_castillo_flux,
        diagrams.del_castillo_v,
        (0.31807369, 0.46732741, 0.61532169, 2.60100492),
    ),
}


PAPER_PARAMS = {
    "greenshields": (2.8602150898540906, -5.644270147515965),
    "weidmann": (2.1246640785666937, -4.186657672933578),
    "triangular": (2.72042969, -3.4958167993226743),
    "idm": (2.56095629955761, -0.66842648814023597),
    "del_castillo": (2.1724725554243847, -0.09043064382541566),
}


class I80PredictionProblem:
    """Shared data, DEC operators, simulator, and metrics for one baseline FD."""

    def __init__(self, baseline: str):
        self.baseline_key = baseline
        self.baseline = BASELINES[baseline]
        self.data = preprocess_data("I80")
        _, _, self.train_data, self.test_data = build_dataset(
            self.data["t_sampled_circ"],
            self.data["S"],
            self.data["density"],
            self.data["v"],
            self.data["flow"],
            "prediction",
        )
        self.S = self.data["S"]
        zeros_p = C.CochainP0(self.S, jnp.zeros_like(self.data["vP0"][:, 0]))
        zeros_d = C.CochainD0(self.S, jnp.zeros_like(self.data["density"][:, 0]))
        all_flats = tf_flat.define_flats(self.S, zeros_p, zeros_d)
        self.flats = {
            "linear_left": all_flats["flat_linear_left_D"],
            "linear_right": all_flats["flat_linear_right_D"],
            "linear_left_P": all_flats["flat_linear_left_P"],
            "linear_right_P": all_flats["flat_linear_right_P"],
        }
        self.ansatz = {
            "flux": self.baseline.flux,
            "v": self.baseline.velocity,
            "opt_coeffs": self.baseline.coefficients,
        }
        self.train_num_t_points = int(self.train_data[-1, 0] * self.data["step"] + 1)
        self.rho_feasibility_grid = jnp.linspace(1.0e-6, 1.0, 400)

    def correction_callable(self, correction: CorrectionSpec):
        def function(rho, params):
            return correction.function(rho, self.flats, params)

        return function

    def objective(self, correction: CorrectionSpec):
        function = self.correction_callable(correction)

        @jax.jit
        def objective_fn(params):
            def individual(rho):
                return function(rho, params)

            invalid_velocity = is_v_unfeasible(
                individual,
                self.rho_feasibility_grid,
                self.S,
                self.baseline.velocity,
                self.baseline.coefficients,
            )
            error, _ = solve(
                function,
                params,
                self.train_data,
                self.data["rho_bnd"],
                self.data["rho_0"],
                self.S,
                self.train_num_t_points,
                self.data["delta_t_refined"],
                self.data["step"],
                self.flats,
                self.ansatz,
                "prediction",
            )
            valid_error = jnp.nan_to_num(error, nan=100.0, posinf=100.0, neginf=100.0)
            return jnp.where(invalid_velocity, 100.0, jnp.minimum(valid_error, 100.0))

        return objective_fn

    def is_feasible(self, correction: CorrectionSpec, params: Sequence[float]) -> bool:
        """Check finiteness and non-increasing corrected velocity on the rho grid."""

        params_array = jnp.asarray(params)
        function = self.correction_callable(correction)

        def individual(rho):
            return function(rho, params_array)

        unfeasible = is_v_unfeasible(
            individual,
            self.rho_feasibility_grid,
            self.S,
            self.baseline.velocity,
            self.baseline.coefficients,
        )
        return not bool(jax.device_get(unfeasible))

    def evaluate(
        self, correction: CorrectionSpec, params: Sequence[float], split: str
    ) -> Metrics:
        function = self.correction_callable(correction)
        data = self.train_data if split == "train" else self.test_data
        num_t_points = (
            self.train_num_t_points if split == "train" else self.data["num_t_points"]
        )
        error, fields = solve(
            function,
            jnp.asarray(params),
            data,
            self.data["rho_bnd"],
            self.data["rho_0"],
            self.S,
            num_t_points,
            self.data["delta_t_refined"],
            self.data["step"],
            self.flats,
            self.ansatz,
            "prediction",
        )
        indices = np.arange(data[0, 0], data[-1, 0] + 1, dtype=np.int64)
        sampled = indices * self.data["step"]
        predictions = {
            "rho": fields["rho"][1:-3, sampled].ravel("F"),
            "v": fields["v"][1:-3, sampled].ravel("F"),
            "f": fields["f"][1:-3, sampled].ravel("F"),
        }

        def rrmse(predicted, observed):
            return float(
                jnp.sqrt(jnp.sum((predicted - observed) ** 2))
                / jnp.sqrt(jnp.sum(observed**2))
            )

        return Metrics(
            data_error=float(error),
            rho_rrmse=rrmse(predictions["rho"], data[:, 1]),
            velocity_rrmse=rrmse(predictions["v"], data[:, 2]),
            flow_rrmse=rrmse(predictions["f"], data[:, 3]),
        )

    def fit(
        self,
        correction: CorrectionSpec,
        seed: int = 0,
        maxiter: int = 20,
        popsize: int = 8,
        polish_maxiter: int = 50,
    ) -> tuple[np.ndarray, float, bool, str]:
        if not correction.bounds:
            params = np.asarray(correction.default_params, dtype=float)
            return (
                params,
                float(self.objective(correction)(params)),
                True,
                "no parameters",
            )

        objective = self.objective(correction)
        # Compile once so compilation time is not multiplied by optimizer calls.
        objective(jnp.asarray(correction.default_params)).block_until_ready()

        def scipy_objective(params):
            return float(objective(jnp.asarray(params)))

        result = differential_evolution(
            scipy_objective,
            correction.bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            polish=False,
            workers=1,
            updating="immediate",
            x0=np.asarray(correction.default_params, dtype=float),
        )
        if polish_maxiter <= 0:
            return result.x, float(result.fun), bool(result.success), result.message
        polished = minimize(
            scipy_objective,
            result.x,
            method="Nelder-Mead",
            bounds=correction.bounds,
            options={"maxiter": polish_maxiter, "xatol": 1.0e-5},
        )
        if polished.fun < result.fun:
            return (
                polished.x,
                float(polished.fun),
                bool(polished.success),
                polished.message,
            )
        return result.x, float(result.fun), bool(result.success), result.message


def run(
    baseline_key: str,
    correction_key: str,
    params: Sequence[float] | None,
    optimize: bool,
    seed: int,
    maxiter: int,
    popsize: int,
    include_test: bool,
):
    started = time.perf_counter()
    problem = I80PredictionProblem(baseline_key)
    correction = CORRECTIONS[correction_key]
    converged = True
    message = "parameters supplied"
    if optimize:
        fitted, objective, converged, message = problem.fit(
            correction, seed=seed, maxiter=maxiter, popsize=popsize
        )
    else:
        fitted = np.asarray(
            correction.default_params if params is None else params, dtype=float
        )
        objective = float(problem.objective(correction)(jnp.asarray(fitted)))
    train_metrics = problem.evaluate(correction, fitted, "train")
    result = {
        "baseline": baseline_key,
        "correction": correction_key,
        "expression": correction.expression,
        "params": fitted.tolist(),
        "n_params": len(fitted),
        "tree_nodes": correction.tree_nodes,
        "training_fitness": train_metrics.data_error + 0.01 * correction.tree_nodes,
        "optimizer_objective": objective,
        "train": asdict(train_metrics),
        "test": (
            asdict(problem.evaluate(correction, fitted, "test"))
            if include_test
            else None
        ),
        "converged": converged,
        "feasible": problem.is_feasible(correction, fitted),
        "optimizer_message": str(message),
        "seed": seed,
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--correction", choices=CORRECTIONS, default="identity")
    parser.add_argument("--params", type=float, nargs="*")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(
        args.baseline,
        args.correction,
        args.params,
        args.optimize,
        args.seed,
        args.maxiter,
        args.popsize,
        args.include_test,
    )
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
