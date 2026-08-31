"""Run one local correction on every baseline without accessing test data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from automodel.evaluate import BASELINES, I80PredictionProblem
from sr_traffic.sr.utils import is_v_unfeasible


def load_correction(model_path: Path):
    module_name = "agent3_" + model_path.parent.name
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CORRECTION


def fit_one(baseline_key, correction, seed, maxiter, popsize, polish_maxiter):
    started = time.perf_counter()
    problem = I80PredictionProblem(baseline_key)
    fitted, objective, converged, message = problem.fit(
        correction,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish_maxiter=polish_maxiter,
    )
    fitted = np.asarray(fitted, dtype=float)
    metrics = problem.evaluate(correction, fitted, "train")

    def individual(rho):
        return correction.function(rho, problem.flats, jnp.asarray(fitted))

    velocity_unfeasible = bool(
        is_v_unfeasible(
            individual,
            problem.rho_feasibility_grid,
            problem.S,
            problem.baseline.velocity,
            problem.baseline.coefficients,
        )
    )
    metrics_dict = asdict(metrics)
    finite = bool(np.all(np.isfinite(list(metrics_dict.values()))))
    feasible = finite and not velocity_unfeasible and objective < 100.0
    return {
        "baseline": baseline_key,
        "baseline_label": BASELINES[baseline_key].name,
        "correction": correction.name,
        "expression": correction.expression,
        "params": fitted.tolist(),
        "n_params": len(fitted),
        "bounds": [list(bound) for bound in correction.bounds],
        "tree_nodes": correction.tree_nodes,
        "E_data": metrics.data_error,
        "training_fitness": metrics.data_error + 0.01 * correction.tree_nodes,
        "optimizer_objective": objective,
        "train": metrics_dict,
        "split_evaluated": "train",
        "test": None,
        "converged": bool(converged),
        "optimizer_message": str(message),
        "feasible": feasible,
        "velocity_nonincreasing": not velocity_unfeasible,
        "all_metrics_finite": finite,
        "seed": seed,
        "maxiter": maxiter,
        "popsize": popsize,
        "polish_maxiter": polish_maxiter,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mb_estimate": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--polish-maxiter", type=int, default=50)
    args = parser.parse_args()

    correction = load_correction(args.model.resolve())
    run_started = time.perf_counter()
    results = {}
    for baseline_key in BASELINES:
        print(f"Fitting {correction.name} on {baseline_key}", flush=True)
        results[baseline_key] = fit_one(
            baseline_key,
            correction,
            args.seed,
            args.maxiter,
            args.popsize,
            args.polish_maxiter,
        )

    fitnesses = [entry["training_fitness"] for entry in results.values()]
    data_errors = [entry["E_data"] for entry in results.values()]
    payload = {
        "attempt": args.model.parent.name,
        "correction": correction.name,
        "expression": correction.expression,
        "default_params": list(correction.default_params),
        "bounds": [list(bound) for bound in correction.bounds],
        "n_params": len(correction.default_params),
        "tree_nodes": correction.tree_nodes,
        "selection_split": "train_first_60_percent",
        "test_evaluated": False,
        "optimizer": {
            "method": "I80PredictionProblem.fit (differential_evolution + Nelder-Mead polish)",
            "seed": args.seed,
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "polish_maxiter": args.polish_maxiter,
        },
        "mean_E_data": float(np.mean(data_errors)),
        "mean_training_fitness": float(np.mean(fitnesses)),
        "all_feasible": all(entry["feasible"] for entry in results.values()),
        "all_metrics_finite": all(
            entry["all_metrics_finite"] for entry in results.values()
        ),
        "total_runtime_seconds": time.perf_counter() - run_started,
        "peak_rss_mb_estimate": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "baselines": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
