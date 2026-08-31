"""Fit one local correction on all baselines without accessing test data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from automodel.evaluate import BASELINES, I80PredictionProblem


META1_WINNER_FITNESS = {
    "greenshields": 7.004364522852866,
    "weidmann": 6.36439193255737,
    "triangular": 7.64845982421528,
    "idm": 7.220317321727018,
    "del_castillo": 6.465030738886299,
}


def load_correction(model_path: Path):
    module_name = "meta2_agent3_" + model_path.parent.name
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CORRECTION


def params_in_bounds(params, bounds, atol=1.0e-10):
    return all(
        low - atol <= value <= high + atol
        for value, (low, high) in zip(params, bounds, strict=True)
    )


def fit_one(baseline_key, correction, seed, maxiter, popsize, polish_maxiter):
    started = time.perf_counter()
    problem = I80PredictionProblem(baseline_key)
    default_feasible = problem.is_feasible(correction, correction.default_params)
    default_objective = float(problem.objective(correction)(correction.default_params))
    fitted, objective, converged, message = problem.fit(
        correction,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish_maxiter=polish_maxiter,
    )
    fitted = np.asarray(fitted, dtype=float)
    metrics = problem.evaluate(correction, fitted, "train")
    metrics_dict = asdict(metrics)
    finite = bool(np.all(np.isfinite(list(metrics_dict.values()))))
    feasible = problem.is_feasible(correction, fitted)
    in_bounds = params_in_bounds(fitted, correction.bounds)
    eligible = finite and feasible and in_bounds and objective < 100.0
    fitness = metrics.data_error + 0.01 * correction.tree_nodes
    parent_fitness = META1_WINNER_FITNESS[baseline_key]
    return {
        "baseline": baseline_key,
        "baseline_label": BASELINES[baseline_key].name,
        "params": fitted.tolist(),
        "n_params": len(fitted),
        "bounds": [list(bound) for bound in correction.bounds],
        "E_data": metrics.data_error,
        "training_fitness": fitness,
        "meta1_winner_training_fitness": parent_fitness,
        "fitness_change_vs_meta1_winner": fitness - parent_fitness,
        "optimizer_objective": objective,
        "train": metrics_dict,
        "split_evaluated": "train",
        "test": None,
        "default_params": list(correction.default_params),
        "default_feasible": default_feasible,
        "default_objective": default_objective,
        "converged": bool(converged),
        "optimizer_message": str(message),
        "finite": finite,
        "feasible": feasible,
        "in_bounds": in_bounds,
        "eligible": eligible,
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
    parser.add_argument("--maxiter", type=int, default=2)
    parser.add_argument("--popsize", type=int, default=3)
    parser.add_argument("--polish-maxiter", type=int, default=10)
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

    data_errors = [entry["E_data"] for entry in results.values()]
    fitnesses = [entry["training_fitness"] for entry in results.values()]
    parent_fitnesses = [
        entry["meta1_winner_training_fitness"] for entry in results.values()
    ]
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
            "method": "I80PredictionProblem.fit (differential_evolution + bounded Nelder-Mead)",
            "seed": args.seed,
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "polish_maxiter": args.polish_maxiter,
        },
        "meta1_winner": {
            "source": "meta_1/agent_2/attempt_2",
            "expression": "1 + (delta flat_right_P exp(c1*rho)) *_3 exp(c2*rho)",
            "tree_nodes": 15,
            "mean_training_fitness": float(np.mean(parent_fitnesses)),
        },
        "mean_E_data": float(np.mean(data_errors)),
        "mean_training_fitness": float(np.mean(fitnesses)),
        "mean_fitness_change_vs_meta1_winner": float(
            np.mean(fitnesses) - np.mean(parent_fitnesses)
        ),
        "all_default_feasible": all(
            entry["default_feasible"] for entry in results.values()
        ),
        "all_finite": all(entry["finite"] for entry in results.values()),
        "all_feasible": all(entry["feasible"] for entry in results.values()),
        "all_in_bounds": all(entry["in_bounds"] for entry in results.values()),
        "all_eligible": all(entry["eligible"] for entry in results.values()),
        "total_runtime_seconds": time.perf_counter() - run_started,
        "peak_rss_mb_estimate": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "baselines": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
