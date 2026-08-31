"""Run one isolated automodel attempt on all I80 prediction baselines.

This runner deliberately exposes no test-split option and evaluates only
``split='train'``.
"""

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

from automodel.evaluate import BASELINES, I80PredictionProblem


def load_correction(model_path: Path):
    module_name = f"isolated_{model_path.parent.name}_model"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CORRECTION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt_dir", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument(
        "--polish-maxiter",
        type=int,
        default=50,
        help="Bounded Nelder-Mead iteration cap passed to fit().",
    )
    args = parser.parse_args()

    correction = load_correction(args.attempt_dir / "model.py")
    total_started = time.perf_counter()
    records = []
    for baseline_key in BASELINES:
        started = time.perf_counter()
        problem = I80PredictionProblem(baseline_key)
        params, optimizer_objective, converged, message = problem.fit(
            correction,
            seed=args.seed,
            maxiter=args.maxiter,
            popsize=args.popsize,
            polish_maxiter=args.polish_maxiter,
        )
        train = problem.evaluate(correction, params, "train")
        objective_check = float(
            problem.objective(correction)(jnp.asarray(params)).block_until_ready()
        )
        finite = all(
            bool(jnp.isfinite(value))
            for value in (
                train.data_error,
                train.rho_rrmse,
                train.velocity_rrmse,
                train.flow_rrmse,
            )
        )
        within_bounds = all(
            lower <= float(value) <= upper
            for value, (lower, upper) in zip(params, correction.bounds)
        )
        physically_feasible = objective_check < 100.0
        raw_fitness = train.data_error + 0.01 * correction.tree_nodes
        records.append(
            {
                "baseline": baseline_key,
                "params": [float(value) for value in params],
                "optimizer_objective": float(optimizer_objective),
                "objective_check": objective_check,
                "raw_evaluated_fitness": raw_fitness,
                "training_fitness": objective_check
                + 0.01 * correction.tree_nodes,
                "train": asdict(train),
                "converged": bool(converged),
                "optimizer_message": str(message),
                "physically_feasible": physically_feasible,
                "finite_metrics": finite,
                "within_declared_bounds": within_bounds,
                "accepted": bool(finite and physically_feasible and within_bounds),
                "runtime_seconds": time.perf_counter() - started,
                "peak_memory_mb": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 1024.0,
            }
        )

    payload = {
        "correction": correction.name,
        "expression": correction.expression,
        "default_params": list(correction.default_params),
        "bounds": [list(bound) for bound in correction.bounds],
        "n_params": len(correction.bounds),
        "tree_nodes": correction.tree_nodes,
        "dataset": "I80",
        "task": "prediction",
        "evaluated_split": "train",
        "test_accessed": False,
        "seed": args.seed,
        "optimizer": {
            "global": "scipy.differential_evolution",
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "polish": (
                "bounded Nelder-Mead as implemented by "
                "I80PredictionProblem.fit, maxiter="
                f"{args.polish_maxiter}"
            ),
        },
        "baselines": records,
        "mean_training_data_error": sum(
            item["train"]["data_error"] for item in records
        )
        / len(records),
        "mean_constraint_objective": sum(
            item["objective_check"] for item in records
        )
        / len(records),
        "mean_training_fitness": sum(
            item["training_fitness"] for item in records
        )
        / len(records),
        "runtime_seconds": time.perf_counter() - total_started,
        "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
    }
    output = args.attempt_dir / "results.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
