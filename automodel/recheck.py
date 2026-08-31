"""Independently re-evaluate a saved automodel correction on the training split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

import jax.numpy as jnp

from automodel.evaluate import BASELINES, I80PredictionProblem


def load_correction(model_path: Path):
    spec = importlib.util.spec_from_file_location(
        "automodel_saved_candidate", model_path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import candidate model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CORRECTION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--params", type=float, nargs="*", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    correction = load_correction(args.model)
    problem = I80PredictionProblem(args.baseline)
    params = jnp.asarray(args.params)
    objective = float(problem.objective(correction)(params))
    metrics = problem.evaluate(correction, params, "train")
    result = {
        "baseline": args.baseline,
        "model": str(args.model),
        "correction": correction.name,
        "expression": correction.expression,
        "params": list(args.params),
        "tree_nodes": correction.tree_nodes,
        "optimizer_objective": objective,
        "training_fitness": metrics.data_error + 0.01 * correction.tree_nodes,
        "train": asdict(metrics),
        "feasible": problem.is_feasible(correction, params),
        "test_evaluated": False,
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    payload = json.dumps(result, indent=2) + "\n"
    print(payload, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
