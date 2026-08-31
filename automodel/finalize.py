"""Run the frozen automodel selections on the held-out I80 prediction interval.

This module is deliberately separate from structural search.  Running it reveals
the final 40% time split, so its output must not be used to retune expressions or
coefficients without introducing a fresh external check.
"""

from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from automodel.evaluate import I80PredictionProblem
from automodel.final_model import SELECTED_CORRECTIONS
from automodel.model import CORRECTIONS

README_TRIANGULAR_SR_TEST_ERROR = 8.0742627959
OUTPUT_JSON = Path("automodel/final_results.json")
OUTPUT_MARKDOWN = Path("automodel/final_results.md")


def _peak_memory_mb() -> float:
    """Return peak resident memory for this process on Linux."""

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_final_check() -> dict:
    """Evaluate each selected correction exactly once on the external split."""

    started = time.perf_counter()
    identity = CORRECTIONS["identity"]
    results = {}

    for baseline_key, selected in SELECTED_CORRECTIONS.items():
        baseline_started = time.perf_counter()
        problem = I80PredictionProblem(baseline_key)

        train_started = time.perf_counter()
        train_metrics = problem.evaluate(
            selected.correction, selected.params, split="train"
        )
        train_seconds = time.perf_counter() - train_started

        identity_started = time.perf_counter()
        identity_test = problem.evaluate(identity, (), split="test")
        identity_test_seconds = time.perf_counter() - identity_started

        selected_started = time.perf_counter()
        selected_test = problem.evaluate(
            selected.correction, selected.params, split="test"
        )
        selected_test_seconds = time.perf_counter() - selected_started

        results[baseline_key] = {
            "baseline": problem.baseline.name,
            "correction": selected.correction.name,
            "expression": selected.correction.expression,
            "params": list(selected.params),
            "n_params": len(selected.params),
            "tree_nodes": selected.correction.tree_nodes,
            "source": selected.source,
            "feasible": problem.is_feasible(selected.correction, selected.params),
            "train": asdict(train_metrics),
            "training_fitness": (
                train_metrics.data_error + 0.01 * selected.correction.tree_nodes
            ),
            "identity_test": asdict(identity_test),
            "selected_test": asdict(selected_test),
            "test_data_error_delta": (
                selected_test.data_error - identity_test.data_error
            ),
            "runtime_seconds": {
                "train": train_seconds,
                "identity_test": identity_test_seconds,
                "selected_test": selected_test_seconds,
                "total": time.perf_counter() - baseline_started,
            },
            "peak_memory_mb_after_model": _peak_memory_mb(),
        }

    identity_errors = np.asarray(
        [row["identity_test"]["data_error"] for row in results.values()]
    )
    selected_errors = np.asarray(
        [row["selected_test"]["data_error"] for row in results.values()]
    )
    triangular_error = results["triangular"]["selected_test"]["data_error"]
    summary = {
        "mean_identity_test_data_error": float(np.mean(identity_errors)),
        "mean_selected_test_data_error": float(np.mean(selected_errors)),
        "mean_test_data_error_delta": float(
            np.mean(selected_errors) - np.mean(identity_errors)
        ),
        "all_selected_feasible": all(row["feasible"] for row in results.values()),
        "test_improvements_over_identity": int(
            np.sum(selected_errors < identity_errors)
        ),
        "total_baselines": len(results),
        "triangular_readme_sr_test_data_error": README_TRIANGULAR_SR_TEST_ERROR,
        "triangular_selected_test_data_error": triangular_error,
        "triangular_vs_readme_sr_delta": (
            triangular_error - README_TRIANGULAR_SR_TEST_ERROR
        ),
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_mb": _peak_memory_mb(),
    }
    return {
        "protocol": (
            "Frozen training-selected models evaluated once on the final 40% "
            "of the I80 prediction interval; test metrics were not used for selection."
        ),
        "results": results,
        "summary": summary,
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# Frozen-model I80/prediction external check",
        "",
        payload["protocol"],
        "",
        "| Baseline | Expression | Parameters | Train fitness | Identity test E_data | Selected test E_data | Delta | Test rho rRMSE | Test v rRMSE | Test flow rRMSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["results"].values():
        selected = row["selected_test"]
        lines.append(
            "| {baseline} | `{expression}` | `{params}` | {fitness:.6f} | "
            "{identity:.6f} | {test:.6f} | {delta:+.6f} | {rho:.6f} | "
            "{velocity:.6f} | {flow:.6f} |".format(
                baseline=row["baseline"],
                expression=row["expression"],
                params=", ".join(f"{value:.10g}" for value in row["params"]),
                fitness=row["training_fitness"],
                identity=row["identity_test"]["data_error"],
                test=selected["data_error"],
                delta=row["test_data_error_delta"],
                rho=selected["rho_rrmse"],
                velocity=selected["velocity_rrmse"],
                flow=selected["flow_rrmse"],
            )
        )

    summary = payload["summary"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Mean identity test E_data: {summary['mean_identity_test_data_error']:.6f}",
            f"- Mean selected test E_data: {summary['mean_selected_test_data_error']:.6f}",
            f"- Mean delta: {summary['mean_test_data_error_delta']:+.6f}",
            f"- Per-baseline improvements: {summary['test_improvements_over_identity']}/{summary['total_baselines']}",
            f"- Triangular delta versus README SR test score: {summary['triangular_vs_readme_sr_delta']:+.6f}",
            f"- Full-check runtime: {summary['runtime_seconds']:.3f} s",
            f"- Peak resident memory: {summary['peak_memory_mb']:.1f} MB",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    payload = run_final_check()
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MARKDOWN.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
