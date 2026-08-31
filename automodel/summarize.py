"""Build the durable automodel leaderboard and training-progress plot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE_ORDER = [
    "greenshields",
    "weidmann",
    "triangular",
    "idm",
    "del_castillo",
]


def _within_bounds(params, bounds):
    return all(
        lower - 1.0e-12 <= float(value) <= upper + 1.0e-12
        for value, (lower, upper) in zip(params, bounds)
    )


def _attempt_rows(result_path: Path):
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    entries = payload["baselines"]
    if isinstance(entries, dict):
        entries = [dict(row, baseline=key) for key, row in entries.items()]
    bounds = payload.get("bounds", [])
    output = []
    for row in entries:
        params = row.get("params", [])
        row_bounds = row.get("bounds", bounds)
        finite = row.get(
            "finite", row.get("finite_metrics", row.get("all_metrics_finite", True))
        )
        feasible = row.get(
            "feasible",
            row.get("physically_feasible", row.get("feasible_and_finite", True)),
        )
        within = row.get(
            "within_bounds",
            row.get(
                "within_declared_bounds",
                row.get("in_bounds", _within_bounds(params, row_bounds)),
            ),
        )
        objective = float(
            row.get("optimizer_objective", row.get("objective_check", 100.0))
        )
        accepted = row.get(
            "eligible", row.get("candidate_eligible", row.get("accepted", True))
        )
        eligible = bool(
            finite and feasible and within and accepted and objective < 100.0
        )
        train = row.get("train", {})
        output.append(
            {
                "source": str(result_path.relative_to(ROOT)),
                "meta": int(result_path.parts[-4].split("_")[-1]),
                "agent": result_path.parts[-3],
                "attempt": result_path.parts[-2],
                "baseline": row["baseline"],
                "correction": payload["correction"],
                "expression": payload["expression"],
                "params": json.dumps(params),
                "n_params": payload.get("n_params", len(params)),
                "tree_nodes": payload["tree_nodes"],
                "data_error": float(train.get("data_error", row.get("E_data"))),
                "training_fitness": float(row["training_fitness"]),
                "rho_rrmse": float(train["rho_rrmse"]),
                "velocity_rrmse": float(train["velocity_rrmse"]),
                "flow_rrmse": float(train["flow_rrmse"]),
                "eligible": eligible,
                "runtime_seconds": float(row.get("runtime_seconds", 0.0)),
                "peak_memory_mb": float(
                    row.get("peak_memory_mb", row.get("peak_rss_mb_estimate", 0.0))
                ),
            }
        )
    return output


def _baseline_rows():
    paths = {
        "greenshields": ROOT / "automodel/baseline/greenshields_identity.json",
        "weidmann": ROOT / "automodel/baseline/weidmann_identity.json",
        "triangular": ROOT / "automodel/baseline_triangular_identity.json",
        "idm": ROOT / "automodel/baseline/idm_identity.json",
        "del_castillo": ROOT / "automodel/baseline/del_castillo_identity.json",
    }
    rows = []
    for baseline, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        train = payload["train"]
        rows.append(
            {
                "source": str(path.relative_to(ROOT)),
                "meta": 0,
                "agent": "baseline",
                "attempt": "identity",
                "baseline": baseline,
                "correction": "identity",
                "expression": "1",
                "params": "[]",
                "n_params": 0,
                "tree_nodes": 1,
                "data_error": train["data_error"],
                "training_fitness": payload["training_fitness"],
                "rho_rrmse": train["rho_rrmse"],
                "velocity_rrmse": train["velocity_rrmse"],
                "flow_rrmse": train["flow_rrmse"],
                "eligible": bool(payload.get("feasible", True)),
                "runtime_seconds": payload["runtime_seconds"],
                "peak_memory_mb": payload["peak_memory_mb"],
            }
        )
    return rows


def main():
    rows = _baseline_rows()
    for path in sorted(ROOT.glob("meta_*/agent_*/attempt_*/results.json")):
        rows.extend(_attempt_rows(path))

    fields = list(rows[0])
    leaderboard_path = ROOT / "automodel/leaderboard.csv"
    with leaderboard_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    unique_sources = {}
    for row in rows:
        if row["meta"] == 0:
            continue
        key = row["source"]
        entry = unique_sources.setdefault(
            key,
            {
                "correction": row["correction"],
                "expression": row["expression"],
                "tree_nodes": row["tree_nodes"],
                "fitness": [],
            },
        )
        if row["eligible"]:
            entry["fitness"].append(row["training_fitness"])
    lines = [
        "# Expressions evaluated during automodel search",
        "",
        "All scores use the first 60% I80/prediction selection interval; test data were not accessed.",
        "",
        "| Source | Expression | Nodes | Eligible baselines | Mean eligible fitness |",
        "|---|---|---:|---:|---:|",
    ]
    for source, entry in sorted(unique_sources.items()):
        scores = entry["fitness"]
        mean = f"{np.mean(scores):.6f}" if scores else "—"
        expression = entry["expression"].replace("|", "\\|")
        lines.append(
            f"| `{source}` | `{expression}` | {entry['tree_nodes']} | {len(scores)}/5 | {mean} |"
        )
    (ROOT / "automodel/expressions.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    best_by_stage = []
    for maximum_meta in (0, 1, 2):
        values = []
        for baseline in BASELINE_ORDER:
            candidates = [
                row["training_fitness"]
                for row in rows
                if row["eligible"]
                and row["baseline"] == baseline
                and row["meta"] <= maximum_meta
            ]
            values.append(min(candidates))
        best_by_stage.append(values)

    positions = np.arange(len(BASELINE_ORDER))
    width = 0.25
    fig, axis = plt.subplots(figsize=(10, 4.8))
    for offset, (label, values) in enumerate(
        zip(("Baseline", "After meta 1", "After meta 2"), best_by_stage)
    ):
        axis.bar(positions + (offset - 1) * width, values, width, label=label)
    axis.set_xticks(
        positions, ["Greenshields", "Weidmann", "Triangular", "IDM", "Del Castillo"]
    )
    axis.set_ylabel("Training selection fitness (lower is better)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "automodel/training_fitness_progress.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
