"""Plot and tabulate the frozen Automodel fundamental diagrams.

The outputs mirror ``sr_traffic.fd.results``.  Tables and plots compare each
calibrated basic fundamental diagram with its Automodel-selected correction.
Automodel selection is currently available only for the I80/prediction
benchmark.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from dctkit import config
from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from jax import vmap

from sr_traffic.data.data import build_dataset, preprocess_data
from sr_traffic.fd import diagrams
from sr_traffic.fd.automodel_registry import I80_PREDICTION_DIAGRAMS
from sr_traffic.utils import flat as traffic_flat
from sr_traffic.utils.godunov import godunov_solver

config()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results"


def rescale_rho_v_f(
    rho_p0: npt.NDArray,
    rho: npt.NDArray,
    velocity: npt.NDArray,
    flow: npt.NDArray,
    data_info: Mapping,
):
    """Restore the dimensional units used by the paper plots."""

    return (
        rho_p0 * data_info["density_max"],
        rho * data_info["density_max"],
        velocity * data_info["V"],
        flow * data_info["density_max"] * data_info["V"],
    )


def compute_errors(
    true_rho: npt.NDArray,
    true_velocity: npt.NDArray,
    true_flow: npt.NDArray,
    model_rho: npt.NDArray,
    model_velocity: npt.NDArray,
    model_flow: npt.NDArray,
) -> tuple[float, float, float]:
    """Return the density, velocity, and flow rRMSEs used in the paper."""

    rho_error = jnp.sqrt(jnp.sum((true_rho - model_rho) ** 2)) / jnp.sqrt(
        jnp.sum(true_rho**2)
    )
    velocity_error = jnp.sqrt(
        jnp.sum((true_velocity - model_velocity) ** 2)
    ) / jnp.sqrt(jnp.sum(true_velocity**2))
    flow_error = jnp.sqrt(jnp.sum((true_flow - model_flow) ** 2)) / jnp.sqrt(
        jnp.sum(true_flow**2)
    )
    return float(rho_error), float(velocity_error), float(flow_error)


def prepare_flats(data_info: Mapping):
    """Build the DEC flats used by both the solver and Automodel expressions."""

    complex_ = data_info["S"]
    zeros_p = C.CochainP0(complex_, jnp.zeros_like(data_info["vP0"][:, 0]))
    zeros_d = C.CochainD0(complex_, jnp.zeros_like(data_info["density"][:, 0]))
    all_flats = traffic_flat.define_flats(complex_, zeros_p, zeros_d)
    flat_linear_left_d = all_flats["flat_linear_left_D"]

    def flat_left_wrap(values):
        return flat_linear_left_d(C.CochainD0(complex_, values)).coeffs

    flat_left = vmap(flat_left_wrap)
    return {
        "linear_left": flat_linear_left_d,
        "linear_right": all_flats["flat_linear_right_D"],
        "linear_left_P": all_flats["flat_linear_left_P"],
        "linear_right_P": all_flats["flat_linear_right_P"],
        "flat_left_v": flat_left,
    }


def make_boundary_array(data_info: Mapping) -> jnp.ndarray:
    boundary = jnp.zeros((len(data_info["rho_bnd"]), data_info["num_t_points"]))
    for index, values in data_info["rho_bnd"].items():
        boundary = boundary.at[int(index), :].set(values[: data_info["num_t_points"]])
    return boundary


def simulate_model(
    flux: Callable,
    flux_derivative: Callable,
    data_info: Mapping,
    flats: Mapping,
    boundary: npt.NDArray,
):
    """Run the same LWR/Godunov simulation used by the paper results script."""

    rho, velocity, flow = godunov_solver(
        data_info["rho_0"],
        data_info["S"],
        boundary,
        flux,
        flux_derivative,
        data_info["delta_t_refined"],
        0,
        flats,
        data_info["num_t_points"],
    )
    flat_rho = C.CochainD1(data_info["S"], flats["flat_left_v"](rho.T)[:, :, 0].T)
    rho_p0 = C.star(flat_rho).coeffs
    step = data_info["step"]
    velocity = velocity[:, ::step]
    flow = flow[:, ::step]
    true_velocity = data_info["v"]
    true_flow = data_info["flow"]

    # Match the paper script: boundary and initial velocity/flow are observed.
    velocity = velocity.at[:, 0].set(true_velocity[:, 0])
    velocity = velocity.at[0, :].set(true_velocity[0, :])
    velocity = velocity.at[-3:, :].set(true_velocity[-3:, :])
    flow = flow.at[:, 0].set(true_flow[:, 0])
    flow = flow.at[0, :].set(true_flow[0, :])
    flow = flow.at[-3:, :].set(true_flow[-3:, :])

    rho_p0, rho, velocity, flow = rescale_rho_v_f(
        rho_p0, rho, velocity, flow, data_info
    )
    return {
        "rho": rho[:, ::step],
        "rhoP0": rho_p0[:-1, ::step],
        "v": velocity,
        "f": flow,
    }


def make_model_functions(
    complex_: SimplicialComplex, flats: Mapping
) -> tuple[dict[str, tuple[Callable, Callable]], list[str], list[str]]:
    """Create baseline and Automodel fluxes in paired display order."""

    models = {}
    plot_names = []
    corrected_names = []
    for diagram in I80_PREDICTION_DIAGRAMS:

        def baseline_flux(rho, diagram=diagram):
            return diagram.baseline_flux(rho, *diagram.baseline_coefficients)

        def corrected_flux(rho, diagram=diagram):
            return diagrams.multiplicatively_corrected_flux(
                rho,
                diagram.baseline_flux,
                diagram.baseline_coefficients,
                diagram.correction,
                flats["linear_right_P"],
                diagram.correction_coefficients,
            )

        baseline_derivative = diagrams.define_flux_der(complex_, baseline_flux)
        corrected_derivative = diagrams.define_flux_der(complex_, corrected_flux)
        corrected_name = f"Automodel-{diagram.name}"
        models[diagram.name] = (baseline_flux, baseline_derivative)
        models[corrected_name] = (corrected_flux, corrected_derivative)
        plot_names.extend((diagram.name, corrected_name))
        corrected_names.append(corrected_name)
    return models, plot_names, corrected_names


def compute_true_fields(data_info: Mapping, flat_left: Callable):
    """Interpolate observations to primal cochains and restore paper units."""

    complex_ = data_info["S"]
    flat_density = C.CochainD1(complex_, flat_left(data_info["density"].T)[:, :, 0].T)
    rho_p0 = C.star(flat_density).coeffs[:-1]
    flat_velocity = C.CochainD1(complex_, flat_left(data_info["v"].T)[:, :, 0].T)
    velocity = C.star(flat_velocity).coeffs[:-1]
    flat_flow = flat_left(data_info["flow"].T)[:, :, 0].T
    flow = C.star(C.CochainD1(complex_, flat_flow)).coeffs[:-1]
    return rescale_rho_v_f(rho_p0, data_info["density"], velocity, flow, data_info)


def format_latex_entry(value: float, rank: int, is_best: bool) -> str:
    formatted = f"{value:.3f} ({rank})"
    return f"\\textbf{{{formatted}}}" if is_best else formatted


def format_markdown_entry(value: float, rank: int, is_best: bool) -> str:
    formatted = f"{value:.3f} ({rank})"
    return f"**{formatted}**" if is_best else formatted


def save_error_tables(
    results: Mapping[str, Mapping[str, npt.NDArray]],
    true_density: npt.NDArray,
    true_velocity: npt.NDArray,
    true_flow: npt.NDArray,
    train_idx: npt.NDArray,
    test_idx: npt.NDArray,
    output_dir: Path,
) -> list[dict]:
    """Write paper-compatible LaTeX/Markdown tables and raw JSON metrics."""

    train_slice = (slice(None), train_idx)
    test_slice = (slice(None), test_idx)
    rows = []
    for name, model in results.items():
        rho_train, velocity_train, _ = compute_errors(
            true_density[train_slice],
            true_velocity[train_slice],
            true_flow[train_slice],
            model["rho"][train_slice],
            model["v"][train_slice],
            model["f"][train_slice],
        )
        rho_test, velocity_test, _ = compute_errors(
            true_density[test_slice],
            true_velocity[test_slice],
            true_flow[test_slice],
            model["rho"][test_slice],
            model["v"][test_slice],
            model["f"][test_slice],
        )
        rows.append(
            {
                "model": name,
                "rho_train": rho_train,
                "velocity_train": velocity_train,
                "rho_test": rho_test,
                "velocity_test": velocity_test,
            }
        )

    error_array = np.asarray(
        [
            [
                row["rho_train"],
                row["velocity_train"],
                row["rho_test"],
                row["velocity_test"],
            ]
            for row in rows
        ]
    )
    ranks = np.argsort(np.argsort(error_array, axis=0), axis=0) + 1
    average_ranks = np.mean(ranks, axis=1)
    best_average_rank = np.min(average_ranks)

    latex_rows = []
    markdown_rows = []
    for index, row in enumerate(rows):
        row["ranks"] = ranks[index].tolist()
        row["average_rank"] = float(average_ranks[index])
        latex_values = [
            row["model"],
            *[
                format_latex_entry(
                    error_array[index, metric],
                    ranks[index, metric],
                    ranks[index, metric] == 1,
                )
                for metric in range(4)
            ],
            (
                f"\\textbf{{{average_ranks[index]:.2f}}}"
                if average_ranks[index] == best_average_rank
                else f"{average_ranks[index]:.2f}"
            ),
        ]
        markdown_values = [
            row["model"],
            *[
                format_markdown_entry(
                    error_array[index, metric],
                    ranks[index, metric],
                    ranks[index, metric] == 1,
                )
                for metric in range(4)
            ],
            (
                f"**{average_ranks[index]:.2f}**"
                if average_ranks[index] == best_average_rank
                else f"{average_ranks[index]:.2f}"
            ),
        ]
        latex_rows.append(latex_values)
        markdown_rows.append(markdown_values)

    caption = (
        "Relative errors between the actual and the computed density and velocity "
        "(training and test) for the prediction task, including the calibrated "
        "baselines and Automodel variants. In bold, the best-performing models "
        "for each metric considered."
    )
    latex_table = (
        r"""\begin{table}[H]
    \caption{"""
        + caption
        + r"""}
    \begin{center}
        \begin{tabular}{c c c c c c}
            \toprule
            Model & $E^{\text{tr}}_\rho$ & $E^{\text{tr}}_v$ & $E^{\text{ts}}_\rho$ & $E^{\text{ts}}_v$ & Avg Rank\\
            \midrule
"""
        + "\n".join("        " + " & ".join(row) + r"\\" for row in latex_rows)
        + r"""
            \bottomrule
        \end{tabular}
    \end{center}
    \label{tab:errors_i80_prediction_automodel}
\end{table}
"""
    )
    markdown_table = "\n".join(
        [
            "## Relative errors — I80 prediction: baseline and Automodel",
            "",
            caption,
            "",
            "| Model | $E^{\\mathrm{tr}}_\\rho$ | $E^{\\mathrm{tr}}_v$ | $E^{\\mathrm{ts}}_\\rho$ | $E^{\\mathrm{ts}}_v$ | Avg Rank |",
            "|---|---:|---:|---:|---:|---:|",
            *["| " + " | ".join(row) + " |" for row in markdown_rows],
            "",
        ]
    )
    (output_dir / "error_table.tex").write_text(latex_table, encoding="utf-8")
    (output_dir / "error_table.md").write_text(markdown_table, encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def _split_slices(train_idx, test_idx):
    return (slice(None), train_idx), (slice(None), test_idx)


def plot_diagrams(
    results: Mapping[str, Mapping[str, npt.NDArray]],
    true_rho_p0: npt.NDArray,
    true_velocity: npt.NDArray,
    true_flow: npt.NDArray,
    quantity: str,
    train_idx: npt.NDArray,
    test_idx: npt.NDArray,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot model and observed velocity/flux fundamental diagrams."""

    if quantity == "velocity":
        observations, field, ylabel = true_velocity, "v", r"$V(\rho)$ (ft/s)"
    elif quantity == "flux":
        observations, field, ylabel = true_flow, "f", r"$\rho V(\rho)$ (veh/s)"
    else:
        raise ValueError(f"Unsupported diagram quantity: {quantity}")

    train_slice, test_slice = _split_slices(train_idx, test_idx)
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 4), squeeze=False)
    axes = axes[0]
    for axis, name in zip(axes, names, strict=True):
        axis.scatter(
            results[name]["rhoP0"][1:-3, 1:].flatten(),
            results[name][field][1:-3, 1:].flatten(),
            marker=".",
            s=5,
            label="Model",
            c="#ff0000",
            zorder=1,
        )
        axis.scatter(
            true_rho_p0[train_slice].flatten(),
            observations[train_slice].flatten(),
            marker=".",
            s=5,
            label="Training data",
            c="#4757fb",
            zorder=0,
        )
        axis.scatter(
            true_rho_p0[test_slice].flatten(),
            observations[test_slice].flatten(),
            marker=".",
            s=5,
            label="Test data",
            c="#0ea4f0",
            zorder=0,
        )
        axis.set_xlabel(r"$\rho$ (veh/ft)")
        axis.set_ylabel(ylabel)
        axis.set_title(name)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fancybox=True,
        shadow=True,
        fontsize=14,
        markerscale=3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_rect(xy, width: float, height: float, color: str):
    return patches.Rectangle(
        xy,
        width,
        height,
        linewidth=2,
        edgecolor=color,
        facecolor="none",
        clip_on=False,
        zorder=10,
    )


def plot_spatiotemporal_fields(
    results: Mapping[str, Mapping[str, npt.NDArray]],
    true_density: npt.NDArray,
    true_velocity: npt.NDArray,
    x_coordinates: npt.NDArray,
    time_coordinates: npt.NDArray,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot paper-style density and velocity fields for data and every model."""

    names = list(results)
    fields = (
        (true_density, "rho", r"$\rho$ (veh/ft)", [0, 0.1, 0.2]),
        (true_velocity, "v", r"$v$ (ft/s)", [1, 40, 75]),
    )
    fig, axes = plt.subplots(
        2, len(names) + 1, figsize=(3 * (len(names) + 1), 4.5), squeeze=False
    )
    x_mesh, time_mesh = np.meshgrid(x_coordinates[:-3], time_coordinates)
    for row_index, (observed, field, colorbar_name, ticks) in enumerate(fields):
        norm = mcolors.Normalize(vmin=np.min(observed), vmax=np.max(observed))
        data_plot = axes[row_index, 0].contourf(
            time_mesh,
            x_mesh,
            observed[:-3].T,
            levels=100,
            cmap="rainbow",
            norm=norm,
        )
        axes[row_index, 0].set_title("Data")
        for column_index, name in enumerate(names, start=1):
            axes[row_index, column_index].contourf(
                time_mesh,
                x_mesh,
                results[name][field][:-3].T,
                levels=100,
                cmap="rainbow",
                norm=norm,
            )
            if row_index == 0:
                axes[row_index, column_index].set_title(name)

        # I80 prediction split, matching the highlighted paper-result regions.
        axes[row_index, 0].add_patch(make_rect((2.5, 10), 535.0, 1500.0, "red"))
        axes[row_index, 0].add_patch(make_rect((542.5, 10), 355.0, 1500.0, "#FF7F50"))
        fig.colorbar(
            data_plot,
            ax=axes[row_index, :],
            orientation="vertical",
            fraction=0.05,
            pad=0.01,
            label=colorbar_name,
            ticks=ticks,
        )

    for row_index in range(2):
        axes[row_index, 0].set_ylabel("x (ft)")
        axes[row_index, 0].set_yticks([10, 760, 1510])
        for axis in axes[row_index]:
            axis.set_xticks([])
            if axis is not axes[row_index, 0]:
                axis.set_yticks([])
    for axis in axes[-1]:
        axis.set_xlabel("t (s)")
        axis.set_xticks([0, 450, 900])

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_actual(
    results: Mapping[str, Mapping[str, npt.NDArray]],
    observations: npt.NDArray,
    field: str,
    label: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot predicted-versus-observed fields in the style of the paper script."""

    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 4), squeeze=False)
    axes = axes[0]
    observed_flat = observations.flatten()
    for axis, name in zip(axes, names, strict=True):
        axis.scatter(
            observed_flat,
            results[name][field].flatten(),
            marker=".",
            s=5,
            c="#0ea4f0",
        )
        axis.scatter(observed_flat, observed_flat, marker=".", s=5, c="#ff0000")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"{label} true")
        axis.set_ylabel(f"{label} predicted")
        axis.set_title(name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_plot_suite(
    results: Mapping[str, Mapping[str, npt.NDArray]],
    true_rho_p0: npt.NDArray,
    true_density: npt.NDArray,
    true_velocity: npt.NDArray,
    true_flow: npt.NDArray,
    train_idx: npt.NDArray,
    test_idx: npt.NDArray,
    x_coordinates: npt.NDArray,
    time_coordinates: npt.NDArray,
    output_dir: Path,
    suffix: str,
    dpi: int,
) -> None:
    plot_diagrams(
        results,
        true_rho_p0,
        true_velocity,
        true_flow,
        "flux",
        train_idx,
        test_idx,
        output_dir / f"flux_{suffix}.png",
        dpi,
    )
    plot_diagrams(
        results,
        true_rho_p0,
        true_velocity,
        true_flow,
        "velocity",
        train_idx,
        test_idx,
        output_dir / f"velocity_{suffix}.png",
        dpi,
    )
    plot_spatiotemporal_fields(
        results,
        true_density,
        true_velocity,
        x_coordinates,
        time_coordinates,
        output_dir / f"rho_v_f_plot_{suffix}.png",
        dpi,
    )
    plot_predicted_actual(
        results,
        true_flow,
        "f",
        "Flux",
        output_dir / f"pred_actual_flux_{suffix}.png",
        dpi,
    )
    plot_predicted_actual(
        results,
        true_velocity,
        "v",
        "Velocity",
        output_dir / f"pred_actual_velocity_{suffix}.png",
        dpi,
    )


def run(output_dir: Path, dpi: int = 300, tables_only: bool = False) -> list[dict]:
    """Simulate all paired models and generate tables and optional plots."""

    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_info = preprocess_data("I80")
    _, _, training_data, test_data = build_dataset(
        data_info["t_sampled_circ"],
        data_info["S"],
        data_info["density"],
        data_info["v"],
        data_info["flow"],
        "prediction",
    )
    flats = prepare_flats(data_info)
    boundary = make_boundary_array(data_info)
    model_functions, plot_names, corrected_names = make_model_functions(
        data_info["S"], flats
    )

    results = {}
    for index, (name, (flux, derivative)) in enumerate(
        model_functions.items(), start=1
    ):
        print(f"[{index}/{len(model_functions)}] Simulating {name}", flush=True)
        results[name] = simulate_model(flux, derivative, data_info, flats, boundary)

    true_rho_p0, true_density, true_velocity, true_flow = compute_true_fields(
        data_info, flats["flat_left_v"]
    )
    train_idx = jnp.arange(
        training_data[0, 0], training_data[-1, 0] + 1, dtype=jnp.int64
    )
    test_idx = jnp.arange(test_data[0, 0], test_data[-1, 0] + 1, dtype=jnp.int64)
    metrics = save_error_tables(
        results,
        true_density,
        true_velocity,
        true_flow,
        train_idx,
        test_idx,
        output_dir,
    )

    if not tables_only:
        x_coordinates = (
            (data_info["x_sampled"][1:] + data_info["x_sampled"][:-1])
            / 2
            * data_info["L_dim"]
        )
        time_coordinates = data_info["t_sampled_circ"] * data_info["t_len"]
        plot_results = {name: results[name] for name in plot_names}
        save_plot_suite(
            plot_results,
            true_rho_p0,
            true_density,
            true_velocity,
            true_flow,
            train_idx,
            test_idx,
            x_coordinates,
            time_coordinates,
            output_dir,
            "I80_prediction_automodel",
            dpi,
        )
        corrected_results = {name: results[name] for name in corrected_names}
        save_plot_suite(
            corrected_results,
            true_rho_p0,
            true_density,
            true_velocity,
            true_flow,
            train_idx,
            test_idx,
            x_coordinates,
            time_coordinates,
            output_dir,
            "I80_prediction_automodel_corrected",
            dpi,
        )

    print(
        f"Saved Automodel results to {output_dir} in {perf_counter() - started:.2f} s",
        flush=True,
    )
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--road_name",
        "--road-name",
        default="I80",
        choices=("I80",),
        help="Automodel selections are currently frozen only for I80.",
    )
    parser.add_argument(
        "--task",
        default="prediction",
        choices=("prediction",),
        help="Automodel selections are currently frozen only for prediction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: results/I80/prediction/automodel",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Generate tables/JSON without rendering plots.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = (
        args.output_dir or RESULTS_ROOT / args.road_name / args.task / "automodel"
    )
    run(output_dir=output_dir, dpi=args.dpi, tables_only=args.tables_only)


if __name__ == "__main__":
    main()
