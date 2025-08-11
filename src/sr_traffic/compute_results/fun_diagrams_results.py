import numpy as np
import jax.numpy as jnp
from jax import vmap
from dctkit.dec import cochain as C
from dctkit.dec.flat import flat
from sr_traffic.data.data import preprocess_data, build_dataset
from sr_traffic.compute_results.plots import plot_velocity_flux_density
from sr_traffic.utils import fund_diagrams as tf_utils
from sr_traffic.utils import flat as tf_flat
from sr_traffic.utils.primitives import *
from sr_traffic.utils.godunov import godunov_solver
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches


def sr_term(rho, flats, params, task):
    if task == "prediction":
        ones = C.Cochain(rho.dim, rho.is_primal, rho.complex, jnp.ones_like(rho.coeffs))
        conv_term = C.convolution(
            C.codifferential(
                flats["linear_left_P"](C.exp(C.scalar_mul(rho, params[0])))
            ),
            C.exp(C.scalar_mul(rho, params[1])),
            1,
        )
        sr_term = C.add(ones, conv_term)
    elif task == "reconstruction":
        sr_term = C.square(
            C.exp(
                C.convolution(
                    C.codifferential(flats["linear_right_P"](rho)),
                    C.scalar_mul(rho, params[0]),
                    3,
                )
            )
        )
    return sr_term


def sr_flux_generator(flux, flats, params, task):
    def sr_flux(rho):
        return C.cochain_mul(flux(rho), sr_term(rho, flats, params, task))

    return sr_flux


def rescale_rho_v_f(rhoP0, rho, v, f, data_info):
    return (
        rhoP0 * data_info["density_max"],
        rho * data_info["density_max"],
        v * data_info["V"],
        f * data_info["density_max"] * data_info["V"],
    )


def compute_errors(true_rho, true_v, true_f, model_rho, model_v, model_f):
    rho_err = jnp.sqrt(jnp.sum((true_rho - model_rho) ** 2)) / jnp.sqrt(
        jnp.sum(true_rho**2)
    )
    v_err = jnp.sqrt(jnp.sum((true_v - model_v) ** 2)) / jnp.sqrt(jnp.sum(true_v**2))
    f_err = jnp.sqrt(jnp.sum((true_f - model_f) ** 2)) / jnp.sqrt(jnp.sum(true_f**2))
    return rho_err, v_err, f_err


def compute_tts_error(true_rho, model_rho, t_vec, x_vec):
    tts_true = np.trapz(np.trapz(true_rho, t_vec, axis=1), x_vec, axis=0)
    tts_model = np.trapz(np.trapz(model_rho, t_vec, axis=1), x_vec, axis=0)
    return np.abs((tts_model - tts_true) / tts_true)


def simulate_model(flux_fn, flux_der_fn, data_info, S, flats, step, rho_bnd_array):
    rho, v, f = godunov_solver(
        data_info["rho_0"],
        S,
        rho_bnd_array,
        flux_fn,
        flux_der_fn,
        data_info["delta_t_refined"],
        0,
        flats,
        data_info["num_t_points"],
    )
    flat_rho = C.CochainD1(S, flats["flat_left_v"](rho.T)[:, :, 0].T)
    rho_computedP0 = C.star(flat_rho).coeffs
    v_comp = v[:, ::step]
    f_comp = f[:, ::step]
    v_true = data_info["v"]
    f_true = data_info["flow"]
    v_comp = v_comp.at[:, 0].set(v_true[:, 0])
    v_comp = v_comp.at[0, :].set(v_true[0, :])
    v_comp = v_comp.at[-3:, :].set(v_true[-3:, :])

    f_comp = f_comp.at[:, 0].set(f_true[:, 0])
    f_comp = f_comp.at[0, :].set(f_true[0, :])
    f_comp = f_comp.at[-3:, :].set(f_true[-3:, :])
    return rho, rho_computedP0, v_comp, f_comp


def plot_diagrams(results, rhoP0, v, f, name_diagram, test_name, train_idx, test_idx):

    if name_diagram == "velocity":
        diagram = v
        diagram_idx = "v"
    elif name_diagram == "flux":
        diagram = f
        diagram_idx = "f"
    models_names = list(results.keys())
    num_models = len(models_names)

    fig_dim = (3 * num_models, num_models - 1.2)
    fig, axes = plt.subplots(1, num_models, figsize=fig_dim)
    for i in range(num_models):
        name = models_names[i]
        axes[i].scatter(
            results[name]["rhoP0"][1:-3, 1:].flatten(),
            results[name][diagram_idx][1:-3, 1:].flatten(),
            marker=".",
            s=5,
            label="Model",
            c="#ff0000",
            zorder=1,
        )
        axes[i].scatter(
            rhoP0[:, train_idx].flatten(),
            diagram[:, train_idx].flatten(),
            marker=".",
            s=5,
            label="Training Data",
            c="#4757fb",
            zorder=0,
        )
        axes[i].scatter(
            rhoP0[:, test_idx].flatten(),
            diagram[:, test_idx].flatten(),
            marker=".",
            s=5,
            label="Test Data",
            c="#0ea4f0",
            zorder=0,
        )
        axes[i].set_xlabel(r"$\rho$ (veh/ft)")
        axes[i].set_ylabel(r"$\rho\,V(\rho)$ (veh/s)")
        # axes[i].legend()
        axes[i].set_title(name)

    handles, labels = axes[i].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.65, 1.1),
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    plt.tight_layout()
    plt.savefig(f"{name_diagram}_{test_name}.png", dpi=300, bbox_inches="tight")
    plt.clf()


def make_rect(xy, width, height, color):
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


def rho_v_plot(
    results, data_info, v, x_sampled_circ, test_name, x_ticks, y_ticks, task
):
    models_names = list(results.keys())
    num_models = len(models_names)
    data_list = [data_info["density"], v]
    data_names = ["rho", "v"]
    cbar_names = [r"$\rho$ (veh/ft)", r"$v$ (ft/s)"]

    fig, axes = plt.subplots(2, 1 + num_models, figsize=(3 * num_models, 4.5))

    x_mesh, t_mesh = np.meshgrid(x_sampled_circ[:-3], data_info["t_sampled_circ"])

    cmap = "rainbow"
    cb_ticks = [[0, 0.1, 0.2], [1, 40, 75]]

    if task == "prediction":
        rect_0_train = make_rect((2.5, 10), 535.0, 1500.0, "red")
        rect_1_train = make_rect((2.5, 10), 535.0, 1500.0, "red")
        rect_0_test = make_rect((542.5, 10), 355, 1500.0, "#FF7F50")
        rect_1_test = make_rect((542.5, 10), 355, 1500.0, "#FF7F50")

        rect_train = [rect_0_train, rect_1_train]
        rect_test = [rect_0_test, rect_1_test]
    elif task == "reconstruction":
        x_idx = x_sampled_circ[train_idx][:-4]
        x_idx = [
            50.0,
            310.0,
            430.0,
            610.0,
            770.0,
            1050.0,
            1210.0,
            1330.0,
            1430.0,
        ]
        x_magn = [40.0, 20.0, 80, 20, 20, 40, 40, 20, 40]

    for i, data_entry in enumerate(data_list):
        vmin = np.min(data_entry.T)
        vmax = np.max(data_entry.T)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        data_plot = axes[i, 0].contourf(
            t_mesh, x_mesh, data_entry[:-3].T, levels=100, cmap=cmap, norm=norm
        )
        for j in range(num_models):
            axes[i, j + 1].contourf(
                t_mesh,
                x_mesh,
                results[models_names[j]][data_names[i]][:-3].T,
                levels=100,
                cmap=cmap,
                norm=norm,
            )
            # Title
            if i == 0:
                if task == "prediction":
                    axes[i, 0].add_patch(rect_train[i])
                    axes[i, 0].add_patch(rect_test[i])
                elif task == "reconstruction":
                    for k in range(len(x_idx)):
                        rect = make_rect((2.5, x_idx[k]), 895.0, x_magn[k], "#FF00FF")
                        axes[i, 0].add_patch(rect)

                axes[i, 0].set_title("Data")
                axes[i, j + 1].set_title(models_names[j])

        # Add one colorbar per row
        fig.colorbar(
            data_plot,
            ax=axes[i, :],
            orientation="vertical",
            fraction=0.05,
            pad=0.01,
            label=cbar_names[i],
            ticks=cb_ticks[i],
        )

    # Axis labels
    for i in range(2):
        axes[i, 0].set_ylabel("x (ft)")
    for j in range(num_models + 1):
        axes[-1, j].set_xlabel("t (s)")

    for i in range(2):
        for j in range(num_models + 1):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    # Axis ticks
    for i in range(2):
        axes[i, 0].set_yticks(y_ticks)
    for j in range(num_models + 1):
        axes[-1, j].set_xticks(x_ticks)

    # plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(f"rho_v_f_plot_{test_name}.png", dpi=300, bbox_inches="tight")
    plt.clf()


def predicted_true_plots(results, v, f, test_name):
    models_names = list(results.keys())
    num_models = len(models_names)
    fig_dim = (3 * num_models, num_models)
    _, axes = plt.subplots(1, num_models, figsize=fig_dim)
    for i in range(num_models):
        axes[i].scatter(
            f.flatten(),
            results[models_names[i]]["f"].flatten(),
            marker=".",
            s=5,
            c="#0ea4f0",
        )
        axes[i].set_aspect("equal")
        axes[i].scatter(
            f.flatten(),
            f.flatten(),
            marker=".",
            s=5,
            c="#ff0000",
        )
        axes[i].set_xlabel(r"Flux true")
        axes[i].set_ylabel(r"Flux predicted")
        axes[i].set_title(models_names[i])
        # axes[i].set_xlim(0, 25)
        # axes[i].set_ylim(0, 25)
    plt.tight_layout()
    plt.savefig(f"pred_actual_flux_{test_name}.png", dpi=300, bbox_inches="tight")
    plt.clf()

    fig_dim = (3 * num_models, num_models)
    _, axes = plt.subplots(1, num_models, figsize=fig_dim)
    for i in range(num_models):
        axes[i].scatter(
            v.flatten(),
            results[models_names[i]]["v"].flatten(),
            marker=".",
            s=5,
            c="#0ea4f0",
        )
        axes[i].set_aspect("equal")
        axes[i].scatter(
            v.flatten(),
            v.flatten(),
            marker=".",
            s=5,
            c="#ff0000",
        )
        axes[i].set_xlabel(r"Velocity true")
        axes[i].set_ylabel(r"Velocity predicted")
        axes[i].set_title(models_names[i])
        # axes[i].set_xlim(0, 25)
        # axes[i].set_ylim(0, 25)
    plt.tight_layout()
    plt.savefig(f"pred_actual_velocity_{test_name}.png", dpi=300, bbox_inches="tight")
    plt.clf()


# Highlight best values
def format_entry(val, rank, is_best):
    formatted = f"{val:.3f} ({rank})"
    return f"\\textbf{{{formatted}}}" if is_best else formatted


def fill_error_table(results, train_idx, test_idx, task):
    t_errors = []
    if task == "prediction":
        train_idx_slice = (slice(None), train_idx)
        test_idx_slice = (slice(None), test_idx)
    elif task == "reconstruction":
        train_idx_slice = (train_idx, slice(None))
        test_idx_slice = (test_idx, slice(None))
    for name, model in results.items():
        # training errors
        e_rho_train, e_v_train, _ = compute_errors(
            data_info["density"][train_idx_slice],
            v[train_idx_slice],
            f[train_idx_slice],
            model["rho"][train_idx_slice],
            model["v"][train_idx_slice],
            model["f"][train_idx_slice],
        )
        # test errors
        e_rho_test, e_v_test, _ = compute_errors(
            data_info["density"][test_idx_slice],
            v[test_idx_slice],
            f[test_idx_slice],
            model["rho"][test_idx_slice],
            model["v"][test_idx_slice],
            model["f"][test_idx_slice],
        )

        # e_tts = compute_tts_error(
        #     data_info["density"],
        #     model["rho"],
        #     data_info["t_sampled_circ"],
        #     x_sampled_circ,
        # )
        t_errors.append((name, e_rho_train, e_v_train, e_rho_test, e_v_test))

    # Compute ranks for each metric
    error_arrays = np.array([[t[1], t[2], t[3], t[4]] for t in t_errors])
    ranks = np.argsort(np.argsort(error_arrays, axis=0), axis=0) + 1
    avg_ranks = np.mean(ranks, axis=1)

    latex_rows = []
    for i, (name, e_rho_train, e_v_train, e_rho_test, e_v_test) in enumerate(t_errors):
        row = [
            name,
            format_entry(e_rho_train, ranks[i, 0], ranks[i, 0] == 1),
            format_entry(e_v_train, ranks[i, 1], ranks[i, 1] == 1),
            format_entry(e_rho_test, ranks[i, 2], ranks[i, 2] == 1),
            format_entry(e_v_test, ranks[i, 3], ranks[i, 3] == 1),
            (
                f"\\textbf{{{avg_ranks[i]:.2f}}}"
                if avg_ranks[i] == min(avg_ranks)
                else f"{avg_ranks[i]:.2f}"
            ),
        ]
        latex_rows.append(row)

    table = (
        r"""\begin{table}[H]
        \caption{Relative errors between the actual and the computed density and velocity (training and test) for the prediction task. In bold, the best-performing models for each metric considered.}
        \begin{center}
            \begin{tabular}{c c c c c c}
                \toprule
                Model & $E^{\text{tr}}_\rho$ & $E^{\text{tr}}_v$ & $E^{\text{ts}}_\rho$ & $E^{\text{ts}}_v$ & Avg Rank\\
                \midrule
    """
        + "\n".join(["            " + " & ".join(row) + r"\\" for row in latex_rows])
        + r"""
                \bottomrule
            \end{tabular}
        \end{center}
        \label{tab:errors_i80_pred}
    \end{table}"""
    )

    print(table)


road_name = "US80"
task = "reconstruction"
test_name = f"i80_{task}"
data_info = preprocess_data(road_name)
X_training, X_test = build_dataset(
    data_info["t_sampled_circ"],
    data_info["S"],
    data_info["density"],
    data_info["v"],
    data_info["flow"],
    task,
)

# plot data
x_sampled_circ = (data_info["x_sampled"][1:] + data_info["x_sampled"][:-1]) / 2
x_mesh, t_mesh = np.meshgrid(x_sampled_circ, data_info["t_sampled_circ"])

plot_velocity_flux_density(
    t_mesh, x_mesh, data_info["v"].T, data_info["flow"].T, data_info["density"].T
)
plt.clf()

S = data_info["S"]

# define flat
dual_edges = C.CochainD1(S, S.dual_edges_vectors)
zeros = C.CochainD0(S, jnp.zeros_like(data_info["density"][:, 0]))
I_linear_left = tf_flat.get_linear_left_interpolation
flat_linear_left_D = partial(
    flat,
    weights=None,
    edges=dual_edges,
    interp_func=I_linear_left,
    interp_func_args={"sigma": zeros},
)


def flat_left_wrap(x):
    return flat_linear_left_D(C.CochainD0(S, x)).coeffs


flat_left = vmap(flat_left_wrap)

zeros_P = C.CochainP0(S, jnp.zeros_like(data_info["vP0"][:, 0]))
zeros_D = C.CochainD0(S, jnp.zeros_like(data_info["density"][:, 0]))

all_flats = tf_flat.define_flats(S, zeros_P, zeros_D)

flats = {
    "linear_left": all_flats["flat_linear_left_D"],
    "linear_left_P": all_flats["flat_linear_left_P"],
    "linear_right": all_flats["flat_linear_right_D"],
    "linear_right_P": all_flats["flat_linear_right_P"],
    "flat_left_v": flat_left,
}

# build the kernel
d = 3 * data_info["delta_x"]
n_d = int(d / data_info["delta_x"])
y_kernel = np.linspace(0.0, d, num=n_d)
eta = linear_kernel(S, y_kernel, d)


if road_name == "US101":
    opt_greenshields = [0.79197062, 0.51105397]
    opt_Weidmann = [0.5893936, 0.54700764, 0.541796]
    opt_triangular = [0.42595455, 1.40987626, 7.19902727]
    opt_idm = [0.13046561, 0.68253887, 0.05752636, 0.49775953]
    opt_del_castillo = [0.21205058, 0.55683342, 0.82425097, 6.70846888]

    sr_greens_params = [1.06722615490511074654, 0.96463871990166261128]
    sr_Weidmann_params = [1.83887498367116108966, 1.33264073296325769036]
    sr_triangular_params = [3.59013933560497733311, -0.26024859915084519457]
    sr_idm_params = [2.26415104806225109257, 1.20329683345590154886]
    sr_del_castillo_params = [2.58099743784204349595, -0.07370838785888800260]

    x_ticks = [0, 675, 1350, 2025, 2700]
    y_ticks = [50, 505, 1010, 1515, 2070]
elif road_name == "US80":
    if task == "prediction":
        opt_greenshields = [0.54673127, 0.55995123]
        opt_Weidmann = [0.63190729, 0.80612097, 0.24947817]
        opt_triangular = [0.37013956, 1.48964708, 6.59672108]
        opt_idm = [0.43936351, 0.93094344, 0.16251414, 0.61353022]
        opt_del_castillo = [0.31807369, 0.46732741, 0.61532169, 2.60100492]
        sr_greens_params = [2.86021508985409056436, -5.64427014751596534126]
        sr_Weidmann_params = [2.12466407856669370346, -4.18665767293357760082]
        sr_triangular_params = [2.72042969, -3.49581679932267430644]
        sr_idm_params = [2.56095629955760983876, -0.66842648814023597481]
        sr_del_castillo_params = [2.17247255542438466591, -0.09043064382541565749]

    elif task == "reconstruction":
        opt_greenshields = [0.67221695, 0.53916011]
        opt_Weidmann = [0.58670242, 0.71605332, 0.32424757]
        opt_triangular = [0.37468432, 1.28975743, 7.48885539]
        sr_greens_params = [5.83882956394043795001]
        sr_Weidmann_params = [5.82940218613048344309]
        sr_triangular_params = [9.39984985913767445709]

    x_ticks = [0, 450, 900]
    y_ticks = [10, 760, 1510]

num_x_points = len(x_sampled_circ)
# set-up boundary conditions in an array
rho_bnd_array = jnp.zeros((len(data_info["rho_bnd"].keys()), data_info["num_t_points"]))
for index in data_info["rho_bnd"].keys():
    rho_bnd_array = rho_bnd_array.at[int(index), :].set(
        data_info["rho_bnd"][index][: data_info["num_t_points"]]
    )

flux_greens = lambda x: tf_utils.Greenshields_flux(x, *opt_greenshields)
flux_weidmann = lambda x: tf_utils.Weidmann_flux(x, *opt_Weidmann)
flux_triang = lambda x: tf_utils.triangular_flux(x, *opt_triangular)
flux_idm = lambda x: tf_utils.IDM_flux(x, *opt_idm)
flux_del_castillo = lambda x: tf_utils.del_castillo_flux(x, *opt_del_castillo)


Greenshields_der = tf_utils.define_flux_der(S, tf_utils.Greenshields_flux)
Weidmann_der = tf_utils.define_flux_der(S, tf_utils.Weidmann_flux)
triangular_flux_der = tf_utils.define_flux_der(S, tf_utils.triangular_flux)
IDM_flux_der = tf_utils.define_flux_der(S, tf_utils.IDM_flux)
del_castillo_flux_der = tf_utils.define_flux_der(S, tf_utils.del_castillo_flux)

flux_greens_der = lambda x: Greenshields_der(x, *opt_greenshields)
flux_weidmann_der = lambda x: Weidmann_der(x, *opt_Weidmann)
flux_triang_der = lambda x: triangular_flux_der(x, *opt_triangular)
flux_idm_der = lambda x: IDM_flux_der(x, *opt_idm)
flux_del_castillo_der = lambda x: del_castillo_flux_der(x, *opt_del_castillo)


# define baseline models
step = data_info["step"]
models = {
    "Greenshields": (flux_greens, flux_greens_der, sr_greens_params),
    # "IDM": (flux_idm, flux_idm_der, sr_idm_params),
    "Weidmann": (flux_weidmann, flux_weidmann_der, sr_Weidmann_params),
    "Triangular": (flux_triang, flux_triang_der, sr_triangular_params),
    # "Del Castillo": (flux_del_castillo, flux_del_castillo_der, sr_del_castillo_params),
}

# define sr models
sr_models = {}
for name, (flux_fn, _, params) in models.items():
    sr_name = "SR-" + name
    sr_flux = sr_flux_generator(flux_fn, flats, params, task)
    sr_flux_der = tf_utils.define_flux_der(S, sr_flux)
    sr_models[sr_name] = (sr_flux, sr_flux_der, params)


models = models | sr_models
# solve LWR model for all the fundamental diagrams
results = {}
for name, (flux_fn, flux_der_fn, _) in models.items():
    rho, rhoP0_model, v_model, f_model = simulate_model(
        flux_fn, flux_der_fn, data_info, S, flats, step, rho_bnd_array
    )
    rhoP0_model, rho, v_model, f_model = rescale_rho_v_f(
        rhoP0_model, rho, v_model, f_model, data_info
    )
    results[name] = dict(
        rho=rho[:, ::step], rhoP0=rhoP0_model[:-1, ::step], v=v_model, f=f_model
    )

sr_results = {}
for name, (flux_fn, flux_der_fn, _) in sr_models.items():
    rho, rhoP0_model, v_model, f_model = simulate_model(
        flux_fn, flux_der_fn, data_info, S, flats, step, rho_bnd_array
    )
    rhoP0_model, rho, v_model, f_model = rescale_rho_v_f(
        rhoP0_model, rho, v_model, f_model, data_info
    )
    sr_results[name] = dict(
        rho=rho[:, ::step], rhoP0=rhoP0_model[:-1, ::step], v=v_model, f=f_model
    )


# Compute true Cochains
flat_density = C.CochainD1(S, flat_left(data_info["density"].T)[:, :, 0].T)
rhoP0 = C.star(flat_density).coeffs[:-1]
flat_v = C.CochainD1(S, flat_left(data_info["v"].T)[:, :, 0].T)
vP0 = C.star(flat_v).coeffs
v = vP0[:-1]
flat_f = flat_left(data_info["flow"].T)[:, :, 0].T
fP0 = C.star(C.CochainD1(S, flat_f))
f = fP0.coeffs[:-1]

# Rescale true data
rhoP0, data_info["density"], v, f = rescale_rho_v_f(
    rhoP0, data_info["density"], v, f, data_info
)


if task == "prediction":
    train_idx = jnp.arange(X_training[0, 0], X_training[-1, 0] + 1, dtype=jnp.int64)
    test_idx = jnp.arange(X_test[0, 0], X_test[-1, 0] + 1, dtype=jnp.int64)
elif task == "reconstruction":
    num_tr = int(X_training.shape[0] / len(data_info["t_sampled_circ"]))
    num_test = int(X_test.shape[0] / len(data_info["t_sampled_circ"]))
    train_idx = X_training[:num_tr, 0].astype(np.int64) + 1
    train_idx = np.concatenate((train_idx, [0, 76, 77, 78]))
    test_idx = X_test[:num_test, 0].astype(np.int64) + 1


x_sampled_circ *= data_info["L_dim"]
data_info["t_sampled_circ"] *= data_info["t_len"]


# plot params
width = 443.57848
fontsize = 12
plt.rcParams["font.size"] = fontsize
plt.rcParams["font.sans-serif"] = "Dejavu Sans"
plt.rcParams["font.family"] = "sans-serif"

plot_diagrams(results, rhoP0, v, f, "flux", test_name, train_idx, test_idx)
plot_diagrams(results, rhoP0, v, f, "velocity", test_name, train_idx, test_idx)

plot_diagrams(sr_results, rhoP0, v, f, "flux", test_name + "_sr", train_idx, test_idx)
plot_diagrams(
    sr_results, rhoP0, v, f, "velocity", test_name + "_sr", train_idx, test_idx
)

rho_v_plot(results, data_info, v, x_sampled_circ, test_name, x_ticks, y_ticks, task)
rho_v_plot(
    sr_results, data_info, v, x_sampled_circ, test_name + "_sr", x_ticks, y_ticks, task
)


# predicted-true plots
predicted_true_plots(results, v, f, test_name)
predicted_true_plots(sr_results, v, f, test_name + "_sr")


# Table computation
fill_error_table(results, train_idx, test_idx, task)
# fill_error_table(sr_results)
