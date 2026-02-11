import matplotlib.pyplot as plt
import numpy as np
from jax import vmap, jacfwd
from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from flex.gp.regressor import GPSymbolicRegressor
from deap import gp
import os
import jax.numpy as jnp
import matplotlib.colors as mcolors
import importlib
from typing import Callable, Dict, Tuple, List
from functools import partial
from sr_traffic.utils.godunov import body_fun, main_loop
import numpy.typing as npt


def resolve_function(full_name: str):
    # helper to resolve function from string
    module = importlib.import_module("sr_traffic.fund_diagrams.fund_diagrams_def")
    return getattr(module, full_name)


def detect_nested_functions(equation: str):
    # list of special functions
    conv_functions = ["conv"]
    # flag to indicate if nested functions are found
    nested = 0
    # track depth within special function calls
    function_depth = 0
    i = 0

    while i < len(equation) and not nested:
        # Look for special function
        trig_found = any(
            equation[i : i + len(trig)].lower() == trig for trig in conv_functions
        )
        if trig_found:
            # If a trig function is found, look for its opening parenthesis
            j = i
            while j < len(equation) and equation[j] not in ["(", " "]:
                j += 1
            if j < len(equation) and equation[j] == "(":
                if function_depth > 0:
                    # We are already inside a trig function, this is a nested trig
                    # function
                    nested = 1
                function_depth += 1
                i = j  # Move i to the position of '('
        elif equation[i] == "(" and function_depth > 0:
            # Increase depth if we're already in a trig function
            function_depth += 1
        elif equation[i] == ")":
            if function_depth > 0:
                # Leaving a trigonometric function or nested parentheses
                function_depth -= 1
        i += 1

    return nested


def compute_error_rho_v_f(
    rho: npt.NDArray,
    v: npt.NDArray,
    rho_norm: float,
    v_norm: float,
    rho_0: npt.NDArray,
    num_t_points: int,
    t_idx: npt.NDArray,
    step: int,
    single_iteration: Callable,
    flux: Callable,
    flat_lin_left: Callable,
    S: SimplicialComplex,
    task: str,
):
    rho_v_f = main_loop(rho_0, single_iteration, num_t_points)

    # extract rho, v and f
    rho_1_T = rho_v_f[0][:, :, 0]
    v_1_T = rho_v_f[1][:, :, 0]
    f_1_T = rho_v_f[2][:, :, 0]

    # first interpolate rho_0, then compute velocity
    rho_0_P0 = C.star(flat_lin_left(C.CochainD0(S, rho_0)))
    f_0 = flux(rho_0_P0).coeffs
    v_0 = f_0 / rho_0_P0.coeffs

    # insert initial values of rho and v
    rho_computed = jnp.vstack([rho_0, rho_1_T]).T
    v_computed = jnp.vstack([v_0[:-1].ravel("F"), v_1_T]).T
    f_computed = jnp.vstack([f_0[:-1].ravel("F"), f_1_T]).T

    # compute total error on the interior of the domain
    if task == "prediction":
        total_rho_error = (
            100
            * jnp.sum((rho_computed[1:-3, t_idx * step].ravel("F") - rho) ** 2)
            / rho_norm
        )
        total_v_error = (
            100 * jnp.sum((v_computed[1:-3, t_idx * step].ravel("F") - v) ** 2) / v_norm
        )
    elif task == "reconstruction":
        total_rho_error = (
            100
            * jnp.sum((rho_computed[1:-3, ::step][t_idx].ravel("F") - rho) ** 2)
            / rho_norm
        )
        total_v_error = (
            100
            * jnp.sum((v_computed[1:-3, ::step][t_idx].ravel("F") - v) ** 2)
            / v_norm
        )
    total_error = 0.5 * (total_rho_error + total_v_error)

    return total_error, rho_computed, v_computed, f_computed


def init_prb(
    individual,
    rho_bnd: Dict,
    S: SimplicialComplex,
    num_t_points: int,
    delta_t: float,
    flats: Dict,
    ansatz: Dict,
):
    # set-up flux
    def flux(x):
        return C.cochain_mul(ansatz["flux"](x, *ansatz["opt_coeffs"]), individual(x))

    # set-up boundary conditions in an array
    rho_bnd_array = jnp.zeros((len(rho_bnd.keys()), num_t_points))
    for index in rho_bnd.keys():
        rho_bnd_array = rho_bnd_array.at[int(index), :].set(
            rho_bnd[index][:num_t_points]
        )

    def flux_array(x):
        return flux(C.CochainP0(S, x)).coeffs.flatten()

    flux_jac = jacfwd(flux_array)

    def flux_der_array(x):
        return jnp.diag(flux_jac(x.flatten()))

    flux_der = partial(flux_der_wrap, flux_der=flux_der_array, S=S)

    single_iteration = partial(
        body_fun, S, rho_bnd_array, flux, flux_der, delta_t, 0.0, flats
    )

    return flux, single_iteration


def solve(
    func: Callable,
    consts: List,
    X: npt.NDArray,
    rho_bnd: npt.NDArray,
    rho_0: npt.NDArray,
    S: SimplicialComplex,
    num_t_points: npt.NDArray,
    delta_t: float,
    step: float,
    flats: Dict,
    ansatz: Dict,
    task: str,
) -> Tuple[float, npt.NDArray]:

    if task == "prediction":
        idx = jnp.arange(X[0, 0], X[-1, 0] + 1, dtype=jnp.int64)
    elif task == "reconstruction":
        num_original_t_points = int((num_t_points - 1) / step) + 1
        num_x = int(X.shape[0] / num_original_t_points)
        idx = X[:num_x, 0].astype(jnp.int64)

    rho = X[:, 1]
    v = X[:, 2]
    # f = t_rho_v_f.y["f"]

    def individual(x):
        return func(x, consts)

    rho_norm = jnp.sum(rho**2)
    v_norm = jnp.sum(v**2)
    # f_norm = jnp.sum(f**2)

    # num_t_points = int(t_idx[-1] * step + 1)

    # init rho and define flux and update_rho fncs
    flux, single_iteration = init_prb(
        individual,
        rho_bnd,
        S,
        num_t_points,
        delta_t,
        flats,
        ansatz,
    )

    total_error, rho_comp, v_comp, f_comp = compute_error_rho_v_f(
        rho,
        v,
        rho_norm,
        v_norm,
        rho_0,
        num_t_points,
        idx,
        step,
        single_iteration,
        flux,
        flats["linear_left"],
        S,
        task,
    )

    return total_error, {"rho": rho_comp, "v": v_comp, "f": f_comp}


class fitting_problem:
    def __init__(self, general_fitness: Callable, n_constants: int):
        self.general_fitness = general_fitness
        self.n_constants = n_constants

    def fitness(self, x):
        return [self.general_fitness(x)]

    def get_bounds(self):
        return (-10.0 * jnp.ones(self.n_constants), 10.0 * jnp.ones(self.n_constants))


def flux_wrap(x: C.CochainP0, func: Callable, S: SimplicialComplex) -> C.CochainP0:
    return C.CochainP0(S, x.coeffs * func(x).coeffs)


def flux_der_wrap(
    x: C.CochainP0, flux_der: Callable, S: SimplicialComplex
) -> C.CochainP0:
    return C.CochainP0(S, flux_der(x.coeffs))


@partial(vmap, in_axes=(0, None, None, None, None))
def compute_v_rho_der(
    rho_val: npt.NDArray,
    func: Callable,
    S: SimplicialComplex,
    v_fun: Callable,
    opt_coeffs: Dict,
):
    # set-up derivative of velocity
    def v_array(x):
        v_ansatz = v_fun(x, *opt_coeffs)
        v = v_ansatz * func(C.CochainP0(S, x)).coeffs.flatten()
        return v

    v_jac = jacfwd(v_array)

    def v_der_array(x):
        return jnp.diag(v_jac(x.flatten()))

    rho = C.CochainP0(S, rho_val * jnp.ones(S.num_nodes))
    v_rho_der = v_der_array(rho.coeffs)
    return v_rho_der[1]


def is_v_unfeasible(
    individual: Callable,
    rho: npt.NDArray,
    S: SimplicialComplex,
    v: npt.NDArray,
    opt_coeffs: Dict,
):
    # compute derivative of velocity on rho
    v_rho_der_in = compute_v_rho_der(rho.T, individual, S, v, opt_coeffs)

    # check that v'(rho) <= 0
    v_der_check = jnp.sum(v_rho_der_in > 1e-12)

    # filter nan
    v_der_cond = jnp.nan_to_num(v_der_check, nan=1e6)

    return v_der_cond > 0


def assign_consts(individuals: List[gp.PrimitiveTree], attributes: Dict):
    for ind, attr in zip(individuals, attributes):
        ind.consts = attr["consts"]
        ind.fitness.values = attr["fitness"]


def custom_logger(best_individuals: List[gp.PrimitiveTree]):
    for ind in best_individuals:
        print(f"The constants of the best individual are: {ind.consts}", flush=True)


def stgp_traffic_plots(
    gpsr: GPSymbolicRegressor,
    S: SimplicialComplex,
    flats: Dict,
    test_data: npt.NDArray,
    density: npt.NDArray,
    v: npt.NDArray,
    f: npt.NDArray,
    t_sampled_circ: npt.NDArray,
    step: int,
    output_path: str,
):
    os.chdir(output_path)

    x_sampled = S.node_coords
    x_sampled_circ = (x_sampled[1:] + x_sampled[:-1]) / 2

    def flat_left_wrap(x):
        return flats["linear_left"](C.CochainD0(S, x)).coeffs

    flat_left = vmap(flat_left_wrap)

    sols = gpsr.predict(test_data)

    rho_comp = sols["rho"][:, ::step]
    v_comp = sols["v"][:, ::step]
    f_comp = sols["f"][:, ::step]

    # initial and boundary values of v_comp and
    # f_comp are known (ic+ bc on rho)
    v_comp = v_comp.at[:, 0].set(v[:, 0])
    v_comp = v_comp.at[0, :].set(v[0, :])
    v_comp = v_comp.at[-3:, :].set(v[-3:, :])

    f_comp = f_comp.at[:, 0].set(f[:, 0])
    f_comp = f_comp.at[0, :].set(f[0, :])
    f_comp = f_comp.at[-3:, :].set(f[-3:, :])

    flat_rho = C.CochainD1(S, flat_left(rho_comp.T)[:, :, 0].T)
    rho_computed = C.star(flat_rho).coeffs[:-1]

    flat_density = C.CochainD1(S, flat_left(density.T)[:, :, 0].T)
    density_data = C.star(flat_density).coeffs[:-1]

    # errors
    rho_error = jnp.sqrt(jnp.sum(jnp.square(density - rho_comp))) / jnp.sqrt(
        jnp.sum(jnp.square(density))
    )
    v_error = jnp.sqrt(jnp.sum(jnp.square(v - v_comp))) / jnp.sqrt(
        jnp.sum(jnp.square(v))
    )
    f_error = jnp.sqrt(jnp.sum(jnp.square(f - f_comp))) / jnp.sqrt(
        jnp.sum(jnp.square(f))
    )

    # tts
    tts_data = np.trapezoid(
        np.trapezoid(density, t_sampled_circ, axis=1),
        x_sampled_circ.flatten(),
        axis=0,
    )
    tts = np.trapezoid(
        np.trapezoid(rho_comp, t_sampled_circ, axis=1),
        x_sampled_circ.flatten(),
        axis=0,
    )

    error_tts = np.abs((tts - tts_data) / tts_data)

    print(rho_error, v_error, f_error, error_tts)

    plt.scatter(
        density_data[1:-3, 1:].flatten(),
        f[1:-3, 1:].flatten(),
        marker=".",
        c="#4757fb",
        label="Data",
    )
    plt.scatter(
        rho_computed[1:-3, 1:].flatten(),
        f_comp[1:-3, 1:].flatten(),
        marker=".",
        c="#ff0000",
        label="SR-Traffic",
    )
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$\rho V(\rho)$")
    plt.legend()
    plt.savefig("flux.png", dpi=300)

    plt.clf()

    plt.figure(1, figsize=(20, 10))

    _, axes = plt.subplots(3, 2, num=1)

    x_mesh, t_mesh = np.meshgrid(x_sampled_circ[1:-3], t_sampled_circ)

    vmin = np.min(density.T)
    vmax = np.max(density.T)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = "rainbow"

    # plot rho
    rho_plot = axes[0, 0].contourf(
        t_mesh, x_mesh, density[1:-3].T, levels=100, cmap=cmap, norm=norm
    )
    axes[0, 0].set_title("Data")
    rho_computed_plot = axes[0, 1].contourf(
        t_mesh, x_mesh, rho_comp[1:-3].T, levels=100, cmap=cmap, norm=norm
    )
    axes[0, 1].set_title("SR-Traffic")
    plt.colorbar(
        rho_plot, label=r"$\rho$", ax=axes[0, 1], ticks=[0, 0.2, 0.4, 0.6, 0.8, 1]
    )
    plt.colorbar(
        rho_plot, label=r"$\rho$", ax=axes[0, 0], ticks=[0, 0.2, 0.4, 0.6, 0.8, 1]
    )

    vmin = np.min(v.T)
    vmax = np.max(v.T)

    # v_comp = v_comp.at[v_comp > vmax].set(vmax)
    # v_comp = v_comp.at[v_comp < vmin].set(vmin)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # plot v
    v_plot = axes[1, 0].contourf(
        t_mesh, x_mesh, v[1:-3].T, levels=100, cmap=cmap, norm=norm
    )
    v_computed_plot = axes[1, 1].contourf(
        t_mesh, x_mesh, v_comp[1:-3].T, levels=100, cmap=cmap, norm=norm
    )
    plt.colorbar(v_plot, ax=axes[1, 1], label=r"$v$")
    plt.colorbar(v_plot, ax=axes[1, 0], label=r"$v$")

    # plot f

    vmin = np.min(f.T)
    vmax = np.max(f.T)

    # f_comp = f_comp.at[f_comp > vmax].set(vmax)
    # f_comp = f_comp.at[f_comp < vmin].set(vmin)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    f_plot = axes[2, 0].contourf(
        t_mesh, x_mesh, f[1:-3].T, levels=100, cmap=cmap, norm=norm
    )
    f_computed_plot = axes[2, 1].contourf(
        t_mesh, x_mesh, f_comp[1:-3].T, levels=100, cmap=cmap, norm=norm
    )
    plt.colorbar(f_plot, ax=axes[2, 1], label=r"$f$")
    plt.colorbar(f_plot, ax=axes[2, 0], label=r"$f$")

    for i in range(3):
        axes[i, 0].set_xlabel(r"t")
        axes[i, 1].set_xlabel(r"t")
        axes[i, 0].set_ylabel(r"x")
        axes[i, 1].set_ylabel(r"x")
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.savefig("plots.png", dpi=300)
