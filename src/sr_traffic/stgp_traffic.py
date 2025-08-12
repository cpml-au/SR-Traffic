from dctkit import config as config
from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from typing import Tuple, Callable, Dict, List
import numpy.typing as npt
from jax import jit, vmap, jacfwd
import jax.numpy as jnp
from functools import partial
import sr_traffic.utils.flat as tf_flat
import sr_traffic.utils.fund_diagrams as fnd_diag
from sr_traffic.utils.primitives import add_new_primitives
from sr_traffic.utils.godunov import body_fun, main_loop
from sr_traffic.data.data import preprocess_data, build_dataset
from flex.gp import util, primitives
from flex.gp.regressor import GPSymbolicRegressor
from deap import gp
from deap.base import Toolbox
import warnings
import pygmo as pg
from sr_traffic.compute_results import plots
import os
import time
import gc
import importlib

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF warnings (JAX uses XLA)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hides all GPUs from JAX
os.environ["JAX_LOG_COMPILES"] = "0"

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()


class fitting_problem:
    def __init__(self, general_fitness, n_constants):
        self.general_fitness = general_fitness
        self.n_constants = n_constants

    def fitness(self, x):
        return [self.general_fitness(x)]

    def get_bounds(self):
        return (-10.0 * jnp.ones(self.n_constants), 10.0 * jnp.ones(self.n_constants))


# Helper to resolve function from string
def resolve_function(full_name):
    module = importlib.import_module("traffic_flow.utils.fund_diagrams")
    return getattr(module, full_name)


def detect_nested_functions(equation):
    # FIXME: move this
    # List of trigonometric functions
    conv_functions = ["conv"]
    nested = 0  # Flag to indicate if nested functions are found
    function_depth = 0  # Track depth within trigonometric function calls
    i = 0

    while i < len(equation) and not nested:
        # Look for trigonometric function
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


def flux_wrap(x: C.CochainP0, func: Callable, S: SimplicialComplex) -> C.CochainP0:
    return C.CochainP0(S, x.coeffs * func(x).coeffs)


def flux_der_wrap(
    x: C.CochainP0, flux_der: Callable, S: SimplicialComplex
) -> C.CochainP0:
    return C.CochainP0(S, flux_der(x.coeffs))


def compute_error_rho_v_f(
    rho,
    v,
    rho_norm,
    v_norm,
    rho_0,
    num_t_points,
    t_idx,
    step,
    single_iteration,
    flux,
    flat_lin_left,
    S,
    task,
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


@partial(vmap, in_axes=(0, None, None, None, None))
def compute_v_rho_der(rho_val, func, S, v_fun, opt_coeffs):
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


def is_v_unfeasible(individual, rho, S, v, opt_coeffs):
    # compute derivative of velocity on rho
    v_rho_der_in = compute_v_rho_der(rho.T, individual, S, v, opt_coeffs)

    # check that v'(rho) <= 0
    v_der_check = jnp.sum(v_rho_der_in > 1e-12)

    # filter nan
    v_der_cond = jnp.nan_to_num(v_der_check, nan=1e6)

    return v_der_cond > 0


def solve(
    func: Callable,
    consts: list,
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


def eval_MSE_sol(
    func: Callable,
    consts: list,
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
    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    total_err, rho_v_dict = solve(
        func,
        consts,
        X,
        rho_bnd,
        rho_0,
        S,
        num_t_points,
        delta_t,
        step,
        flats,
        ansatz,
        task,
    )
    tol = 1e2

    if jnp.isnan(total_err) or total_err > tol:
        total_err = tol

    return total_err, rho_v_dict


def eval_MSE_and_tune_constants(
    tree,
    toolbox,
    X: npt.NDArray,
    rho_bnd: npt.NDArray,
    rho_0: npt.NDArray,
    S: SimplicialComplex,
    num_t_points: npt.NDArray,
    delta_t: float,
    step: float,
    flats: Dict,
    rho_test: npt.NDArray,
    ansatz: Dict,
    task: str,
    v_check_fn,
):
    warnings.filterwarnings("ignore")
    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    individual, n_constants = util.compile_individual_with_consts(tree, toolbox)
    if task == "prediction":
        t_idx = jnp.arange(X[0, 0], X[-1, 0] + 1, dtype=jnp.int64)
        num_t_points_X = int(t_idx[-1] * step + 1)
    elif task == "reconstruction":
        num_t_points_X = num_t_points

    def eval_err(consts):
        error, _ = solve(
            individual,
            consts,
            X,
            rho_bnd,
            rho_0,
            S,
            num_t_points_X,
            delta_t,
            step,
            flats,
            ansatz,
            task,
        )
        return error

    # in this case we use an evolutionary algorithm to optimize
    algo = pg.algorithm(pg.sea(gen=10))

    pop_size = 10
    threshold = 1e2

    objective = jit(eval_err)

    def general_fitness(x):
        def ind_consts(t):
            return individual(t, x)

        # check feasibility of the solution
        v_unf = v_check_fn(ind_consts, rho_test, S, ansatz["v"], ansatz["opt_coeffs"])
        if v_unf:
            return threshold

        total_err = objective(x)
        return total_err

    if n_constants > 0:

        prb = pg.problem(fitting_problem(general_fitness, n_constants))

        pop = pg.population(prb, size=pop_size)
        pop = algo.evolve(pop)
        best_fit = pop.champion_f[0]
        best_consts = pop.champion_x
    else:
        best_consts = []
        best_fit = general_fitness(best_consts)

    if jnp.isnan(best_fit) or best_fit > threshold:
        best_fit = threshold

    return best_fit, best_consts


def eval_MSE(
    individuals_batch: list[gp.PrimitiveSet],
    toolbox: Toolbox,
    X: npt.NDArray,
    rho_bnd: npt.NDArray,
    rho_0: npt.NDArray,
    S: SimplicialComplex,
    num_t_points: npt.NDArray,
    delta_t: float,
    step: float,
    flats: Dict,
    rho_test: npt.NDArray,
    ansatz: Dict,
    task: str,
    penalty: dict,
    v_check_fn,
) -> float:

    objvals = [None] * len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        callable, _ = util.compile_individual_with_consts(individual, toolbox)
        objvals[i], _ = eval_MSE_sol(
            callable,
            individual.consts,
            X,
            rho_bnd,
            rho_0,
            S,
            num_t_points,
            delta_t,
            step,
            flats,
            ansatz,
            task,
        )

    return objvals


def predict(
    individuals_batch: list[gp.PrimitiveSet],
    toolbox: Toolbox,
    X: npt.NDArray,
    rho_bnd: npt.NDArray,
    rho_0: npt.NDArray,
    S: SimplicialComplex,
    num_t_points: npt.NDArray,
    delta_t: float,
    step: float,
    flats: Dict,
    rho_test: npt.NDArray,
    ansatz: Dict,
    task: str,
    penalty: dict,
    v_check_fn,
) -> npt.NDArray:

    best_sols = [None] * len(individuals_batch)
    for i, individual in enumerate(individuals_batch):
        callable, _ = util.compile_individual_with_consts(individual, toolbox)
        _, best_sols[i] = eval_MSE_sol(
            callable,
            individual.consts,
            X,
            rho_bnd,
            rho_0,
            S,
            num_t_points,
            delta_t,
            step,
            flats,
            ansatz,
            task,
        )
    return best_sols


def fitness(
    individuals_batch: list[gp.PrimitiveSet],
    toolbox: Toolbox,
    X: npt.NDArray,
    rho_bnd: npt.NDArray,
    rho_0: npt.NDArray,
    S: SimplicialComplex,
    num_t_points: npt.NDArray,
    delta_t: float,
    step: float,
    flats: Dict,
    rho_test: npt.NDArray,
    ansatz: Dict,
    task: str,
    penalty: dict,
    v_check_fn,
) -> Tuple[float,]:

    attributes = [] * len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        if detect_nested_functions(str(individual)) or len(individual) > 50:
            MSE = 100.0
            consts = []
        else:
            MSE, consts = eval_MSE_and_tune_constants(
                individual,
                toolbox,
                X,
                rho_bnd,
                rho_0,
                S,
                num_t_points,
                delta_t,
                step,
                flats,
                rho_test,
                ansatz,
                task,
                v_check_fn,
            )

        fitness = (MSE + penalty["reg_param"] * len(individual),)
        attributes.append({"consts": consts, "fitness": fitness})

    gc.collect()
    return attributes


def assign_consts(individuals, attributes):
    for ind, attr in zip(individuals, attributes):
        ind.consts = attr["consts"]
        ind.fitness.values = attr["fitness"]


def custom_logger(best_individuals):
    for ind in best_individuals:
        print(f"The constants of the best individual are: {ind.consts}", flush=True)


def set_prb(
    regressor_params,
    config_file_data,
    S,
    delta_x,
    delta_t,
    num_t_points,
    step,
    rho_0,
    rho_bnd,
    seed=None,
    output_path="./",
):

    penalty = config_file_data["gp"]["penalty"]
    ansatz = config_file_data["gp"]["ansatz"]
    task = config_file_data["gp"]["task"]
    ansatz["flux"] = resolve_function(ansatz["flux"])
    ansatz["v"] = resolve_function(ansatz["v"])

    zeros_P = C.CochainP0(S, jnp.zeros_like(data_info["vP0"][:, 0]))
    zeros_D = C.CochainD0(S, jnp.zeros_like(data_info["density"][:, 0]))

    all_flats = tf_flat.define_flats(S, zeros_P, zeros_D)

    flats = {
        "linear_left": all_flats["flat_linear_left_D"],
        "linear_right": all_flats["flat_linear_right_D"],
    }

    def flat_par_P_wrap(x):
        return all_flats["flat_parabolic_P"](C.CochainP0(S, x)).coeffs

    def flat_par_D_wrap(x):
        return all_flats["flat_parabolic_D"](C.CochainD0(S, x)).coeffs

    def flat_lin_left_P_wrap(x):
        return all_flats["flat_linear_left_P"](C.CochainP0(S, x)).coeffs

    def flat_lin_left_D_wrap(x):
        return all_flats["flat_linear_left_D"](C.CochainD0(S, x)).coeffs

    def flat_lin_right_P_wrap(x):
        return all_flats["flat_linear_right_P"](C.CochainP0(S, x)).coeffs

    def flat_lin_right_D_wrap(x):
        return all_flats["flat_linear_right_D"](C.CochainD0(S, x)).coeffs

    flat_par_P = vmap(flat_par_P_wrap)
    flat_par_D = vmap(flat_par_D_wrap)
    flat_lin_left_P = vmap(flat_lin_left_P_wrap)
    flat_lin_left_D = vmap(flat_lin_left_D_wrap)
    flat_lin_right_P = vmap(flat_lin_right_P_wrap)
    flat_lin_right_D = vmap(flat_lin_right_D_wrap)

    def flat_primitive_par_P(c: C.CochainD0):
        return C.CochainP1(S, flat_par_P(c.coeffs.T)[:, :, 0].T)

    def flat_primitive_par_D(c: C.CochainD0):
        return C.CochainD1(S, flat_par_D(c.coeffs.T)[:, :, 0].T)

    def flat_primitive_lin_left_P(c: C.CochainD0):
        return C.CochainP1(S, flat_lin_left_P(c.coeffs.T)[:, :, 0].T)

    def flat_primitive_lin_left_D(c: C.CochainD0):
        return C.CochainD1(S, flat_lin_left_D(c.coeffs.T)[:, :, 0].T)

    def flat_primitive_lin_right_P(c: C.CochainD0):
        return C.CochainP1(S, flat_lin_right_P(c.coeffs.T)[:, :, 0].T)

    def flat_primitive_lin_right_D(c: C.CochainD0):
        return C.CochainD1(S, flat_lin_right_D(c.coeffs.T)[:, :, 0].T)

    pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0], C.CochainP0)
    pset.addPrimitive(
        flat_primitive_par_P, [C.CochainP0], C.CochainP1, name="flat_parP0"
    )
    pset.addPrimitive(
        flat_primitive_par_D, [C.CochainD0], C.CochainD1, name="flat_parD0"
    )
    pset.addPrimitive(
        flat_primitive_lin_left_P,
        [C.CochainP0],
        C.CochainP1,
        name="flat_lin_leftP0",
    )
    pset.addPrimitive(
        flat_primitive_lin_left_D,
        [C.CochainD0],
        C.CochainD1,
        name="flat_lin_leftD0",
    )
    pset.addPrimitive(
        flat_primitive_lin_right_P,
        [C.CochainP0],
        C.CochainP1,
        name="flat_lin_rightP0",
    )
    pset.addPrimitive(
        flat_primitive_lin_right_D,
        [C.CochainD0],
        C.CochainD1,
        name="flat_lin_rightD0",
    )

    # add float primitives

    # add special primitives
    add_new_primitives(pset)

    # add constants and terminals
    pset.addTerminal(object, float, "c")
    pset.addTerminal(C.CochainP0(S, jnp.ones(S.num_nodes)), C.CochainP0, "ones")

    # rename argument
    pset.renameArguments(ARG0="rho")

    rho_test = jnp.linspace(0, 1.0, 400)

    v_check_fn = jit(is_v_unfeasible, static_argnums=(0, 2, 3))

    common_params = {
        "rho_bnd": rho_bnd,
        "rho_0": rho_0,
        "S": S,
        "penalty": penalty,
        "num_t_points": num_t_points,
        "delta_t": delta_t,
        "step": step,
        "flats": flats,
        "rho_test": rho_test,
        "ansatz": ansatz,
        "task": task,
        "v_check_fn": v_check_fn,
    }

    pset = primitives.add_primitives_to_pset_from_dict(
        pset, config_file_data["gp"]["primitives"]
    )
    num_cpus = config_file_data["gp"]["num_cpus"]
    batch_size = config_file_data["gp"]["batch_size"]
    max_calls = config_file_data["gp"]["max_calls"]

    gpsr = GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness,
        score_func=eval_MSE,
        predict_func=predict,
        callback_func=assign_consts,
        print_log=True,
        common_data=common_params,
        save_best_individual=True,
        save_train_fit_history=True,
        output_path=output_path,
        seed_str=seed,
        num_cpus=num_cpus,
        batch_size=batch_size,
        max_calls=max_calls,
        custom_logger=custom_logger,
        **regressor_params,
    )

    return gpsr, flats


def stgp_traffic(
    regressor_params,
    config_file_data,
    density,
    v,
    f,
    X_training,
    X_test,
    S,
    delta_x,
    delta_t,
    num_t_points,
    step,
    rho_0,
    rho_bnd,
    t_sampled_circ,
    seed=None,
    output_path="./",
):

    gpsr, flats = set_prb(
        regressor_params,
        config_file_data,
        S,
        delta_x,
        delta_t,
        num_t_points,
        step,
        rho_0,
        rho_bnd,
        seed,
        output_path,
    )
    start = time.perf_counter()
    gpsr.fit(X_training)

    # test error
    print(f"Best MSE on the test set: ", gpsr.score(X_test))

    best_ind = gpsr.get_best_individuals()[0]
    best_consts = best_ind.consts

    print("Best constants = ", [f"{f:.20f}" for f in best_consts])

    # PLOTS
    test_errors = plots.stgp_traffic_plots(
        gpsr,
        S,
        flats,
        X_test,
        density,
        v,
        f,
        t_sampled_circ,
        step,
        output_path,
    )

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")
    return test_errors


if __name__ == "__main__":
    yamlfile = "stgp_traffic.yaml"
    filename = os.path.join(os.path.dirname(__file__), yamlfile)

    regressor_params, config_file_data = util.load_config_data(filename)
    road_name = config_file_data["gp"]["road_name"]
    task = config_file_data["gp"]["task"]

    data_info = preprocess_data(road_name)
    X_training, X_test = build_dataset(
        data_info["t_sampled_circ"],
        data_info["S"],
        data_info["density"],
        data_info["v"],
        data_info["flow"],
        task,
    )

    # seed = [
    #     "AddCP0(ones, conv_1P0(delP1(flat_lin_leftP0(ExpP0(MFP0(SqrtP0(SquareP0(rho)), c)))), ExpP0(MFP0(rho, c))))"
    # ]
    seed = [
        "SquareP0(ExpP0(conv_3P0(delP1(flat_lin_rightP0(rho)), MFP0(rho, 5.82940218613048344309))))"
    ]
    # seed = None
    output_path = "."

    dt = data_info["delta_t_refined"]

    stgp_traffic(
        regressor_params,
        config_file_data,
        data_info["density"],
        data_info["vP0"][:-1],
        data_info["fP0"][:-1],
        X_training,
        X_test,
        data_info["S"],
        data_info["delta_x"],
        dt,
        data_info["num_t_points"],
        data_info["step"],
        data_info["rho_0"],
        data_info["rho_bnd"],
        data_info["t_sampled_circ"],
        seed,
        output_path,
    )
