from dctkit import config as config
from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from typing import Tuple, Callable, Dict, List
import numpy.typing as npt
from jax import jit
import jax.numpy as jnp
import sr_traffic.utils.flat as tf_flat
from sr_traffic.learning.primitives import add_new_primitives
from sr_traffic.data.data import preprocess_data, build_dataset
from flex.gp import util, primitives
from flex.gp.regressor import GPSymbolicRegressor
from deap import gp
from deap.base import Toolbox
import warnings
import pygmo as pg
from sr_traffic.learning.utils import *
import os
import time
import gc

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF warnings (JAX uses XLA)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hides all GPUs from JAX
os.environ["JAX_LOG_COMPILES"] = "0"

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()


def eval_MSE_sol(
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
    tree: gp.PrimitiveTree,
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
    v_check_fn: Callable,
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


def score(
    individuals_batch: List[gp.PrimitiveTree],
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
    v_check_fn: Callable,
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
        # we want to maximize the score -> negative MSE
        objvals[i] *= -1.0

    return objvals


def predict(
    individuals_batch: List[gp.PrimitiveTree],
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
    v_check_fn: Callable,
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
    individuals_batch: List[gp.PrimitiveTree],
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
    v_check_fn: Callable,
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


def stgp_traffic(
    regressor_params: Dict,
    config_file_data: Dict,
    density: npt.NDArray,
    v: npt.NDArray,
    f: npt.NDArray,
    X_training: npt.NDArray,
    X_validation: npt.NDArray,
    X_train_val: npt.NDArray,
    X_test: npt.NDArray,
    S: SimplicialComplex,
    delta_t: float,
    num_t_points: int,
    step: int,
    rho_0: npt.NDArray,
    rho_bnd: Dict,
    t_sampled_circ: npt.NDArray,
    seed: List = None,
    output_path: str = "./",
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

    pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0], C.CochainP0)

    # add special primitives
    add_new_primitives(pset, S, all_flats)

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
        score_func=score,
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
    validate = config_file_data["gp"]["validate"]
    start = time.perf_counter()
    if validate:
        gpsr.fit(X_training, None, X_validation, None)
    else:
        gpsr.fit(X_train_val)

        # PLOTS
        stgp_traffic_plots(
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

        # test error
        print(f"Best MSE on the test set: ", gpsr.score(X_test))

    best_ind = gpsr.get_best_individuals()[0]
    best_consts = best_ind.consts

    print("Best constants = ", [f"{f:.20f}" for f in best_consts])

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")


if __name__ == "__main__":
    yamlfile = "stgp_traffic.yaml"
    filename = os.path.join(os.path.dirname(__file__), yamlfile)

    regressor_params, config_file_data = util.load_config_data(filename)
    road_name = config_file_data["gp"]["road_name"]
    task = config_file_data["gp"]["task"]
    set_seed = config_file_data["gp"]["set_seed"]

    data_info = preprocess_data(road_name)
    X_tr, X_val, X_tr_val, X_test = build_dataset(
        data_info["t_sampled_circ"],
        data_info["S"],
        data_info["density"],
        data_info["v"],
        data_info["flow"],
        task,
    )

    if task == "prediction":
        seed = [
            "AddCP0(ones, conv_1P0(delP1(flat_lin_leftP0(ExpP0(MFP0(SqrtP0(SquareP0(rho)), c)))), ExpP0(MFP0(rho, c))))"
        ]
    elif task == "reconstruction":
        seed = ["SquareP0(ExpP0(conv_3P0(delP1(flat_lin_rightP0(rho)), MFP0(rho, c))))"]
    if not set_seed:
        seed = None
    output_path = "."

    dt = data_info["delta_t_refined"]

    stgp_traffic(
        regressor_params,
        config_file_data,
        data_info["density"],
        data_info["vP0"][:-1],
        data_info["fP0"][:-1],
        X_tr,
        X_val,
        X_tr_val,
        X_test,
        data_info["S"],
        dt,
        data_info["num_t_points"],
        data_info["step"],
        data_info["rho_0"],
        data_info["rho_bnd"],
        data_info["t_sampled_circ"],
        seed,
        output_path,
    )
