from dctkit.dec import cochain as C
from sr_traffic.data.data import preprocess_data, build_dataset
import sr_traffic.fund_diagrams.fund_diagrams_def as tf_utils
from sr_traffic.utils.godunov import godunov_solver
import jax.numpy as jnp
from jax import vmap
import pygmo as pg
import time
from functools import partial
import sr_traffic.utils.flat as tf_flat
from dctkit.dec.flat import flat
from jax import jit
from dctkit import config

config()


def relative_squared_error(x, x_true, idx, step, task):
    x_norm = jnp.sum(x_true**2)
    if task == "prediction":
        error = jnp.sum((x[1:-3, idx * step].ravel("F") - x_true[:]) ** 2)
    elif task == "reconstruction":
        error = jnp.sum((x[1:-3, ::step][idx].ravel("F") - x_true[:]) ** 2)
    return error / x_norm * 100


class Calibration:
    def __init__(
        self,
        S,
        rho,
        v,
        f,
        rho_god,
        num_t_points,
        delta_t_refined,
        step,
        train_idx,
        flux,
        flux_der,
        flats,
        task,
    ):
        self.S = S
        self.rho = rho
        self.v = v
        self.f = f
        self.rho_god = rho_god
        self.num_t_points = num_t_points
        self.delta_t_refined = delta_t_refined
        self.step = step
        self.train_idx = train_idx
        self.flux = flux
        self.flux_der = flux_der
        self.flats = flats
        self.task = task

    @partial(jit, static_argnums=(0, 2))
    def error(self, flux_cal_params, num_t_points):
        flux = lambda x: self.flux(x, *flux_cal_params)
        flux_der = lambda x: self.flux_der(x, *flux_cal_params)
        rho_computed, v_computed, f_computed = godunov_solver(
            self.rho_god[:, 0],
            S,
            self.rho_god,
            flux,
            flux_der,
            self.delta_t_refined,
            0.0,
            flats,
            num_t_points,
        )

        rho_error = relative_squared_error(
            rho_computed, self.rho, self.train_idx, self.step, self.task
        )
        v_error = relative_squared_error(
            v_computed, self.v, self.train_idx, self.step, self.task
        )
        total_error = 0.5 * rho_error + 0.5 * v_error
        return total_error, rho_computed, v_computed, f_computed

    def fitness(self, flux_cal_params):
        if self.task == "prediction":
            num_t_points = int(self.train_idx[-1] * self.step + 1)
        elif self.task == "reconstruction":
            num_t_points = self.num_t_points
        # num_t_points = int(self.train_idx[-1] + 1)
        total_error = self.error(flux_cal_params, num_t_points)[0]
        if jnp.isnan(total_error) or total_error >= 1e3:
            total_error = 1e3

        return [total_error]

    def get_bounds(self):
        # return ([0.1, 0.0, 0.0], [1.0, 1.3, 10.0])
        return ([0.0, 0.0, 0.0], [0.7, 5.0, 10.0])
        # return ([0.0, 0], [1, 1.3])
        # return ([0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
        # return ([0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 10.0])


if __name__ == "__main__":
    task = "prediction"
    data_info = preprocess_data("US80")
    X_training, X_test = build_dataset(
        data_info["t_sampled_circ"],
        data_info["S"],
        data_info["density"],
        data_info["v"],
        data_info["flow"],
        task,
    )

    rho_godunov = jnp.zeros(
        (len(data_info["x_sampled"]) - 1, data_info["num_t_points"])
    )
    rho_godunov = rho_godunov.at[:, 0].set(data_info["rho_0"])
    for index in data_info["rho_bnd"].keys():
        rho_godunov = rho_godunov.at[int(index), :].set(data_info["rho_bnd"][index])

    S = data_info["S"]

    zeros = C.CochainD0(S, jnp.zeros_like(data_info["density"][:, 0]))

    I_linear_left = tf_flat.get_linear_left_interpolation
    I_linear_right = tf_flat.get_linear_right_interpolation
    dual_edges = C.CochainD1(S, S.dual_edges_vectors)
    flat_linear_left_D = partial(
        flat,
        weights=None,
        edges=dual_edges,
        interp_func=I_linear_left,
        interp_func_args={"sigma": zeros},
    )
    flat_linear_right_D = partial(
        flat,
        weights=None,
        edges=dual_edges,
        interp_func=I_linear_right,
        interp_func_args={"sigma": zeros},
    )
    W_parabolic_P_T = tf_flat.get_parabolic_weights(S.num_nodes, True)
    primal_edges = C.CochainP1(S, S.primal_edges_vectors)
    flat_parabolic_P = partial(flat, weights=W_parabolic_P_T.T, edges=primal_edges)

    def flat_left_wrap(x):
        return flat_linear_left_D(C.CochainD0(S, x)).coeffs

    flat_left = vmap(flat_left_wrap)

    flats = {
        "linear_left": flat_linear_left_D,
        "linear_right": flat_linear_right_D,
        "linear_left_v": flat_left,
    }

    flux = tf_utils.triangular_flux
    flux_der = tf_utils.define_flux_der(S, flux)

    if task == "prediction":
        train_idx = jnp.arange(X_training[0, 0], X_training[-1, 0] + 1, dtype=jnp.int64)
        test_idx = jnp.arange(X_test[0, 0], X_test[-1, 0] + 1, dtype=jnp.int64)
    elif task == "reconstruction":
        num_tr = int(X_training.shape[0] / len(data_info["t_sampled_circ"]))
        num_test = int(X_test.shape[0] / len(data_info["t_sampled_circ"]))
        train_idx = X_training[:num_tr, 0].astype(jnp.int64)
        test_idx = X_test[:num_test, 0].astype(jnp.int64)

    calib = Calibration(
        S,
        X_training[:, 1],
        X_training[:, 2],
        X_training[:, 3],
        rho_godunov,
        data_info["num_t_points"],
        data_info["delta_t_refined"],
        data_info["step"],
        train_idx,
        flux,
        flux_der,
        flats,
        task,
    )
    algo = pg.algorithm(pg.sea(gen=100))
    print("Calibration started")
    tic = time.time()
    prob = pg.problem(calib)
    algo.set_verbosity(1)
    pop = pg.population(prob, size=1000)
    pop = algo.evolve(pop)
    toc = time.time()
    print(f"Done in {toc-tic} s!")
    best_consts = pop.champion_x
    print(best_consts)
    # compute rho, v and f with the best constants
    _, rho_computed, v_computed, f_computed = calib.error(
        best_consts, data_info["num_t_points"]
    )

    rho_computed = rho_computed[:, :: data_info["step"]]
    v_computed = v_computed[:, :: data_info["step"]]
    f_computed = f_computed[:, :: data_info["step"]]

    v_computed = v_computed.at[:, 0].set(data_info["v"][:, 0])
    v_computed = v_computed.at[0, :].set(data_info["v"][0, :])
    v_computed = v_computed.at[-3:, :].set(data_info["v"][-3:, :])

    f_computed = f_computed.at[:, 0].set(data_info["flow"][:, 0])
    f_computed = f_computed.at[0, :].set(data_info["flow"][0, :])
    f_computed = f_computed.at[-3:, :].set(data_info["flow"][-3:, :])

    # compute errors
    rho_norm = jnp.sqrt(jnp.sum(jnp.square(data_info["density"])))
    v_norm = jnp.sqrt(jnp.sum(jnp.square(data_info["v"])))
    f_norm = jnp.sqrt(jnp.sum(jnp.square(data_info["flow"])))

    rho_error = (
        jnp.sqrt(jnp.sum(jnp.square(rho_computed - data_info["density"]))) / rho_norm
    )
    v_error = jnp.sqrt(jnp.sum(jnp.square(v_computed - data_info["v"]))) / v_norm
    f_error = jnp.sqrt(jnp.sum(jnp.square(f_computed - data_info["flow"]))) / f_norm

    print(rho_error, v_error, f_error)
