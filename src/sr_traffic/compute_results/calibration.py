from dctkit.dec import cochain as C
from sr_traffic.data.data import preprocess_data, build_dataset
import sr_traffic.utils.fund_diagrams as tf_utils
from sr_traffic.utils.primitives import constant_sub
from sr_traffic.stgp_traffic import body_fun
import jax.numpy as jnp
from jax import lax, vmap
import pygmo as pg
import time
from functools import partial
import sr_traffic.utils.flat as tf_flat
from dctkit.dec.flat import flat
from jax import jit, jacfwd
from sr_traffic.utils.primitives import *
from dctkit import config

config()


def relative_squared_error(x, x_true, idx, step):
    x_norm = jnp.sum(x_true**2)
    error = jnp.sum((x[1:-3, ::step][idx].ravel("F") - x_true[:]) ** 2)
    # error = jnp.sum((x[1:-3, idx * step].ravel("F") - x_true[:]) ** 2)
    # error = jnp.sum((x[:-1, idx] - x_true[:-1, idx]) ** 2)
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
        v_max,
        rho_max,
        compute_errors=False,
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
        self.v_max = v_max
        self.rho_max = rho_max
        self.compute_errors = compute_errors

    @partial(jit, static_argnums=(0, 2))
    def error(self, v_rho_jam, num_t_points):
        # flux_args = {"V_0": v_rho_jam[0], "l_eff": v_rho_jam[1], "T": v_rho_jam[2]}
        # flux_args = {"v_max": v_rho_jam[0], "rho_max": v_rho_jam[1]}
        # flux_args = {
        #     "v_max": v_rho_jam[0],
        #     "rho_max": v_rho_jam[1],
        #     "lambda_w": v_rho_jam[2],
        # }
        # flux_args = {
        #     "s0": v_rho_jam[0],
        #     "T": v_rho_jam[1],
        #     "delta": v_rho_jam[2],
        #     "v0": v_rho_jam[3],
        # }
        # flux_args = {
        #     "C_jam": v_rho_jam[0],
        #     "V_max": v_rho_jam[1],
        #     "rho_max": v_rho_jam[2],
        #     "theta": v_rho_jam[3],
        # }
        flux = lambda x: self.flux(x, *v_rho_jam)
        flux_der = lambda x: self.flux_der(x, *v_rho_jam)
        f0 = partial(
            body_fun,
            self.S,
            self.rho_god,
            flux,
            flux_der,
            self.delta_t_refined,
            0.0,
            self.flats,
        )

        _, rho_v_f = lax.scan(
            f0, (self.rho_god[:, 0].reshape(-1, 1), False), jnp.arange(num_t_points - 1)
        )
        rho_1_T = rho_v_f[0][:, :, 0]
        v_1_T = rho_v_f[1][:, :, 0]
        f_1_T = rho_v_f[2][:, :, 0]

        # first interpolate rho_0, then compute velocity
        rho_0_P0 = C.star(flats["linear_left"](C.CochainD0(S, self.rho_god[:, 0])))
        f_0 = flux(rho_0_P0).coeffs
        v_0 = f_0 / rho_0_P0.coeffs

        # insert initial values of v and f
        rho_computed = jnp.vstack([self.rho_god[:, 0], rho_1_T]).T
        v_computed = jnp.vstack([v_0[:-1].flatten(), v_1_T]).T
        f_computed = jnp.vstack([f_0[:-1].flatten(), f_1_T]).T
        # rho_computed = self.rho
        # f_computed = flux(C.CochainP0(S, rho_computed)).coeffs
        # v_computed = f_computed / rho_computed

        if self.compute_errors:
            rho_error = relative_squared_error(
                rho_computed, self.rho, self.train_idx, self.step
            )
            v_error = relative_squared_error(
                v_computed, self.v, self.train_idx, self.step
            )
            f_error = relative_squared_error(
                f_computed, self.f, self.train_idx, self.step
            )
            total_error = 0.5 * rho_error + 0.5 * v_error
        else:
            total_error = jnp.nan
        return total_error, rho_computed, v_computed, f_computed

    def fitness(self, v_rho_jam):
        # if self.jitted_is_v_unfeasible(v_rho_jam) == 0:
        #    return [1e3]
        # num_t_points = int(self.train_idx[-1] * self.step + 1)
        # num_t_points = int(self.train_idx[-1] + 1)
        total_error = self.error(v_rho_jam, self.num_t_points)[0]
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
    data_info = preprocess_data("US80")
    X_training, X_test = build_dataset(
        data_info["t_sampled_circ"],
        data_info["S"],
        data_info["density"],
        data_info["v"],
        data_info["flow"],
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

    # flat_v = C.CochainD1(S, flat_par(data_info['v'].T)[:, :, 0].T)
    # vP0 = C.star(flat_v).coeffs
    # build the kernel
    d = 3 * data_info["delta_x"]
    n_d = int(d / data_info["delta_x"])
    y_kernel = jnp.linspace(0.0, d, num=n_d)
    eta = linear_kernel(S, y_kernel, d)

    flux = tf_utils.triangular_flux
    flux_der = tf_utils.define_flux_der(S, flux)

    v_max = 1.0
    rho_max = 1.0

    # train_idx = jnp.arange(X_training[0, 0], X_training[-1, 0] + 1, dtype=jnp.int64)
    # test_idx = jnp.arange(X_test[0, 0], X_test[-1, 0] + 1, dtype=jnp.int64)
    num_tr = int(X_training.shape[0] / len(data_info["t_sampled_circ"]))
    num_test = int(X_test.shape[0] / len(data_info["t_sampled_circ"]))
    train_idx = X_training[:num_tr, 0].astype(jnp.int64)  # + 1
    # train_idx = jnp.concatenate((train_idx, jnp.array([0, 76, 77, 78])))
    test_idx = X_test[:num_test, 0].astype(jnp.int64)  # + 1

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
        v_max,
        rho_max,
        True,
    )

    # calib = Calibration(
    #     S,
    #     data_info["rhoP0"],
    #     data_info["vP0"],
    #     data_info["fP0"],
    #     rho_godunov,
    #     data_info["num_t_points"],
    #     data_info["delta_t_refined"],
    #     data_info["step"],
    #     train_idx,
    #     flux,
    #     flux_der,
    #     flats,
    #     v_max,
    #     rho_max,
    #     True,
    # )

    algo = pg.algorithm(pg.pso(gen=50))
    print("Calibration started")
    tic = time.time()
    prob = pg.problem(calib)
    algo.set_verbosity(1)
    pop = pg.population(prob, size=100)
    pop = algo.evolve(pop)
    toc = time.time()
    print(f"Done in {toc-tic} s!")
    best_consts = pop.champion_x
    print(best_consts)
    # best_consts = [0.43020225, 1.40674734, 7.40130068]
    # compute rho, v and f with the best constants
    # in this case I want to solve the PDE in the full domain so I need to update
    #  the t-grid
    # calib.train_idx = list(range(data_info["density"].shape[1]))
    calib.compute_errors = False
    # num_t_points = int(calib.train_idx[-1] * calib.step + 1)
    num_t_points = data_info["num_t_points"]
    _, rho_computed, v_computed, f_computed = calib.error(best_consts, num_t_points)

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
