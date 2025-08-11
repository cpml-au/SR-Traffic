import numpy as np
import os
from typing import Tuple
import numpy.typing as npt
from dctkit import config
from sklearn.model_selection import train_test_split
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.dec import cochain as C
from dctkit.dec.flat import flat
import traffic_flow.utils.flat as tf_flat
from functools import partial
from jax import vmap
import math
from dctkit.mesh import util
from scipy.interpolate import interp1d


config()

data_path = os.path.dirname(os.path.realpath(__file__))


def preprocess_data(road_name):
    road_path = os.path.join(data_path, f"{road_name}")
    if road_name == "US101":
        v_int_path = os.path.join(road_path, f"NGSIM_{road_name}_Velocity_Data.txt")
        density_int_path = os.path.join(
            road_path, f"NGSIM_{road_name}_Density_Data.txt"
        )

        x0 = 40
        x_max = 2080
        t0 = 0
        t_max_dim = 45 * 60
        delta_t_factor = 0.025

    elif road_name == "US80":
        v_int_path = os.path.join(road_path, f"NGSIM_{road_name}_4pm_Velocity_Data.txt")
        density_int_path = os.path.join(
            road_path, f"NGSIM_{road_name}_4pm_Density_Data.txt"
        )

        x0 = 20
        x_max = 1600  # - 16 * 20
        t0 = 0
        t_max_dim = 15 * 60
        delta_t_factor = 0.025

    L_dim = x_max - x0
    t_len_dim = t_max_dim - t0
    delta_x_dim = 20.0
    delta_t_dim = 5.0

    v_int_dim = np.loadtxt(v_int_path, skiprows=0, dtype=np.float64)[1:-1, :]
    density_int_dim = np.loadtxt(density_int_path, skiprows=0, dtype=np.float64)[
        1:-1, :
    ]
    flow_int_dim = v_int_dim * density_int_dim

    # dimensionless variables
    x_max = 1.0
    V = 100.0
    t_len = L_dim / V
    t_max = t_len_dim / t_len
    delta_x = delta_x_dim / L_dim
    delta_t = delta_t_dim / t_len
    x_sampled = np.linspace(0, x_max, math.ceil(x_max / delta_x) + 1)
    t_sampled = np.linspace(0, t_max, math.ceil(t_max / delta_t) + 1)

    density_max = np.max(density_int_dim)
    density_int = density_int_dim / density_max
    v_int = v_int_dim / V
    flow_int = flow_int_dim / (density_max * V)

    # generate_mesh and define simplicial complex
    mesh, _ = util.generate_line_mesh(len(x_sampled), L=1.0, x_min=0.0)
    S = util.build_complex_from_mesh(mesh, space_dim=1)
    S.get_hodge_star()
    S.get_primal_edge_vectors()
    S.get_dual_edge_vectors()

    t_sampled_circ = (t_sampled[:-1] + t_sampled[1:]) / 2

    delta_t_refined = delta_t_factor * delta_t
    num_t_points = (
        round(((t_sampled_circ[-1] - t_sampled_circ[0])) / delta_t_refined) + 1
    )
    step = round(delta_t / delta_t_refined)

    dual_edges = C.CochainD1(S, S.dual_edges_vectors)
    zeros = C.CochainD0(S, np.zeros_like(density_int[:, 0]))
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

    flat_rho = flat_left(density_int.T)[:, :, 0].T
    flat_v = flat_left(v_int.T)[:, :, 0].T
    flat_f = flat_left(flow_int.T)[:, :, 0].T

    rhoP0 = C.star(C.CochainD1(S, flat_rho))
    vP0 = C.star(C.CochainD1(S, flat_v))
    fP0 = C.star(C.CochainD1(S, flat_f))

    # set-up bnd conditions
    rho_x_0 = density_int[0, :]
    rho_x_m_1 = density_int[-1, :]
    rho_x_m_2 = density_int[-2, :]
    rho_x_m_3 = density_int[-3, :]

    int_rho_x_0 = interp1d(t_sampled_circ, rho_x_0)
    int_rho_x_m_1 = interp1d(t_sampled_circ, rho_x_m_1)
    int_rho_x_m_2 = interp1d(t_sampled_circ, rho_x_m_2)
    int_rho_x_m_3 = interp1d(t_sampled_circ, rho_x_m_3)

    t_circ_refined = np.linspace(t_sampled_circ[0], t_sampled_circ[-1], num_t_points)

    rho_0 = density_int[:, 0]
    rho_bnd = {
        "0": int_rho_x_0(t_circ_refined),
        "-1": int_rho_x_m_1(t_circ_refined),
        "-2": int_rho_x_m_2(t_circ_refined),
        "-3": int_rho_x_m_3(t_circ_refined),
    }
    data_info = {
        "density": density_int,
        "v": v_int,
        "flow": flow_int,
        "rhoP0": rhoP0.coeffs,
        "vP0": vP0.coeffs,
        "fP0": fP0.coeffs,
        "x_sampled": x_sampled,
        "t_sampled_circ": t_sampled_circ,
        "S": S,
        "delta_x": delta_x,
        "delta_t": delta_t,
        "delta_t_refined": delta_t_refined,
        "num_t_points": num_t_points,
        "step": step,
        "rho_0": rho_0,
        "rho_bnd": rho_bnd,
        "density_max": density_max,
        "t_len": t_len,
        "L_dim": L_dim,
        "V": V,
    }

    return data_info


def build_dataset(
    t: npt.NDArray,
    S: SimplicialComplex,
    density: npt.NDArray,
    velocity: npt.NDArray,
    flow: npt.NDArray,
    task: str,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    dual_edges = C.CochainD1(S, S.dual_edges_vectors)
    zeros = C.CochainD0(S, np.zeros_like(density[:, 0]))
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

    flat_v = flat_left(velocity.T)[:, :, 0].T
    flat_f = flat_left(flow.T)[:, :, 0].T

    vP0 = C.star(C.CochainD1(S, flat_v))
    fP0 = C.star(C.CochainD1(S, flat_f))
    velocity_data = vP0.coeffs[:-1]
    flow_data = fP0.coeffs[:-1]

    num_t_points = velocity_data.shape[1]

    if task == "prediction":
        # time splitting (prediction)
        tr_test_tuple = train_test_split(
            np.arange(num_t_points),
            velocity_data.T,
            test_size=0.4,
            random_state=42,
            shuffle=False,
        )
        t_tr_idx, t_test_idx, _, _ = tr_test_tuple
        x_idx = np.arange(1, density.shape[0] - 3)

        _, ax_tr = np.meshgrid(x_idx, t_tr_idx)
        _, ax_test = np.meshgrid(x_idx, t_test_idx)
        rho_tr = density[1:-3, t_tr_idx].ravel("F")
        rho_test = density[1:-3, t_test_idx].ravel("F")

        v_tr = velocity_data[1:-3, t_tr_idx].ravel("F")
        v_test = velocity_data[1:-3, t_test_idx].ravel("F")

        f_tr = flow_data[1:-3, t_tr_idx].ravel("F")
        f_test = flow_data[1:-3, t_test_idx].ravel("F")
    elif task == "reconstruction":
        # space splitting (reconstruction)
        num_x_points = velocity_data[1:-3].shape[0]

        tr_test_tuple = train_test_split(
            np.arange(num_x_points),
            velocity_data[1:-3],
            test_size=0.8,
            random_state=42,
            shuffle=True,
        )
        x_tr_idx, x_test_idx, _, _ = tr_test_tuple
        # reorder indices
        x_tr_idx = np.sort(x_tr_idx)
        x_test_idx = np.sort(x_test_idx)
        t_idx = np.arange(num_t_points)

        ax_tr, _ = np.meshgrid(x_tr_idx, t_idx)
        ax_test, _ = np.meshgrid(x_test_idx, t_idx)

        rho_tr = density[1:-3, :][x_tr_idx, :].ravel("F")
        rho_test = density[1:-3, :][x_test_idx, :].ravel("F")

        v_tr = velocity_data[1:-3, :][x_tr_idx, :].ravel("F")
        v_test = velocity_data[1:-3, :][x_test_idx, :].ravel("F")

        f_tr = flow_data[1:-3, :][x_tr_idx, :].ravel("F")
        f_test = flow_data[1:-3, :][x_test_idx, :].ravel("F")

    # build X
    X_tr = np.column_stack((ax_tr.ravel(), rho_tr, v_tr, f_tr))
    X_test = np.column_stack((ax_test.ravel(), rho_test, v_test, f_test))

    return X_tr, X_test
