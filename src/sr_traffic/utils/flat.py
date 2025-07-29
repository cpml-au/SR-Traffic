import jax.numpy as jnp
from jax import Array
import dctkit as dt
import dctkit.dec.cochain as C
from dctkit.mesh import util
from dctkit.dec.flat import flat
from functools import partial

# FIXME: add the docs!

dt.config()


def build_rolling_matrix(v: Array):
    n = len(v)
    buffer = jnp.empty((n, n * 2 - 1))

    # generate a wider array that we want a slice into
    buffer = buffer.at[:, :n].set(v[:n])
    buffer = buffer.at[:, n:].set(v[: n - 1])

    rolled = buffer.reshape(-1)[n - 1 : -1].reshape(n, -1)
    K_full_roll = jnp.roll(rolled[:, :n], shift=1, axis=0)
    return K_full_roll[:-1, :]


def get_parabolic_weights(num_nodes: int, is_primal: bool):
    # NOTE: only 1D implementation
    if is_primal:
        # in this case num_cols = num_nodes and num_rows = num_nodes-1
        num_cols = num_nodes
        v = jnp.zeros(num_cols, dtype=dt.float_dtype)
        v = v.at[:2].set(0.5)
        return build_rolling_matrix(v)
    # otherwise, num_cols = num_nodes-1 and num_rows = num_nodes
    num_cols = num_nodes - 1
    v = jnp.zeros(num_cols, dtype=dt.float_dtype)
    v = v.at[:2].set(0.5)
    W = jnp.zeros((num_cols + 1, num_cols), dtype=dt.float_dtype)
    W = W.at[1:-1, :].set(build_rolling_matrix(v))
    return W


def get_upwind_left_weights(num_nodes: int, is_primal: bool):
    # NOTE: only 1D implementation
    if is_primal:
        # in this case num_cols = num_nodes and num_rows = num_nodes-1
        num_cols = num_nodes
        v = jnp.zeros(num_cols, dtype=dt.float_dtype)
        v = v.at[0].set(1.0)
        return build_rolling_matrix(v)
    # otherwise, num_cols = num_nodes-1 and num_rows = num_nodes
    num_cols = num_nodes - 1
    v = jnp.zeros(num_cols, dtype=dt.float_dtype)
    v = v.at[0].set(1.0)
    W = jnp.zeros((num_cols + 1, num_cols), dtype=dt.float_dtype)
    W = W.at[1:-1, :].set(build_rolling_matrix(v))
    W = W.at[-1, -1].set(1.0)
    return W


def get_upwind_right_weights(num_nodes: int, is_primal: bool):
    # NOTE: only 1D implementation
    if is_primal:
        # in this case num_cols = num_nodes and num_rows = num_nodes-1
        num_cols = num_nodes
        v = jnp.zeros(num_cols, dtype=dt.float_dtype)
        v = v.at[1].set(1.0)
        return build_rolling_matrix(v)
    # otherwise, num_cols = num_nodes-1 and num_rows = num_nodes
    num_cols = num_nodes - 1
    v = jnp.zeros(num_cols, dtype=dt.float_dtype)
    v = v.at[0].set(1.0)
    W = jnp.zeros((num_cols + 1, num_cols), dtype=dt.float_dtype)
    W = W.at[:-2, :].set(build_rolling_matrix(v))
    W = W.at[-2, -1].set(1.0)
    return W


def get_linear_left_interpolation(
    c: C.CochainP0 | C.CochainD0, sigma: C.CochainP0 | C.CochainD0
) -> Array:
    if c.is_primal:
        primal_volumes = c.complex.primal_volumes[1].reshape(-1, 1)
        coeffs = jnp.zeros(
            (c.complex.num_nodes - 1, c.coeffs.shape[1]), dtype=dt.float_dtype
        )
        coeffs = c.coeffs[1:] - 1 / 2 * primal_volumes * sigma.coeffs[1:]
        return coeffs

    dual_volumes = c.complex.dual_volumes[0].reshape(-1, 1)
    coeffs = jnp.zeros((c.complex.num_nodes, c.coeffs.shape[1]), dtype=dt.float_dtype)
    coeffs = coeffs.at[:-1].set(c.coeffs - 1 / 2 * dual_volumes[:-1] * sigma.coeffs)
    return coeffs


def get_linear_right_interpolation(
    c: C.CochainP0 | C.CochainD0, sigma: C.CochainP0 | C.CochainD0
) -> Array:
    if c.is_primal:
        primal_volumes = c.complex.primal_volumes[1].reshape(-1, 1)
        coeffs = jnp.zeros(
            (c.complex.num_nodes - 1, c.coeffs.shape[1]), dtype=dt.float_dtype
        )
        coeffs = c.coeffs[:-1] + 1 / 2 * primal_volumes * sigma.coeffs[:-1]
        return coeffs

    dual_volumes = c.complex.dual_volumes[0].reshape(-1, 1)
    coeffs = jnp.zeros((c.complex.num_nodes, c.coeffs.shape[1]), dtype=dt.float_dtype)
    coeffs = coeffs.at[1:].set(c.coeffs + 1 / 2 * dual_volumes[1:] * sigma.coeffs)
    return coeffs


def define_flats(S, zeros_P, zeros_D):
    W_parabolic_D_T = get_parabolic_weights(S.num_nodes, False)
    W_parabolic_P_T = get_parabolic_weights(S.num_nodes, True)
    I_linear_left = get_linear_left_interpolation
    I_linear_right = get_linear_right_interpolation
    primal_edges = C.CochainP1(S, S.primal_edges_vectors)
    dual_edges = C.CochainD1(S, S.dual_edges_vectors)
    flat_parabolic_P = partial(flat, weights=W_parabolic_P_T.T, edges=primal_edges)
    flat_parabolic_D = partial(flat, weights=W_parabolic_D_T.T, edges=dual_edges)
    flat_linear_left_D = partial(
        flat,
        weights=None,
        edges=dual_edges,
        interp_func=I_linear_left,
        interp_func_args={"sigma": zeros_D},
    )
    flat_linear_right_D = partial(
        flat,
        weights=None,
        edges=dual_edges,
        interp_func=I_linear_right,
        interp_func_args={"sigma": zeros_D},
    )
    flat_linear_left_P = partial(
        flat,
        weights=None,
        edges=primal_edges,
        interp_func=I_linear_left,
        interp_func_args={"sigma": zeros_P},
    )
    flat_linear_right_P = partial(
        flat,
        weights=None,
        edges=primal_edges,
        interp_func=I_linear_right,
        interp_func_args={"sigma": zeros_P},
    )
    flats = {
        "flat_linear_left_D": flat_linear_left_D,
        "flat_linear_right_D": flat_linear_right_D,
        "flat_linear_left_P": flat_linear_left_P,
        "flat_linear_right_P": flat_linear_right_P,
        "flat_parabolic_P": flat_parabolic_P,
        "flat_parabolic_D": flat_parabolic_D,
    }

    return flats
