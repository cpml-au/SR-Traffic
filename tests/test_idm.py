import jax.numpy as jnp
from jax import jit
from sr_traffic.fund_diagrams.fund_diagrams_def import inverse_IDM, IDM_fn


def test_idm():
    opt_idm = [0.95870305, 0.9524395, 0.68423716, 0.41707987]
    flux_idm_args = {
        "s0": opt_idm[0],
        "T": opt_idm[1],
        "delta": opt_idm[2],
        "v0": opt_idm[3],
    }
    s_targets = jnp.linspace(10, 100, 100)
    jit_inv = jit(inverse_IDM, static_argnums=(1, 2, 3, 4))
    v = jit_inv(s_targets, opt_idm[0], opt_idm[1], opt_idm[2], opt_idm[3])
    s_comp = IDM_fn(v, **flux_idm_args)
    assert jnp.allclose(s_comp, s_targets, 1e-3)
