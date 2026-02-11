from flex.gp.primitives import generate_primitive_variants
from flex.gp.jax_primitives import *
from functools import partial
from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from jax import vmap
from deap import gp
from typing import Dict


def constant_sub(k: float, c: C.Cochain) -> C.Cochain:
    """Compute the cochain subtraction between a constant cochain and another cochain.

    Args:
        k: a constant.
        c: a cochain.

    Returns:
        the resulting subtraction
    """
    return C.Cochain(c.dim, c.is_primal, c.complex, k - c.coeffs)


def add_new_primitives(
    pset: gp.PrimitiveSetTyped, S: SimplicialComplex, all_flats: Dict
):
    # Define the modules and functions needed to eval inputs and outputs
    modules_functions = {"dctkit.dec": ["cochain"]}

    subFC = {
        "fun_info": {"name": "SubFC", "fun": constant_sub},
        "input": ["float", "cochain.Cochain"],
        "output": "cochain.Cochain",
        "att_input": {
            "complex": ("P", "D"),
            "dimension": ("0", "1", "2"),
            "rank": ("SC",),
        },
        "map_rule": {
            "complex": lambda x: x,
            "dimension": lambda x: x,
            "rank": lambda x: x,
        },
    }
    new_primitives = [subFC]
    for i in range(1, 4):
        conv_i = {
            "fun_info": {
                "name": "conv_" + str(i),
                "fun": partial(C.convolution, kernel_window=int(i)),
            },
            "input": ["cochain.Cochain", "cochain.Cochain"],
            "output": "cochain.Cochain",
            "att_input": {"complex": ("P", "D"), "dimension": ("0",), "rank": ("SC",)},
            "map_rule": {
                "complex": lambda x: x,
                "dimension": lambda x: x,
                "rank": lambda x: x,
            },
        }
        new_primitives.append(conv_i)

    new_generated_primitives = list(
        map(
            partial(generate_primitive_variants, imports=modules_functions),
            new_primitives,
        )
    )
    for new_primitive in new_generated_primitives:
        for primitive_name in new_primitive.keys():
            op = new_primitive[primitive_name].op
            in_types = new_primitive[primitive_name].in_types
            out_type = new_primitive[primitive_name].out_type
            pset.addPrimitive(op, in_types, out_type, name=primitive_name)

    # add flats primitives
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
