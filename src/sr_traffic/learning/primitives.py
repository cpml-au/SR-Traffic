from flex.gp.primitives import generate_primitive_variants, PrimitiveParams
from flex.gp.jax_primitives import *
from functools import partial
from dctkit.dec import cochain as C
from jax import Array
import jax.numpy as jnp


def constant_sub(k: float, c: C.Cochain) -> C.Cochain:
    """Compute the cochain subtraction between a constant cochain and another cochain.

    Args:
        k: a constant.
        c: a cochain.

    Returns:
        the resulting subtraction
    """
    return C.Cochain(c.dim, c.is_primal, c.complex, k - c.coeffs)


def add_new_primitives(pset):
    # Define the modules and functions needed to eval inputs and outputs
    modules_functions = {"dctkit.dec": ["cochain"]}

    subFC = {
        "fun_info": {"name": "SubFC", "fun": constant_sub},
        "input": ["float", "cochain.Cochain"],
        "output": "cochain.Cochain",
        "att_input": {
            "category": ("P", "D"),
            "dimension": ("0", "1", "2"),
            "rank": ("SC",),
        },
        "map_rule": {
            "category": lambda x: x,
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
            "att_input": {"category": ("P", "D"), "dimension": ("0",), "rank": ("SC",)},
            "map_rule": {
                "category": lambda x: x,
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
