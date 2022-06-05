import jax.numpy as jnp
from typing import Callable

from rootfind import rootfind
#import rootfind_grad


def deq(params: dict, rng, x: jnp.ndarray, f: Callable, max_iter: int, *args) -> jnp.ndarray:
    # Define the equilibrium point
    # g(z)=f(z)-z
    def g(params, rng, x, *args): return f(params, rng, x, *args) - x

    # Analytically find the root using one
    # of the following solvers:
    # - Broyden
    # - Anderson
    z_star = rootfind(g, max_iter, params, rng, x, *args)

    # TODO

    return z_star
