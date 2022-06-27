import jax.numpy as jnp
from jax import grad
from typing import Callable

from .rootfind import rootfind, rootfind_grad


def deq(params: dict, solver: int, mode: int, rng, x: jnp.ndarray, f: Callable, max_iter: int, *args) -> jnp.ndarray:
    # Define the equilibrium point
    # g(z)=f(z)-z
    def g(params, rng, x, *args):
        return f(params, rng, x, *args) - x

    # Analytically find the root using one
    # of the following solvers:
    # - Broyden
    # - Anderson
    z_star = rootfind(g,
                      max_iter,
                      solver,
                      mode,
                      params,
                      rng,
                      x,
                      *args
                      )

    # TODO
    # Why are we re-calculating this?
    # Isn't it supposed to be the same?
    z_star = f(params, rng, z_star, None)
    z_star = rootfind_grad(g,
                           max_iter,
                           solver,
                           mode,
                           params,
                           rng,
                           z_star,
                           None
                           )

    return z_star
