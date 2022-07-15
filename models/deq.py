import jax.numpy as jnp
from jax import grad
from typing import Callable

from .rootfind import rootfind, rootfind_grad


def deq(params: dict, solver: int, mode: int, rng, x: jnp.ndarray, f: Callable, max_iter: int, *args) -> jnp.ndarray:
    # Define the equilibrium point
    # g(z)=f(z)-z
    def g(params, rng, x):
        return f(params, rng, x) - x

    # Analytically find the root using one
    # of the following solvers:
    # - Broyden
    # - Anderson
    z_star_est = rootfind(g,
                          max_iter,
                          solver,
                          mode,
                          params,
                          rng,
                          x,
                          None
                          )

    # TODO
    # Why are we re-calculating this?
    # Isn't it supposed to be the same?
    z_star = f(params, rng, z_star_est)

    # Calculate the trace of the Jacobian using hte Hutchinson Estimator here
    # The trace with respect to the function doesnt need to be calculated again
    # since we already have an estimate from rootfind_grad
    # temp = grad(z_star, z_star_est)
    # print(temp)
    # TODO: This shouldn't happen during evaluation
    # as the gradient isn't required then.
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
