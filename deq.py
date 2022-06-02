import jax.numpy as jnp
from typing import Callable

from rootfind import rootfind
#import rootfind_grad


def deq(f: Callable, x: jnp.ndarray, *args) -> jnp.ndarray:
    
    # Define the equilibrium point
    # g(z)=f(z)-z
    def g(x, *args): return f(x, *args) - x

    # Analytically find the root using one 
    # of the following solvers:
    # - Broyden
    # - Anderson
    z_star = rootfind(g, x, *args)

    # TODO
    
    return z_star


