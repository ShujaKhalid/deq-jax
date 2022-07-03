# core
import jax
import jaxopt
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Callable
import utils.utils as u

# A beautiful guide here for reference
# https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#gradient-clipping

'''
Here we define custom functions to calculate the gradient of the `rootfind` function.
- custom fwd: 
    - {rootfind_fwd}
        - Set it such that the boryden method isn't included in gradient calculations.
          We don't want to back-prop through the while-loop, would prefer to use implicit function theorem instead.
        - return primal output and residuals of the forward pass
- custom bwd:
    - {dumb_bwd}
        - 
'''

# ===================================== rootfind ======================================================
# Do a forward pass where  you actually find z_star


def rootfind_fwd(fun: Callable, max_iter: int, solver: int, mode: int, params: dict, rng: jnp.ndarray, x: jnp.ndarray, *args):
    z_star = rootfind(fun, max_iter, solver, mode, params, rng, x, *args)
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    # primal: z_star
    # residual: (params, rng, z_star, *args)
    return z_star, (params, rng, z_star, *args)

# Just padd back the gradient and dont do anything special


def dumb_bwd(fun, max_iter, solver, mode, res, grad):
    (params, rng, z_star, *args) = res
    # passed back gradient via d/dx and return nothing to other params
    arg_grads = tuple([None for _ in args])
    return_tuple = (None, None, grad, *arg_grads)
    return return_tuple
# =====================================================================================================


# ==================================== rootfind_grad ==================================================
def dumb_fwd(fun: Callable, max_iter: int, solver: int, mode: int, params: dict, rng, x: jnp.ndarray, *args):
    return x, (params, rng, x, *args)


def rootfind_bwd(fun, max_iter, solver, mode, res, grad):
    # returns dl/dz_star * J^(-1)_{g}
    (params, rng, z_star, *args) = res
    (_, vjp_fun) = jax.vjp(fun, params, rng, z_star)

    def h_fun(x, args=None):
        #  J^(-1)_{g} x^T + (dl/dz_star)^T
        (JTp, JTr, JTx, *_) = vjp_fun(x)
        return JTx + grad

    eps = 2e-10 * jnp.sqrt(grad.size)
    dl_df_est = jnp.zeros_like(grad)

    # Iteratively calculate an estimate for this function instead of solving it analytically
    result_info = u.qnm(h_fun, dl_df_est, max_iter, eps, solver, mode)
    # Your estimate of the gradient through the solver
    dl_df_est = result_info

    # passed back gradient via d/dx and return nothing to other params
    arg_grads = tuple([None for _ in args])
    return_tuple = (None, None, dl_df_est, *arg_grads)
    return return_tuple
# ====================================================================================================


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def rootfind(g: Callable, max_iter: int, solver: int, mode: int, params: dict, rng: jnp.ndarray, x: jnp.ndarray, *args):
    eps = 1e-6 * jnp.sqrt(x.size)
    fun = partial(g, params, rng)

    result_info = u.qnm(fun, x, max_iter, eps, solver, mode, *args)

    return result_info


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def rootfind_grad(g: Callable, max_iter: int, solver: int, mode: int, params: dict, rng, x: jnp.ndarray, *args):
    eps = 1e-6 * jnp.sqrt(x.size)
    fun = partial(g, params, rng)

    result_info = u.qnm(fun, x, max_iter, eps, solver, mode, *args)

    return result_info


rootfind.defvjp(rootfind_fwd, dumb_bwd)
rootfind_grad.defvjp(dumb_fwd, rootfind_bwd)
