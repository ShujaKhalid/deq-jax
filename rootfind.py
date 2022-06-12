from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax
from broyden import broyden
from anderson import AndersonAcceleration as anderson

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
def rootfind_fwd(fun: Callable, max_iter: int, params: dict, rng: jnp.ndarray, x: jnp.ndarray, *args):
    z_star = rootfind(fun, max_iter, params, rng, x, *args)
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    # primal: z_star
    # residual: (params, rng, z_star, *args)
    return z_star, (params, rng, z_star, *args)

# Just padd back the gradient and dont do anything special
def dumb_bwd(fun, max_iter, res, grad):
    (params, rng, z_star, *args) = res
    # passed back gradient via d/dx and return nothing to other params
    arg_grads = tuple([None for _ in args])
    return_tuple = (None, None, grad, *arg_grads)
    return return_tuple
# =====================================================================================================


# ==================================== rootfind_grad ==================================================
def dumb_fwd(fun: Callable, max_iter: int, params: dict, rng, x: jnp.ndarray, *args):
    return x, (params, rng, x, *args)
    
def rootfind_bwd(fun, max_iter, res, grad):
    # returns dl/dz_star * J^(-1)_{g}
    (params, rng, z_star, *args) = res
    (_, vjp_fun) = jax.vjp(fun, params, rng, z_star, *args)

    def h_fun(x):
        #  J^(-1)_{g} x^T + (dl/dz_star)^T
        (JTp, JTr, JTx, *_) = vjp_fun(x)
        return JTx + grad

    eps = 2e-10 * jnp.sqrt(grad.size)
    dl_df_est = jnp.zeros_like(grad)

    # Iteratively calculate an estimate for this function instead of solving it analytically
    result_info = broyden(h_fun, dl_df_est, max_iter, eps)
    dl_df_est = result_info['result']

    # passed back gradient via d/dx and return nothing to other params
    arg_grads = tuple([None for _ in args])
    return_tuple = (None, None, dl_df_est, *arg_grads)
    return return_tuple
# ====================================================================================================

@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def rootfind(g: Callable, max_iter: int, params: dict, rng: jnp.ndarray, x: jnp.ndarray, solver: str, *args):
    eps = 1e-6 * jnp.sqrt(x.size)
    g = partial(g, params, rng)
    # Dont include during gradient calculations
    if (solver=='anderson'):
        result_info = jax.lax.stop_gradient(
            broyden(fun, x, max_iter, eps, *args)
        ) 
    elif (solver=='broyden'):
        result_info = jax.lax.stop_gradient(
            anderson(fun, history_size=5, maxiter=10*1000, ridge=1e-6, tol=tol).run(x)
        )
    return result_info['result']


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def rootfind_grad(fun: Callable, max_iter: int, params: dict, rng, x: jnp.ndarray, solver: str, *args):
    eps = 1e-6 * jnp.sqrt(x.size)
    fun = partial(fun, params, rng)
    if (solver=='anderson'):
        result_info = jax.lax.stop_gradient(
            broyden(fun, x, max_iter, eps, *args)
        ) 
    elif (solver=='broyden'):
        result_info = jax.lax.stop_gradient(
            anderson(fun, history_size=5, maxiter=10*1000, ridge=1e-6, tol=tol).run(x)
        )
    return result_info['result']


rootfind.defvjp(rootfind_fwd, dumb_bwd)
rootfind_grad.defvjp(dumb_fwd, rootfind_bwd)
