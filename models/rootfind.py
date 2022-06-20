# core
import jax
import jaxopt
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Callable

# solvers
from solvers.broyden import broyden
from solvers.anderson import AndersonAcceleration as anderson
#from tensorflow_probability.substrates.jax.math import secant_root as secant
from solvers.secant import find_root_secant as secant

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


def rootfind_fwd(fun: Callable, max_iter: int, solver: int, params: dict, rng: jnp.ndarray, x: jnp.ndarray, *args):
    z_star = rootfind(fun, max_iter, solver, params, rng, x, *args)
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    # primal: z_star
    # residual: (params, rng, z_star, *args)
    return z_star, (params, rng, z_star, *args)

# Just padd back the gradient and dont do anything special


def dumb_bwd(fun, max_iter, solver, res, grad):
    (params, rng, z_star, *args) = res
    # passed back gradient via d/dx and return nothing to other params
    arg_grads = tuple([None for _ in args])
    return_tuple = (None, None, grad, *arg_grads)
    return return_tuple
# =====================================================================================================


# ==================================== rootfind_grad ==================================================
def dumb_fwd(fun: Callable, max_iter: int, solver: int, params: dict, rng, x: jnp.ndarray, *args):
    return x, (params, rng, x, *args)


def rootfind_bwd(fun, max_iter, solver, res, grad):
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
    # Your estimate of the gradient through the solver
    dl_df_est = result_info['result']

    # passed back gradient via d/dx and return nothing to other params
    arg_grads = tuple([None for _ in args])
    return_tuple = (None, None, dl_df_est, *arg_grads)
    return return_tuple
# ====================================================================================================


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def rootfind(g: Callable, max_iter: int, solver: int, params: dict, rng: jnp.ndarray, x: jnp.ndarray, *args):
    eps = 1e-6 * jnp.sqrt(x.size)
    fun = partial(g, params, rng)
    # Dont include during gradient calculations
    if (solver == 0):
        result_info = jax.lax.stop_gradient(
            broyden(fun, x, max_iter, eps, *args)
        )['result']
    elif (solver == 1):
        # print("x: {}".format(x))
        # print("x.shape: {}".format(x.shape))
        result_info, state = jax.lax.stop_gradient(
            anderson(fun, history_size=5, maxiter=max_iter,
                     ridge=1e-6, tol=eps).run(x, *args)
        )
    elif (solver == 2):
        pytree_init = {'x1': np.zeros((2, 64, 128))}
        pytree_init_args = {'b1': np.zeros((2, 64))}
        # print(pytree_init)
        # print(pytree_init_args)
        result_info = jax.lax.stop_gradient(
            #secant(objective_fn=fun, initial_position=x)
            # TODO look at the format of PyTrees
            jaxopt.ScipyRootFinding(method='anderson', optimality_fun=fun, jit=False,
                                    tol=eps, has_aux=True).run(pytree_init, pytree_init_args)
        )
        # print(result_info)
    else:
        print('SOLVER not provided (rootfind)...')
    return result_info


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def rootfind_grad(g: Callable, max_iter: int, solver: int, params: dict, rng, x: jnp.ndarray, *args):
    eps = 1e-6 * jnp.sqrt(x.size)
    fun = partial(g, params, rng)
    if (solver == 0):
        result_info = jax.lax.stop_gradient(
            broyden(fun, x, max_iter, eps, *args)
        )['result']
    elif (solver == 1):
        result_info, state = jax.lax.stop_gradient(
            anderson(fun, history_size=5, maxiter=max_iter,
                     ridge=1e-4, tol=eps).run(jnp.zeros(1), *args)
        )
    elif (solver == 2):
        result_info = jax.lax.stop_gradient(
            secant(objective_fn=fun, initial_position=x)
        )
        print(result_info)
    else:
        print('SOLVER not provided (rootfind_grad)...')
    return result_info


rootfind.defvjp(rootfind_fwd, dumb_bwd)
rootfind_grad.defvjp(dumb_fwd, rootfind_bwd)
