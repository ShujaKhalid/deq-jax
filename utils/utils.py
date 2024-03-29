from re import X
import jax
import numpy as np
import haiku as hk
import pandas as pd
import jax.numpy as jnp

from torch.utils import data
from termcolor import cprint
from models.deq import deq
from PIL import Image
from einops import rearrange


from models.architectures.mdeqformer import Patchify


class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


# def tabulate(d, headers):
#     return (tabulate([(k,) + v for k, v in d.items()], headers=headers))
def logger(data, order):
    # custom logging
    lng = len(order)
    for i, key in enumerate(order):
        msg = str(key) + ': ' + str(data[key]).zfill(6)
        cprint(msg, 'green', attrs=['bold'], end='\n') if i == lng-1 else cprint(
            msg + '  --- ', 'green', attrs=['bold'], end=' ')


def save_img_to_folder(epoch, i, config, x, y, y_hat, ver):
    save_loc = config["checkpoint_dir"]
    # print("Saving to {}".format(save_loc+str(i)+"_pred.png"))
    img_orig = (x[-1, :, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)
    img_orig = Image.fromarray(img_orig)
    img_orig.save(save_loc+str(ver)+str(epoch)+"_orig.png")

    # Print the results out class by class
    if (len(y.shape) == 4 and epoch % config["logging"]["save_imgs_step"] == 0):
        for j in range(y.shape[-1]):
            img_seg = (np.asarray(y[-1, :, :, j]) * 255).astype(np.uint8)
            # print("y[-1, :, :, j]: {}".format(np.unique(y[-1, :, :, j])))
            # print(
            #     "y[-1, :, :, j] * 255: {}".format(np.unique(y[-1, :, :, j] * 255)))
            img_seg = Image.fromarray(img_seg)
            img_seg.save(save_loc+str(ver)+"_" +
                         str(epoch)+"_"+str(j)+"_seg.png")

            img_pred = (np.asarray(y_hat[-1, :, :, j]) * 255).astype(np.uint8)
            # print("y_hat[-1, :, :, j]: {}".format(np.unique(y_hat[-1, :, :, j])))
            # print(
            #     "y_hat[-1, :, :, j] * 255: {}".format(np.unique(y_hat[-1, :, :, j] * 255)))
            img_pred = Image.fromarray(img_pred)
            img_pred.save(save_loc+str(ver)+"_" +
                          str(epoch)+"_"+str(j)+"_pred.png")
    elif (len(y.shape) == 3):
        img_seg = (np.asarray(y[-1, :, :]) * 255).astype(np.uint8)
        img_seg = Image.fromarray(img_seg)
        img_seg.save(save_loc+str(ver)+"_"+str(epoch)+"_seg.png")

        img_pred = (np.asarray(y_hat[-1, :, :]) * 255).astype(np.uint8)
        img_pred = Image.fromarray(img_pred)
        img_pred.save(save_loc+str(ver)+"_"+str(epoch)+"_pred.png")

    return


def run(config, x, model):
    '''
    gen_stats:
    max_iter = 10
    solver = 0  # 0: Broyden ; 1: Anderson ; 2: secant
    '''
    if (config["mode"] == 'text'):
        rng = hk.next_rng_key()
        params = hk.experimental.lift(
            model.init)(rng, x, is_training=True)
        # Define a callable function for ease of access downstream
        if (config["deq_flag"] == "True"):
            def f(params, rng, x, input_mask):
                return model.apply(params, rng, x, input_mask, is_training=True)

            z_star = deq(
                params,
                config["solver"],
                0 if config["mode"] == "text" else 1,
                hk.next_rng_key(),
                x,
                f,
                max_iter=config["deq_attrs"]["max_iter"])
        else:
            z_star = model.apply(params,
                                 rng,
                                 x,
                                 is_training=True)
    elif (config["mode"] == 'cls'):
        # params, state = model.init(hk.next_rng_key(), x, is_training=True)
        rng = hk.next_rng_key()
        params_and_state_fn, updater = hk.experimental.lift_with_state(
            model.init)
        params, state = params_and_state_fn(rng, x, is_training=True)
        if (config["deq_flag"] == "True"):
            # Define a callable function for ease of access downstream
            def f(params, state, rng, x):
                return model.apply(params, state, rng, x)

            z_star = deq(
                params,
                config["solver"],
                0 if config["mode"] == "text" else 1,
                hk.next_rng_key(),
                x,
                f,
                max_iter=config["deq_attrs"]["max_iter"]
            )
        else:
            z_star, state = model.apply(params,
                                        state,
                                        None,
                                        x,
                                        is_training=True)
    elif (config["mode"] == 'cls_trans'):
        # params, state = model.init(hk.next_rng_key(), x, is_training=True)
        rng = hk.next_rng_key()
        params = hk.experimental.lift(
            model.init)(rng, x)
        if (config["deq_attrs"]["deq_flag"] == "True"):
            # Define a callable function for ease of access downstream
            def f(params, rng, x):
                return model.apply(params, rng, x)
            # print("\nRUNNING DEQ...\n")
            z_star = deq(
                params,
                config["deq_attrs"]["solver"],
                0 if config["mode"] == "text" else 1,
                hk.next_rng_key(),
                x,
                f,
                max_iter=config["deq_attrs"]["max_iter"]
            )
        else:
            z_star = model.apply(params,
                                 None,
                                 x)
    elif (config["mode"] == 'seg'):
        # params, state = model.init(hk.next_rng_key(), x, is_training=True)
        rng = hk.next_rng_key()
        params = hk.experimental.lift(
            model.init)(rng, x)
        if (config["deq_attrs"]["deq_flag"] == "True"):
            # Define a callable function for ease of access downstream
            def f(params, rng, x):
                return model.apply(params, rng, x)
            print("\nRUNNING DEQ...\n")
            z_star = deq(
                params,
                config["deq_attrs"]["solver"],
                0 if config["mode"] == "text" else 1,
                hk.next_rng_key(),
                x,
                f,
                max_iter=config["deq_attrs"]["max_iter"]
            )
        else:
            z_star = model.apply(params,
                                 None,
                                 x)

    # Get the prediction head
    def head_fn(x, config):
        return get_outputs(x, config)
    head_fn = hk.transform(head_fn)
    params = hk.experimental.lift(
        head_fn.init)(rng, z_star, config)
    z_star = head_fn.apply(params, rng, z_star, config)

    return z_star


def qnm(fun, x, max_iter, eps, solver, mode, *args):
    # solvers
    # from solvers.broyden_nlp import broyden
    # from solvers.broyden_cv import broyden
    # from tensorflow_probability.substrates.jax.math import secant_root as secant

    # Choose solver type (cv vc. nlp) # unify later
    # 0: "text"
    if (solver == 0 and mode == 0):
        from solvers.broyden_nlp import broyden as find_root
        result_info = jax.lax.stop_gradient(
            find_root(fun, x, max_iter, eps, *args)
        )['result']
    elif (solver == 0 and mode == 1):
        from solvers.broyden_cv import broyden as find_root
        result_info = jax.lax.stop_gradient(
            find_root(fun, x, max_iter, eps, *args)
        )
    elif (solver == 1 and mode == 0):
        from jaxopt import AndersonAcceleration as find_root
        result_info = jax.lax.stop_gradient(
            find_root(fun, history_size=5, maxiter=max_iter,
                      tol=eps).run(x)[0]
        )
    elif (solver == 1 and mode == 1):
        from jaxopt import AndersonAcceleration as find_root
        result_info = jax.lax.stop_gradient(
            find_root(fun, history_size=5, maxiter=max_iter,
                      tol=eps).run(x)[0]
        )
    elif (solver == 2 and mode == 0):
        from jaxopt import ScipyRootFinding as find_root
        result_info = jax.lax.stop_gradient(
            find_root(method='anderson',
                      optimality_fun=fun,
                      tol=eps).run(x)[0]
        )
    elif (solver == 2 and mode == 1):
        from jaxopt import ScipyRootFinding as find_root
        result_info = jax.lax.stop_gradient(
            find_root(
                optimality_fun=fun,
                tol=eps).run(x)
        )
        print("result_info: {}".format(result_info))
    elif (solver == 3 and mode == 0):
        from jaxopt import solve_normal_cg as find_root
    elif (solver == 3 and mode == 1):
        from jaxopt import solve_normal_cg as find_root
    else:
        raise Exception('Invalid solver/mode combination')

    return result_info


def preproc(x, config):
    # x = np.expand_dims(x, axis=3) if len(x.shape) == 3 else x
    # # TODO: fix
    # x = np.repeat(x, 3, axis=3) if x.shape[3] == 1 else x
    if (x.shape[-1] == 3):
        # shift c axis to the end
        # [B, C, H, W] -> [B, H, W, C]
        x = np.transpose(x, (0, 2, 3, 1))
    # print(x.shape)
    # print(x.min())
    # print(x.max())
    # save_loc = config["checkpoint_dir"]
    # for i in range(x.shape[0]):
    #     img_orig = (x[i, :, :, :] * 255).astype(np.uint8)
    #     img_orig = Image.fromarray(img_orig)
    #     img_orig.save(save_loc+str(i)+str(i)+"_orig.png")

    # Change the format of the data
    # from img -> img_patch
    # patch_size = config["model_attrs"]["cv"]["patch_size"]
    # patch_scales = eval(config["model_attrs"]["cv"]["scales"])

    # Get the segmentation head
    def patchify_fn(config, x):
        patchify = Patchify(config)
        return patchify(x)

    rng = jax.random.PRNGKey(999)  # FIXME I dont thhink this is right...

    init, apply = hk.transform(patchify_fn)
    params = init(rng, config, x)
    patched = apply(params, rng, config, x)

    return patched


def unpatchify(patches, config):
    # patches = patches[:, :-1, :]
    # bsz, patches_qty, patches_dim = patches.shape
    # hgt = wdt = int(np.sqrt(patches_qty * (patch_size * patch_size)))
    # #cnl = patches_dim / patch_size**2
    # cnl = (patches_qty * patches_dim) // (hgt * wdt)
    # x = patches.reshape(bsz, cnl, hgt, wdt).transpose(0, 2, 3, 1)
    patches_size = config["model_attrs"]["cv"]["patch_size"]
    _, patches_qty, _ = patches.shape
    h = w = int(np.sqrt(patches_qty))
    if (config["data_attrs"]["dataset"] == "VOCSegmentation"):
        x = rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      h=h, w=w, p1=patches_size, p2=patches_size).transpose(0, 2, 3, 1)
    elif (config["data_attrs"]["dataset"] == "Cityscapes"):
        x = rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      h=h, w=w, p1=patches_size, p2=patches_size).transpose(0, 2, 3, 1)
    else:
        raise Exception("check unpatchify...")

        # patches = patches[:, :, :]
        # bsz, patches_qty, patches_dim = patches.shape
        # base_factor = 1
        # hgt = wdt = int(np.sqrt(patches_dim)) * base_factor
        # wdt = wdt * base_factor
        # cnl = (patches_qty) // (2**(base_factor+1))
        # d_new = (cnl) * (2**(base_factor+1))
        # x = patches[:bsz, :d_new, :patches_dim].reshape(
        #     bsz, cnl, hgt, wdt).transpose(0, 2, 3, 1)

    return x


def get_outputs(x, config):
    '''
    Exclusive to CV - NLP models wont come here
    '''
    mode = config["mode"]
    resample_dim = config["model_attrs"]["cv"]["resample_dim"] if mode != "text" else config["model_attrs"]["lm"]["resample_dim"]
    patch_size = config["model_attrs"]["cv"]["patch_size"]
    num_classes = config["data_attrs"]["num_classes"]

    if (config["model_attrs"]["cv"]["arch"] == "deqformer"):
        from models.architectures.deqformer import HeadSeg
        from models.architectures.deqformer import HeadDepth
    elif (config["model_attrs"]["cv"]["arch"] == "mdeqformer"):
        from models.architectures.mdeqformer import HeadSeg
        from models.architectures.mdeqformer import HeadDepth

    if (mode == "seg"):
        head_seg = HeadSeg(x.shape, resample_dim,
                           patch_size, config, num_classes)
        x = head_seg(x)
    elif (mode == "depth"):
        head_dep = HeadDepth(resample_dim, patch_size)
        x = head_dep(x)
    elif (mode == "cls" or mode == "cls_trans"):
        x = jnp.mean(x, axis=1)
        #x = x[:, 0]
        # x = hk.Linear(resample_dim)(x)
        # x = jax.nn.gelu(x)
        x = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True)(x)
        x = hk.Linear(num_classes)(x)
    else:
        raise Exception("get_outputs incorrectly selected")

    return x
