from pickletools import uint8
from models.transformer_cv import HeadDepth
from models.transformer_cv import HeadSeg
import jax
import numpy as np
import haiku as hk
import pandas as pd
from tqdm import tqdm
import jax.numpy as jnp
from torch.utils import data
from tabulate import tabulate
from termcolor import cprint
from models.deq import deq
from PIL import Image
import functools


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


def evaluate_cls(rng, state, epoch, config, ds_dict, preproc, accuracy):
    eval_trn = []
    eval_tst = []
    log_policy = eval(config["log_policy"])
    if ("train" in log_policy):
        for i, (x, y) in enumerate(tqdm(ds_dict['dl_trn'])):
            x = preproc(x, config)
            train_acc = accuracy(state['params'],
                                 rng,
                                 x,
                                 jax.nn.one_hot(y, config["num_classes"]))
            eval_trn.append(train_acc)
    if ("valid" in log_policy):
        for i, (x, y) in enumerate(tqdm(ds_dict['dl_tst'])):
            x = preproc(x, config)
            test_acc = accuracy(state['params'],
                                rng,
                                x,
                                jax.nn.one_hot(y, config["num_classes"]))
            eval_tst.append(test_acc)
            print("epoch: {} - iter: {} - acc_trn {:.2f} - acc_tst: {:.2f}".format(epoch, i,
                                                                                   np.mean(eval_trn), np.mean(eval_tst)))


def evaluate_seg(rng, state, epoch, config, ds_dict, preproc, jaccard):
    eval_trn = []
    eval_tst = []
    log_policy = eval(config["log_policy"])
    if ("train" in log_policy):
        for i, (x, y) in enumerate(tqdm(ds_dict['dl_trn'])):
            x = preproc(x, config)
            train_jac = jaccard(state['params'],
                                rng,
                                x,
                                jax.nn.one_hot(y, config["num_classes"]))
            eval_trn.append(train_jac)
    if ("valid" in log_policy):
        for i, (x, y) in enumerate(tqdm(ds_dict['dl_tst'])):
            # print("x (before preproc): {}".format(x))
            # print("y: {}".format(y))
            x_patch = jnp.array(preproc(x, config))
            # print("np.unique(y): {}".format(np.unique(y)))
            test_jac = jaccard(state['params'],
                               rng,
                               x_patch,
                               x,
                               jax.nn.one_hot(y, config["num_classes"]),
                               functools.partial(save_img_to_folder, i))
            eval_tst.append(test_jac)
        print("epoch: {} - iter: {} - jac_trn {:.2f} - jac_tst: {:.2f}".format(epoch, i,
                                                                               np.mean(eval_trn), np.mean(eval_tst)))


def save_img_to_folder(i, config, x, y, y_hat):
    save_loc = config["checkpoint_dir"]
    #print("Saving to {}".format(save_loc+str(i)+"_pred.png"))

    img_orig = (x[-1, :, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)
    img_orig = Image.fromarray(img_orig)
    img_orig.save(save_loc+str(i)+"_orig.png")

    # Print the results out class by class
    for j in range(y.shape[-1]):
        img_seg = (np.asarray(y[-1, :, :, j]) * 255).astype(np.uint8)
        img_seg = Image.fromarray(img_seg)
        img_seg.save(save_loc+str(i)+"_"+str(j)+"_seg.png")

        img_pred = (np.asarray(y_hat[-1, :, :, j]) * 255).astype(np.uint8)
        img_pred = Image.fromarray(img_pred)
        img_pred.save(save_loc+str(i)+"_"+str(j)+"_pred.png")

    return


def run(config, x, model, input_mask, max_iter=10):
    '''
    gen_stats:
    max_iter = 10
    solver = 0  # 0: Broyden ; 1: Anderson ; 2: secant
    '''
    if (config["mode"] == 'text'):
        rng = hk.next_rng_key()
        params = hk.experimental.lift(
            model.init)(rng, x, input_mask, is_training=True)
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
                max_iter,
                input_mask
            )
        else:
            z_star = model.apply(params,
                                 rng,
                                 x,
                                 input_mask,
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
                max_iter
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
            model.init)(rng, x, is_training=True)
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
                max_iter
            )
        else:
            z_star = model.apply(params,
                                 None,
                                 x,
                                 is_training=True)
    elif (config["mode"] == 'seg'):
        # params, state = model.init(hk.next_rng_key(), x, is_training=True)
        rng = hk.next_rng_key()
        params = hk.experimental.lift(
            model.init)(rng, x, is_training=True)
        if (config["deq_flag"] == "True"):
            # Define a callable function for ease of access downstream
            def f(params, state, rng, x):
                return model.apply(params, state, rng, x)

            z_star = deq(
                params,
                config["solver"] == "",
                0 if config["mode"] == "text" else 1,
                hk.next_rng_key(),
                x,
                f,
                max_iter
            )
        else:
            z_star = model.apply(params,
                                 None,
                                 x,
                                 is_training=True)

        # Get the segmentation head
        def seg_head_fn(x, config):
            return get_outputs(x, config)
        seg_head_fn = hk.transform(seg_head_fn)
        params = hk.experimental.lift(
            seg_head_fn.init)(rng, z_star, config)
        z_star = seg_head_fn.apply(params, rng, z_star, config)

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
    elif (solver == 0 and mode == 1):
        from solvers.broyden_cv import broyden as find_root
    elif (solver == 1 and mode == 0):
        from solvers.anderson import AndersonAcceleration as find_root
    elif (solver == 1 and mode == 1):
        from solvers.anderson import AndersonAcceleration as find_root
    else:
        raise Exception('Invalid solver/mode combination')

    result_info = jax.lax.stop_gradient(
        find_root(fun, x, max_iter, eps, *args)
    )['result']

    return result_info


def patchify(patch_size, x):
    bsz, cnl, hgt, wdt = x.shape
    patches_qty = (hgt*wdt)//(patch_size * patch_size)
    patches_dim = cnl*patch_size**2
    patches = x.reshape(bsz, patches_qty, patches_dim)
    # print("patches.shape: {}".format(patches.shape))
    # print("self.embed_pos.shape: {}".format(self.embed_pos.shape))
    return patches


def unpatchify(patch_size, patches):
    # patches = patches[:, :-1, :]
    # bsz, patches_qty, patches_dim = patches.shape
    # hgt = wdt = int(np.sqrt(patches_qty * (patch_size * patch_size)))
    # #cnl = patches_dim / patch_size**2
    # cnl = (patches_qty * patches_dim) // (hgt * wdt)
    # x = patches.reshape(bsz, cnl, hgt, wdt).transpose(0, 2, 3, 1)
    patches = patches[:, :, :]
    bsz, patches_qty, patches_dim = patches.shape
    # TODO: arbitrarily set - specifically for Cityscapes...
    base_factor = 2
    hgt = wdt = int(np.sqrt(patches_dim)) * base_factor
    wdt = wdt * base_factor
    cnl = (patches_qty) // (4*base_factor)
    x = patches.reshape(bsz, cnl, hgt, wdt).transpose(0, 2, 3, 1)
    return x


def get_outputs(x, config):
    mode = config["mode"]
    resample_dim = config["model_attrs"]["resample_dim"]
    patch_size = config["model_attrs"]["patch_size"]
    num_classes = config["num_classes"]

    if (mode == "seg"):
        head_seg = HeadSeg(resample_dim, patch_size, num_classes)
        x = head_seg(x)
    elif (mode == "depth"):
        head_dep = HeadDepth(resample_dim, patch_size)
        x = head_dep(x)
    elif (mode == "cls"):
        # z_star = jnp.mean(z_star[:, 0])
        x = jnp.mean(x, axis=1)
        x = hk.Linear(resample_dim)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(num_classes)(x)
    else:
        raise Exception("get_outputs incorrectly selected")
    # elif (mode == "segdepth"):
    #     seg = head_seg(x)
    #     depth = depth(x)
    #     return seg, depth

    return x
