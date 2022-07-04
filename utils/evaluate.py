import jax
import functools
from tqdm import tqdm
import jax.numpy as jnp
import numpy as np
from utils.utils import save_img_to_folder


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
                                 jax.nn.one_hot(y, config["data_attrs"]["num_classes"]))
            eval_trn.append(train_acc)
    if ("valid" in log_policy):
        for i, (x, y) in enumerate(tqdm(ds_dict['dl_tst'])):
            x = preproc(x, config)
            test_acc = accuracy(state['params'],
                                rng,
                                x,
                                jax.nn.one_hot(y, config["data_attrs"]["num_classes"]))
            eval_tst.append(test_acc)
            print("epoch: {} - iter: {} - acc_trn {:.2f} - acc_tst: {:.2f}".format(epoch, i,
                                                                                   np.mean(eval_trn), np.mean(eval_tst)))


def evaluate_seg(rng, state, epoch, config, ds_dict, preproc, jaccard):
    eval_trn = []
    eval_tst = []
    log_policy = eval(config["logging"]["log_policy"])
    if ("train" in log_policy):
        for i, (x, y) in enumerate(tqdm(ds_dict['dl_trn'])):
            x = preproc(x, config)
            train_jac = jaccard(state['params'],
                                rng,
                                x,
                                y)
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
                               y,
                               functools.partial(save_img_to_folder, i))
            eval_tst.append(test_jac)
        print("epoch: {} - iter: {} - jac_trn {:.2f} - jac_tst: {:.2f}".format(epoch, i,
                                                                               np.mean(eval_trn), np.mean(eval_tst)))
