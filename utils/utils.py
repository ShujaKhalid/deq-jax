import numpy as np
import haiku as hk
import pandas as pd
from torch.utils import data
from tabulate import tabulate
from termcolor import cprint
from models.deq import deq


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
        msg = key + ': ' + \
            str(data[key]).zfill(6) if type(
                data[key]) == int else str(data[key])
        cprint(msg, 'green', attrs=['bold'], end='\n') if i == lng-1 else cprint(
            msg + ' --- ', 'green', attrs=['bold'], end=' ')


def run(flag, mode, x, model, input_mask, max_iter=10, solver=0):
    '''
    gen_stats:
    max_iter = 10
    solver = 0  # 0: Broyden ; 1: Anderson ; 2: secant
    '''
    if (mode == 'text'):
        if (flag):
            params = hk.experimental.lift(
                model.init)(hk.next_rng_key(), x, input_mask, is_training=True)
           # Define a callable function for ease of access downstream

            def f(params, rng, x, input_mask):
                return model.apply(params, rng, x, input_mask, is_training=True)

            z_star = deq(
                params, hk.next_rng_key(), x, f, max_iter, solver, input_mask)
        else:
            z_star = model(
                x, input_mask, is_training=True)
    elif (mode == 'cls'):
        params, state = model.init(hk.next_rng_key(), x, is_training=True)
        if (flag):
            # Define a callable function for ease of access downstream
            def f(params, state, rng, x):
                return model.apply(params, state, rng, x)

            z_star = deq(
                params, hk.next_rng_key(), x, f, max_iter, solver
            )
            print("z_star.shape: {}".format(z_star.shape))
        else:
            z_star, state = model.apply(params, state, None,
                                        x, is_training=True)
            print("z_star.shape: {}".format(z_star.shape))

    elif (mode == 'seg'):
        params, state = model.init(hk.next_rng_key(), x, is_training=True)
        if (flag):
            # Define a callable function for ease of access downstream
            def f(params, state, rng, x):
                return model.apply(params, state, rng, x)

            z_star = deq(
                params, hk.next_rng_key(), x, f, max_iter, solver
            )
            print("z_star.shape: {}".format(z_star.shape))
        else:
            z_star, state = model.apply(params, state, None,
                                        x, is_training=True)
            print("z_star.shape: {}".format(z_star.shape))

    return z_star
