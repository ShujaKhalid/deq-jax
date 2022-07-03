# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Train a transformer for language modeling on a small text dataset.

This example serves to demonstrate:
  - A clean Haiku transformer implementation.
  - An example minimal training loop around it.

This example runs on ASCII text files.
We have not tuned the hyperparameters at all.

Example, using Karpathy's tiny_shakespeare dataset:
$ wget -O /tmp/shakespeare.txt \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
$ python3 examples/transformer/train.py --dataset_path=/tmp/shakespeare.txt

Note: Run with --alsologtostderr to see outputs.
"""

import os
from telnetlib import X3PAD
import time
import json
import pickle
import argparse
import functools
from tqdm import tqdm
from tkinter import X
from utils.evaluate import evaluate_cls, evaluate_seg
from typing import Any, Mapping


from absl import logging

# from examples.transformer import dataset
# from examples.transformer import model
from utils.forward import Forward
from utils.losses import Losses
from utils.datasets import Datasets
from utils.metrics import jaccard, accuracy

import jax
import optax
import haiku as hk
import numpy as np
import jax.numpy as jnp
import utils.dataset as dataset
from utils.utils import logger
from utils.utils import preproc


class Updater:
    """A stateless abstraction around an init_fn/update_fn pair.

    This extracts some common boilerplate from the training loop.
    """

    def __init__(self, net_init, loss_fn,
                 optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, rng, data):
        """Initializes state of the updater."""
        out_rng, init_rng = jax.random.split(rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data: Mapping[str, jnp.ndarray]):
        """Updates the state using some data and returns metrics."""
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)

        updates, opt_state = self._opt.update(g, state['opt_state'])
        params = optax.apply_updates(params, updates)

        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
        }

        metrics = {
            'step': state['step'],
            'loss': loss,
        }
        return new_state, metrics


class CheckpointingUpdater:
    """A didactic checkpointing wrapper around an Updater.

    A more mature checkpointing implementation might:
      - Use np.savez() to store the core data instead of pickle.
      - Not block JAX async dispatch.
      - Automatically garbage collect old checkpoints.
    """

    def __init__(self,
                 inner: Updater,
                 checkpoint_dir: str,
                 checkpoint_every_n: int = 10000):
        self._inner = inner
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every_n = checkpoint_every_n

    def _checkpoint_paths(self):
        return [p for p in os.listdir(self._checkpoint_dir) if 'checkpoint_' in p]

    def init(self, rng, data):
        """Initialize experiment state."""
        if not os.path.exists(self._checkpoint_dir) or not self._checkpoint_paths():
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            # Save a snapshot of the code to the checkpoint directory
            os.system("cp -pr ./models ./solvers ./utils " +
                      config["checkpoint_dir"])
            return self._inner.init(rng, data)
        else:
            checkpoint = os.path.join(self._checkpoint_dir,
                                      max(self._checkpoint_paths()))
            logging.info('Loading checkpoint from %s', checkpoint)
            with open(checkpoint, 'rb') as f:
                state = pickle.load(f)
            return state

    def update(self, state, data):
        """Update experiment state."""
        # NOTE: This blocks until `state` is computed. If you want to use JAX async
        # dispatch, maintain state['step'] as a NumPy scalar instead of a JAX array.
        # Context: https://jax.readthedocs.io/en/latest/async_dispatch.html
        step = np.array(state['step'])
        if step % self._checkpoint_every_n == 0:
            path = os.path.join(self._checkpoint_dir,
                                'checkpoint_{:07d}.pkl'.format(step))
            checkpoint_state = jax.device_get(state)
            logging.info('Serializing experiment state to %s', path)
            # TODO: dont add to the checkpoint
            # with open(path, 'wb') as f:
            #     pickle.dump(checkpoint_state, f)

        state, out = self._inner.update(state, data)
        return state, out


def main(config):

    config["checkpoint_dir"] = config["logging"]["logdir"] + \
        str(int(time.time())) + "/"

    # Get the dataset in the required format
    data = Datasets(config)
    ds_dict = data.get_datasets()
    print(ds_dict['ds_trn'])
    print(ds_dict['ds_tst'])

    # Setup the forward pass
    build_forward_fn = Forward(config).build_forward_fn
    forward_fn = hk.transform(build_forward_fn())

    # Load loss function
    loss_fn = Losses(config, forward_fn).get_loss_fn()

    # Define optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config["opt_attrs"]["grad_clip_value"]),
        optax.adam(config["opt_attrs"]["learning_rate"],
                   b1=config["opt_attrs"]["b1"],
                   b2=config["opt_attrs"]["b2"]))

    updater = Updater(forward_fn.init, loss_fn, optimizer)
    updater = CheckpointingUpdater(updater, config["checkpoint_dir"])

    # Initialize parameters.
    rng = jax.random.PRNGKey(428)

    prev_time = time.time()

    if (config["mode"] == 'text'):
        # TODO: Improve structure
        data = next(ds_dict['ds_trn'])
        state = updater.init(rng, data)
        for step in range(config["max_steps"]):
            data = next(ds_dict['ds_trn'])
            state, metrics = updater.update(state, data)
            # We use JAX runahead to mask data preprocessing and JAX dispatch overheads.
            # Using values from state/metrics too often will block the runahead and can
            # cause these overheads to become more prominent.
            if step % config["log_every"] == 0:
                steps_per_sec = config["log_every"] / (time.time() - prev_time)
                prev_time = time.time()
                metrics.update({'steps_per_sec': steps_per_sec})
                logger(metrics, order=[
                    'step',
                    'loss',
                    'steps_per_sec']
                )

    else:
        # Train the model
        for epoch in range(config["opt_attrs"]["epochs"]):
            for step, (x, y) in enumerate(ds_dict['dl_trn']):
                x = preproc(x, config)
                # print('x.shape: {}, y.shape: {}'.format(x.shape, y.shape))
                data = {'obs': x, 'target': y}
                if (step < config["opt_attrs"]["max_steps"]):
                    if (epoch == 0 and step == 0):
                        # Initialize state
                        state = updater.init(rng, data)

                    state, metrics = updater.update(state, data)

                    # Training logs
                    if step % config["logging"]["log_every"] == 0:
                        steps_per_sec = config["logging"]["log_every"] / \
                            (time.time() - prev_time)
                        prev_time = time.time()
                        metrics.update({'steps_per_sec': steps_per_sec})
                        logger(metrics, order=[
                            'step',
                            'loss',
                            'steps_per_sec'
                        ])
                else:
                    break

            # for j, k in ds_dict['dl_trn']:
            #     print(j)
            # ============================ Evaluation logs ===========================
            if (config["mode"] == "cls" or config["mode"] == "cls_trans"):
                evaluate_cls(rng, state, epoch, config,
                             ds_dict, preproc, functools.partial(accuracy, forward_fn=forward_fn))
            elif (config["mode"] == "seg"):
                evaluate_seg(rng, state, epoch, config,
                             ds_dict, preproc, functools.partial(jaccard, forward_fn=forward_fn, config=config))
            else:
                raise Exception(
                    "Incorrect mode selected... review configuration file")
            # ============================ Evaluation logs ===========================

    return "Complete!"


if __name__ == '__main__':
    # TODO add to config file

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--job_id', help='job id for the set of jobs in ./jobs')
    args = parser.parse_args()
    print(args)
    config = json.load(open(args.job_id, "r"))

    out = json.dumps(config, indent=4, sort_keys=True)
    print(out)
    main(config)
