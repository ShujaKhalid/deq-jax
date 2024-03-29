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
from utils.metrics import seg_metrics, cls_metrics

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
                 config: dict,
                 checkpoint_dir: str,
                 checkpoint_every_n: int = 1000,
                 num_classes: int = 10):
        self._inner = inner
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every_n = checkpoint_every_n
        self.num_classes = num_classes
        self.config = config

    def _checkpoint_paths(self):
        return [p for p in os.listdir(self._checkpoint_dir) if 'checkpoint_' in p]

    def init(self, rng, data):
        """Initialize experiment state."""
        if not os.path.exists(self._checkpoint_dir) or not self._checkpoint_paths():
            os.makedirs(self._checkpoint_dir, exist_ok=True)

            return self._inner.init(rng, data)
        else:
            checkpoint = os.path.join(self._checkpoint_dir,
                                      max(self._checkpoint_paths()))
            logging.info('Loading checkpoint from %s', checkpoint)
            with open(checkpoint, 'rb') as f:
                state = pickle.load(f)

            if (self.config["model_attrs"]["pretrained"] == "True"):
                hgt = wdt = 128
                patch_size = self.config["model_attrs"]["cv"]["patch_size"]
                patch_qty = (hgt*wdt)//(patch_size**2)
                resample_dim = self.config["model_attrs"]["cv"]["resample_dim"]
                # embedding = np.random.normal(
                #    0, 1, size=(1, patch_qty+1, resample_dim))
                embedding = np.ones((1, patch_qty+1, resample_dim))

                state["params"]["lifted/transformer"]["embed_pos"] = embedding
                state["opt_state"][1][0][1]["lifted/transformer"]["embed_pos"] = embedding
                state["opt_state"][1][0][2]["lifted/transformer"]["embed_pos"] = embedding
                # FIXME - find a better and cleaner way to do this...
                # Add gaussian initialization?
                if (self.config["mode"] == "cls_trans"):
                    state["params"]["lifted_1/linear"]["w"] = state["params"]["lifted_1/linear"]["w"][:, :self.num_classes]
                    state["params"]["lifted_1/linear"]["b"] = state["params"]["lifted_1/linear"]["b"][:self.num_classes]
                    state["opt_state"][1][0][1]["lifted_1/linear"]["w"] = state["opt_state"][1][0][1]["lifted_1/linear"]["w"][:, :self.num_classes]
                    state["opt_state"][1][0][1]["lifted_1/linear"]["b"] = state["opt_state"][1][0][1]["lifted_1/linear"]["b"][:self.num_classes]
                    state["opt_state"][1][0][2]["lifted_1/linear"]["w"] = state["opt_state"][1][0][2]["lifted_1/linear"]["w"][:, :self.num_classes]
                    state["opt_state"][1][0][2]["lifted_1/linear"]["b"] = state["opt_state"][1][0][2]["lifted_1/linear"]["b"][:self.num_classes]
                elif (self.config["mode"] == "seg"):
                    # Initialize parameters
                    state["params"]["lifted_1/head_seg/~/conv2_d"] = {
                        "w": 0, "b": 0}
                    state["params"]["lifted_1/head_seg/~/conv2_d_1"] = {
                        "w": 0, "b": 0}
                    state["params"]["lifted_1/head_seg/~/conv2_d"]["w"] = np.random.normal(
                        0, 0.1, size=(3, 3, 32, 160))
                    state["params"]["lifted_1/head_seg/~/conv2_d"]["b"] = np.random.normal(
                        0, 0.1, size=(160))
                    state["params"]["lifted_1/head_seg/~/conv2_d_1"]["w"] = np.random.normal(
                        0, 0.1, size=(1, 1, 160, 128))
                    state["params"]["lifted_1/head_seg/~/conv2_d_1"]["b"] = np.random.normal(
                        0, 0.1, size=(128))
                    if (self.config["model_attrs"]["cv"]["transpose"] == "True"):
                        state["params"]["lifted_1/head_seg/~/conv2_d_transpose"] = {
                            "w": 0, "b": 0}
                        state["params"]["lifted_1/head_seg/~/conv2_d_transpose"]["w"] = np.random.normal(
                            0, 0.1, size=(3, 3, 20, 128))
                        state["params"]["lifted_1/head_seg/~/conv2_d_transpose"]["b"] = np.random.normal(
                            0, 0.1, size=(20))

                    # Initialize opt_state
                    state["opt_state"][1][0][1]["lifted_1/head_seg/~/conv2_d"] = {
                        "w": 0, "b": 0}
                    state["opt_state"][1][0][2]["lifted_1/head_seg/~/conv2_d"] = {
                        "w": 0, "b": 0}
                    state["opt_state"][1][0][1]["lifted_1/head_seg/~/conv2_d_1"] = {
                        "w": 0, "b": 0}
                    state["opt_state"][1][0][2]["lifted_1/head_seg/~/conv2_d_1"] = {
                        "w": 0, "b": 0}
                    # state["opt_state"][1][0][1]["lifted_1/head_seg/~/conv2_d"]["w"] = np.ones(
                    #     (3, 3, 32, 160))
                    # state["opt_state"][1][0][2]["lifted_1/head_seg/~/conv2_d"]["b"] = np.ones(
                    #     (160))
                    # state["opt_state"][1][0][1]["lifted_1/head_seg/~/conv2_d_1"]["w"] = np.ones(
                    #     (1, 1, 160, 128))
                    # state["opt_state"][1][0][2]["lifted_1/head_seg/~/conv2_d_1"]["b"] = np.ones(
                    #     (128))

                    if (self.config["model_attrs"]["cv"]["transpose"] == "True"):
                        state["opt_state"][1][0][1]["lifted_1/head_seg/~/conv2_d_transpose"] = {
                            "w": 0, "b": 0}
                        state["opt_state"][1][0][2]["lifted_1/head_seg/~/conv2_d_transpose"] = {
                            "w": 0, "b": 0}
                        # state["opt_state"][1][0][1]["lifted_1/head_seg/~/conv2_d_transpose"]["w"] = np.random.normal(
                        #     0, 0.1, size=(3, 3, 20, 128))
                        # state["opt_state"][1][0][2]["lifted_1/head_seg/~/conv2_d_transpose"]["b"] = np.random.normal(
                        #     0, 0.1, size=(20))
                else:
                    print("Pre-trained weights not used...")

            else:
                raise Exception("Check checkpointer...")

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
            with open(path, 'wb') as f:
                pickle.dump(checkpoint_state, f)

        state, out = self._inner.update(state, data)
        return state, out


def main(config):
    if ("datalake" in config["data_attrs"]["dataset_path"]):
        config["checkpoint_dir"] = config["logging"]["logdir"] + "/test/"
    else:
        # config["checkpoint_dir"] = config["logging"]["logdir"] + \
        #     str(int(time.time_ns()/1000)) + "/"
        config["checkpoint_dir"] = config["logging"]["logdir"] + "/"

    # Save a snapshot of the code to the checkpoint directory
    os.system("mkdir -p "+config["checkpoint_dir"])
    os.system("cp -pr ./models ./solvers ./utils " + config["job"] + " " +
              config["checkpoint_dir"])

    # copy over pre-trained weights
    if (config["model_attrs"]["pretrained"] == "True"):
        os.system("cp -pr ../imagenet_pretrained.pkl " + " " +
                  config["checkpoint_dir"]+"/checkpoint_0785000.pkl")

    # Get the dataset in the required format
    data = Datasets(config)
    ds_dict = data.get_datasets()
    print(ds_dict['ds_trn'])
    print(ds_dict['ds_tst'])

    checkpoint_every_n = config["logging"]["checkpoint_every_n"]

    # Setup the forward pass
    build_forward_fn = Forward(config).build_forward_fn
    forward_fn = hk.transform(build_forward_fn())

    # Load loss function
    loss_fn = Losses(config, forward_fn).get_loss_fn()

    # cosine scheduler
    total_steps = config["opt_attrs"]["epochs"]  # Total Batches
    cosine_decay_scheduler = optax.cosine_decay_schedule(
        -config["opt_attrs"]["learning_rate"], total_steps)

    # Define optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config["opt_attrs"]["grad_clip_value"]),
        optax.adam(learning_rate=config["opt_attrs"]["learning_rate"],
                   b1=config["opt_attrs"]["b1"],
                   b2=config["opt_attrs"]["b2"])
    )

    # optimizer = optax.chain(
    #     optax.adam(config["opt_attrs"]["learning_rate"],
    #                b1=config["opt_attrs"]["b1"], b2=config["opt_attrs"]["b2"]),
    #     # optax.scale_by_schedule(cosine_decay_scheduler)
    # )

    updater = Updater(forward_fn.init, loss_fn, optimizer)
    updater = CheckpointingUpdater(
        updater, config, config["checkpoint_dir"], checkpoint_every_n, num_classes=config["data_attrs"]["num_classes"])

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
        for epoch in range(1, config["opt_attrs"]["epochs"]):
            loss = []
            for step, (x, y) in enumerate(ds_dict['dl_trn']):
                x = preproc(x, config)
                # print('x.shape: {}, y.shape: {}'.format(x.shape, y.shape))
                data = {'obs': x, 'target': y}
                if (step < config["opt_attrs"]["max_steps"]):
                    if (epoch == 1 and step == 0):
                        # Initialize state
                        state = updater.init(rng, data)

                    state, metrics = updater.update(state, data)
                    loss.append(metrics['loss'])
                    # Training logs
                    if step % config["logging"]["log_every"] == 0:
                        steps_per_sec = config["logging"]["log_every"] / \
                            (time.time() - prev_time)
                        prev_time = time.time()
                        # print(metrics)
                        metrics.update({'steps_per_sec': steps_per_sec})
                        metrics.update({'loss': np.mean(loss)})
                        metrics.update({'epoch': np.mean(epoch)})
                        logger(metrics, order=[
                            'epoch',
                            'step',
                            'loss',
                            'steps_per_sec'
                        ])
                else:
                    break

            # ============================ Evaluation logs ===========================
            if (epoch % config["logging"]["eval_every"] == 0 and epoch != 0):
                if (config["mode"] == "cls" or config["mode"] == "cls_trans"):
                    evaluate_cls(rng, state, epoch, config,
                                 ds_dict, preproc, functools.partial(cls_metrics, forward_fn=forward_fn))
                elif (config["mode"] == "seg"):
                    evaluate_seg(rng, state, epoch, config,
                                 ds_dict, preproc, functools.partial(seg_metrics, forward_fn=forward_fn, config=config))
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
    config["job"] = args.job_id

    out = json.dumps(config, indent=4, sort_keys=True)
    print(out)
    main(config)
