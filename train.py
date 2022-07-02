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
from utils.utils import evaluate_cls, evaluate_seg
from typing import Any, Mapping


from absl import app
from absl import flags
from absl import logging

# from examples.transformer import dataset
# from examples.transformer import model
from models import resnet, transformer_lm
from utils.datasets import Datasets

import jax
import optax
import haiku as hk
import numpy as np
import jax.numpy as jnp
import utils.dataset as dataset
from utils.utils import logger
from utils.utils import run
from utils.utils import get_outputs
from utils.utils import patchify
# from tabulate import tabulate


def build_forward_fn(vocab_size: int, d_model: int, num_heads: int,
                     num_layers: int, dropout_rate: float):
    """Create the model's forward pass."""

    def forward_fn(data: Mapping[str, jnp.ndarray],
                   is_training: bool = True) -> jnp.ndarray:
        """Forward pass."""

        if (config["mode"] == 'text'):
            from models import transformer_lm
            tokens = data['obs']
            input_mask = jnp.greater(tokens, 0)
            seq_length = tokens.shape[1]
            vocab_size = tokens.shape[-1] * 2  # TODO-clean

            # Embed the input tokens and positions.
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            token_embedding_map = hk.Embed(
                vocab_size, d_model, w_init=embed_init)
            token_embs = token_embedding_map(tokens)
            positional_embeddings = hk.get_parameter(
                'pos_embs', [seq_length, d_model], init=embed_init)
            x = token_embs + positional_embeddings
            # Run the transformer over the inputs.
            # Transform the transformer
            transformer = transformer_lm.Transformer(
                num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_rate
            )
            transformer_pure = hk.transform(transformer)
            z_star = run(config["deq_flag"] == "True", config["solver"], config["mode"], x,
                         transformer_pure, input_mask, max_iter=10)

            return hk.Linear(vocab_size)(z_star)

            # TODO: Fix state_with_list updater - non-functional because
            # updater needs to be passed downstream...
        elif (config["mode"] == 'cls'):
            x = data['obs'].astype('float32')
            num_classes = config["num_classes"]

            def resnet_fn(x, is_training):
                model = resnet.ResNet18(
                    num_classes=num_classes, resnet_v2=True)
                return model(x, is_training=is_training)

            transformer_cv = hk.transform_with_state(resnet_fn)
            z_star = run(config["deq_flag"] == "True", config["solver"], config["mode"], x,
                         transformer_cv, input_mask=None, max_iter=10)
            return z_star
        elif (config["mode"] == 'cls_trans'):
            from models.transformer_cv import TransformerCV
            x = data['obs'].astype('float32')
            num_classes = config["num_classes"]
            latent_dims = eval(config["model_attrs"]["latent_dims"])

            def cls_fn(x, is_training):
                # TODO: move to config
                patch_size = config["model_attrs"]["patch_size"]
                num_heads = config["model_attrs"]["num_heads"]
                depth = config["model_attrs"]["depth"]
                resample_dim = config["model_attrs"]["resample_dim"]
                vit_mode = config["model_attrs"]["vit_mode"]
                latent_dims = eval(config["model_attrs"]["latent_dims"])
                model = TransformerCV(x.shape, patch_size, num_heads, num_classes,
                                      depth, resample_dim, vit_mode, latent_dims=latent_dims)
                return model(x)

            transformer_seg = hk.transform(cls_fn)
            z_star = run(config["deq_flag"] == "True", config["solver"], config["mode"], x,
                         transformer_seg, input_mask=False, max_iter=10)

            # Additional steps for downstream classification
            z_star = get_outputs(
                z_star,
                mode="cls",
                resample_dim=config["model_attrs"]["resample_dim"],
                patch_size=config["model_attrs"]["patch_size"],
                num_classes=num_classes)
            return z_star

        # TODO: Add fusion modeule or the like...
        elif (config["mode"] == 'seg'):
            from models.transformer_cv import TransformerCV
            # print("data[\'obs\']: {}".format(data['obs']))
            x = data['obs'].astype('float32')
            num_classes = config["num_classes"]

            def seg_fn(x, is_training):
                # TODO: move to config
                patch_size = config["model_attrs"]["patch_size"]
                num_heads = config["model_attrs"]["num_heads"]
                depth = config["model_attrs"]["depth"]
                resample_dim = config["model_attrs"]["resample_dim"]
                vit_mode = config["model_attrs"]["vit_mode"]
                latent_dims = eval(config["model_attrs"]["latent_dims"])
                model = TransformerCV(x.shape, patch_size, num_heads, num_classes,
                                      depth, resample_dim, vit_mode, latent_dims=latent_dims)
                return model(x)

            transformer_seg = hk.transform(seg_fn)
            z_star = run(config, x,
                         transformer_seg, input_mask=False, max_iter=10)
            # Additional steps for downstream segmentation
            # config["mode"] = "seg_head"
            # print("z_star: {}".format(z_star))

            # transformer_seg_out = hk.transform(get_outputs)
            # z_star = run(config, z_star,
            #              transformer_seg_out, input_mask=False, max_iter=10)
            # print("z_star: {}".format(z_star))
            # z_star = get_outputs(
            #     z_star,
            #     mode="seg",
            #     resample_dim=config["model_attrs"]["resample_dim"],
            #     patch_size=config["model_attrs"]["patch_size"],
            #     num_classes=num_classes)
            return z_star

    return forward_fn


def lm_loss_fn(forward_fn,
               vocab_size: int,
               params,
               rng,
               data: Mapping[str, jnp.ndarray],
               is_training: bool = True) -> jnp.ndarray:
    """Compute the loss on data wrt params."""
    logits = forward_fn(params, rng, data, is_training)
    targets = jax.nn.one_hot(data['target'], vocab_size)
    print("logits.shape: {} - targets.shape: {}".format(logits.shape, targets.shape))
    assert logits.shape == targets.shape

    mask = jnp.greater(data['obs'], 0)
    loss = jnp.sum(-jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1))
    loss = jnp.sum(loss * mask) / jnp.sum(mask)
    return loss


def vm_loss_fn(forward_fn,
               classes: int,
               params,
               rng,
               data: Mapping[str, jnp.ndarray],
               is_training: bool = True) -> jnp.ndarray:
    """Compute the loss on data wrt params."""
    # print('data.shape: {}'.format(data))
    logits = forward_fn(params, rng, data, is_training)
    targets = jax.nn.one_hot(data['target'], classes)
    print("logits.shape: {} - targets.shape: {}".format(logits.shape, targets.shape))
    assert logits.shape == targets.shape
    loss = jnp.sum(-jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1))

    return loss


# TODO: custom segmentation loss
def seg_loss_fn(forward_fn,
                classes: int,
                params,
                rng,
                data: Mapping[str, jnp.ndarray],
                is_training: bool = True) -> jnp.ndarray:
    """Compute the loss on data wrt params."""
    # print('data.shape: {}'.format(data))
    logits = forward_fn(params, rng, data, is_training)
    targets = jax.nn.one_hot(data['target'], classes)
    #targets = data['target']
    # print("targets.shape: {}".format(targets.shape))
    # print("logits.shape: {}".format(logits.shape))
    print("logits.shape: {} - targets.shape: {}".format(logits.shape, targets.shape))
    assert logits.shape == targets.shape
    loss = jnp.sum(-jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1))

    return loss


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


def preproc(x, config):
    x = np.expand_dims(x, axis=3) if len(x.shape) == 3 else x
    # TODO: fix
    x = np.repeat(x, 3, axis=3) if x.shape[3] == 1 else x
    if (x.shape[-1] == 3):
        # shift c axis to the end
        # [B, C, H, W] -> [B, H, W, C]
        x = np.transpose(x, (0, 3, 1, 2))

    # Change the format of the data
    # from img -> img_patch
    patch_size = config["model_attrs"]["patch_size"]
    return patchify(patch_size, x)


def main(config):

    config["checkpoint_dir"] = config["logdir"] + str(int(time.time())) + "/"

    # Create the dataset.
    if (config["mode"] == 'text'):
        # TODO: move to datasets
        train_dataset = dataset.AsciiDataset(
            config['dataset_path']+'shakespeare.txt', config["batch_size"], config["sequence_length"])
        vocab_size = train_dataset.vocab_size
        # Set up the model, loss, and updater.
        forward_fn = build_forward_fn(vocab_size, config["d_model"], config["num_heads"],
                                      config["num_layers"], config["dropout_rate"])
        forward_fn = hk.transform(forward_fn)
        loss_fn = functools.partial(
            lm_loss_fn, forward_fn.apply, vocab_size)
    elif (config["mode"] == 'cls'):
        forward_fn = build_forward_fn(config['num_classes'], config["d_model"], config["num_heads"],
                                      config["num_layers"], config["dropout_rate"])
        forward_fn = hk.transform(forward_fn)
        loss_fn = functools.partial(
            vm_loss_fn, forward_fn.apply, config['classes'])
    elif (config["mode"] == 'cls_trans'):
        forward_fn = build_forward_fn(config['num_classes'], config["d_model"], config["num_heads"],
                                      config["num_layers"], config["dropout_rate"])
        forward_fn = hk.transform(forward_fn)
        loss_fn = functools.partial(
            vm_loss_fn, forward_fn.apply, config['classes'])
    elif (config["mode"] == 'seg'):
        forward_fn = build_forward_fn(config['num_classes'], config["d_model"], config["num_heads"],
                                      config["num_layers"], config["dropout_rate"])
        forward_fn = hk.transform(forward_fn)
        loss_fn = functools.partial(
            seg_loss_fn, forward_fn.apply, config['num_classes'])

    optimizer = optax.chain(
        optax.clip_by_global_norm(config["grad_clip_value"]),
        optax.adam(config["learning_rate"], b1=0.9, b2=0.99))

    updater = Updater(forward_fn.init, loss_fn, optimizer)
    updater = CheckpointingUpdater(updater, config["checkpoint_dir"])

    # Initialize parameters.
    rng = jax.random.PRNGKey(428)

    prev_time = time.time()

    if (config["mode"] == 'text'):
        # TODO: Improve structure
        data = next(train_dataset)
        state = updater.init(rng, data)
        for step in range(config["max_steps"]):
            data = next(train_dataset)
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

    elif (config["mode"] == 'cls'):

        # Get the dataset in the required format
        d = Datasets(config)
        ds_dict = d.get_datasets()
        print(ds_dict['ds_trn'])
        print(ds_dict['ds_tst'])

        # Train the model
        for epoch in range(config["epochs"]):
            for step, (x, y) in enumerate(ds_dict['dl_trn']):
                x = preproc(x, config)
                # print('x.shape: {}, y.shape: {}'.format(x.shape, y.shape))
                data = {'obs': x, 'target': y}
                if (step < config["max_steps"]):
                    if (epoch == 0 and step == 0):
                        # Initialize state
                        state = updater.init(rng, data)

                    def accuracy(params, rng, x, y):
                        target_class = jnp.argmax(y, axis=1)
                        predicted_class = jnp.argmax(
                            forward_fn.apply(params, rng, data={'obs': x, 'target': y}, is_training=False), axis=1)
                        return jnp.mean(predicted_class == target_class)

                    state, metrics = updater.update(state, data)

                    # Training logs
                    if step % config["log_every"] == 0:
                        steps_per_sec = config["log_every"] / \
                            (time.time() - prev_time)
                        prev_time = time.time()
                        metrics.update({'steps_per_sec': steps_per_sec})
                        logger(metrics, order=[
                               'step',
                               'loss',
                               'steps_per_sec']
                               )

            # ============================ Evaluation logs ===========================
            evaluate_cls(rng, state, epoch, config, ds_dict, preproc, accuracy)
            # ============================ Evaluation logs ===========================

    elif (config["mode"] == 'cls_trans'):

        # Get the dataset in the required format
        d = Datasets(config)
        ds_dict = d.get_datasets()
        print(ds_dict['ds_trn'])
        print(ds_dict['ds_tst'])

        # Train the model
        for epoch in range(config["epochs"]):
            for step, (x, y) in enumerate(ds_dict['dl_trn']):
                x = preproc(x, config)
                # print('x.shape: {}, y.shape: {}'.format(x.shape, y.shape))
                data = {'obs': x, 'target': y}
                if (step < config["max_steps"]):
                    if (epoch == 0 and step == 0):
                        # Initialize state
                        state = updater.init(rng, data)

                    def accuracy(params, rng, x, y):
                        target_class = jnp.argmax(y, axis=1)
                        predicted_class = jnp.argmax(
                            forward_fn.apply(params, rng, data={'obs': x, 'target': y}, is_training=False), axis=1)
                        return jnp.mean(predicted_class == target_class)

                    state, metrics = updater.update(state, data)

                    # Training logs
                    if step % config["log_every"] == 0:
                        steps_per_sec = config["log_every"] / \
                            (time.time() - prev_time)
                        prev_time = time.time()
                        metrics.update({'steps_per_sec': steps_per_sec})
                        logger(metrics, order=[
                            'step',
                            'loss',
                            'steps_per_sec'
                        ])

            # ============================ Evaluation logs ===========================
            evaluate_cls(rng, state, epoch, config, ds_dict, preproc, accuracy)
            # ============================ Evaluation logs ===========================

    elif (config["mode"] == 'seg'):

        # Get the dataset in the required format
        d = Datasets(config)
        ds_dict = d.get_datasets()
        print(ds_dict['ds_trn'])
        print(ds_dict['ds_tst'])

        # Train the model
        for epoch in range(config["epochs"]):
            for step, (x, y) in enumerate(ds_dict['dl_trn']):
                x = preproc(x, config)
                # print('x.shape: {}, y.shape: {}'.format(x.shape, y.shape))
                data = {'obs': x, 'target': y}
                if (step < config["max_steps"]):
                    if (epoch == 0 and step == 0):
                        # Initialize state
                        state = updater.init(rng, data)

                    # Jaccard similarity
                    def jaccard(params, rng, x_patch, x, y, save_img_to_folder):
                        y_hat = jax.jit(forward_fn.apply)(params, rng, data={
                            'obs': x_patch, 'target': y}, is_training=False)
                        # print("x.shape: {}".format(x.shape))
                        # print("y.shape: {}".format(y.shape))
                        # print("y_hat.shape: {}".format(y_hat.shape))
                        print("np.unique(y_hat): {}".format(np.unique(y_hat)))
                        print("np.unique(y): {}".format(np.unique(y)))
                        save_img_to_folder(config, x, y, y_hat)
                        xt = y_hat != 0
                        yt = y != 0
                        num = jnp.sum(jnp.logical_xor(
                            xt, yt).astype(jnp.int32))
                        denom = jnp.sum(jnp.logical_or(jnp.logical_and(
                            xt, yt), jnp.logical_xor(xt, yt)).astype(jnp.int32))
                        return jnp.where(denom == 0, 0.0, num.astype(jnp.float32) / denom)

                    # mean average precision
                    def accuracy(params, rng, x, y):
                        target_class = jnp.argmax(y, axis=1)
                        predicted_class = jnp.argmax(
                            forward_fn.apply(params, rng, data={'obs': x, 'target': y}, is_training=False), axis=1)
                        return jnp.mean(predicted_class == target_class)

                    state, metrics = updater.update(state, data)

                    # Training logs
                    if step % config["log_every"] == 0:
                        steps_per_sec = config["log_every"] / \
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
            print("Calling evaluate function...")
            evaluate_seg(rng, state, epoch, config,
                         ds_dict, preproc, jaccard)
            # ============================ Evaluation logs ===========================

    return "Complete!"


if __name__ == '__main__':
    # TODO add to config file
    # config = {
    #     "dataset_path": "/home/skhalid/Documents/datalake/",
    #     "dataset": "MNIST",  # ["ImageNet", "CIFAR10", "MNIST", "Cityscapes"]
    #     "batch_size": 32,
    #     "transform": None,
    #     "n_threads": 1,
    #     "log_every": 100,
    #     "max_steps": 1000,
    #     "deq_flag": False,
    #     "solver": 0,  # {"0": "Broyden", "1": "Anderson", "2": "ETC"}
    #     "log": False,
    #     "mode": "seg",  # ["text", "cls", "seg"]
    #     "epochs": 100,
    #     "classes": 10,
    #     "num_heads": 4,
    #     "num_layers": 4,
    #     "learning_rate": 3e-4,
    #     "grad_clip_value": 1,
    #     "dropout_rate": 0.1,
    #     "d_model": 128,
    #     "sequence_length": 64,
    #     "checkpoint_dir": "/tmp/haiku-transformer",
    #     "logdir": "./logs/"

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--job_id', help='job id for the set of jobs in ./jobs')
    args = parser.parse_args()
    print(args)
    config = json.load(open(args.job_id, "r"))

    out = json.dumps(config, indent=4, sort_keys=True)
    print(out)
    main(config)
