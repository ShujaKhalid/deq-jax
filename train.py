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

from jax.scipy.special import logsumexp
import functools
import os
import pickle
import time
from typing import Any, Mapping
from tqdm import tqdm

from absl import app
from absl import flags
from absl import logging
import haiku as hk
# from examples.transformer import dataset
# from examples.transformer import model
import model
import dataset
from datasets import Datasets
import dataset_resnet
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import grad, jit, vmap, random

import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST

flags.DEFINE_string('dataset_path', None,
                    'Single-file dataset location.')

flags.DEFINE_integer('batch_size', 16, 'Train batch size per core')
flags.DEFINE_integer('sequence_length', 64, 'Sequence length to learn on')

flags.DEFINE_integer('d_model', 128, 'model width')
flags.DEFINE_integer('num_heads', 4, 'Number of attention heads')
flags.DEFINE_integer('num_layers', 4, 'Number of transformer layers')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate')

flags.DEFINE_float('learning_rate', 3e-4, 'Max learning-rate')
flags.DEFINE_float('grad_clip_value', 1, 'Gradient norm clip value')

flags.DEFINE_string('checkpoint_dir', '/tmp/haiku-transformer',
                    'Directory to store checkpoints.')

FLAGS = flags.FLAGS
LOG_EVERY = 100
MAX_STEPS = 1000  # 10**6
DEQ_FLAG = False
LOG = False
MODE = 'cls'  # ['text', 'cls', 'seg']


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


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


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


def build_forward_fn(vocab_size: int, d_model: int, num_heads: int,
                     num_layers: int, dropout_rate: float):
    """Create the model's forward pass."""

    def forward_fn(data: Mapping[str, jnp.ndarray],
                   is_training: bool = True) -> jnp.ndarray:
        """Forward pass."""

        if (MODE == 'text'):
            tokens = data['obs']
            input_mask = jnp.greater(tokens, 0)
            seq_length = tokens.shape[1]

            # Embed the input tokens and positions.
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            token_embedding_map = hk.Embed(
                vocab_size, d_model, w_init=embed_init)
            token_embs = token_embedding_map(tokens)
            positional_embeddings = hk.get_parameter(
                'pos_embs', [seq_length, d_model], init=embed_init)
            input_embeddings = x = token_embs + positional_embeddings
            # Run the transformer over the inputs.
            # Transform the transformer
            transformer = model.Transformer(
                num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_rate
            )
            transformer_pure = hk.transform(transformer)

            # lift params
            h = jnp.zeros_like(x)
            inner_params = hk.experimental.lift(
                transformer_pure.init)(hk.next_rng_key(), x, input_mask, is_training)

            if (DEQ_FLAG):
                from deq import deq
                max_iter = 10
                solver = 1  # 0: Broyden ; 1: Anderson ; 2: secant

                # Define a callable function for ease of access downstream
                def f(params, rng, x, input_mask):
                    # print('params: {}'.format(params))
                    # print('rng: {}'.format(rng))
                    # print('x: {}'.format(x))
                    # print('input_mask: {}'.format(input_mask))
                    # print('is_training: {}'.format(is_training))
                    return transformer_pure.apply(params, rng, x, input_mask, is_training=is_training)

                z_star = deq(
                    inner_params, hk.next_rng_key(), x, f, max_iter, solver, input_mask)
            else:
                z_star = transformer(
                    x, input_mask, is_training)

        elif (MODE == 'cls'):
            # TODO import resnet model here
            # Use `single` and `multi-scale` approach here
            from resnet50 import ResNet50
            x = data['obs'].astype('float32')
            rng_key = jax.random.PRNGKey(0)
            batch_size = FLAGS.batch_size
            num_classes = 10
            # TODO: resize original imgs from 32 -> 224
            input_shape = (32, 32, 3, batch_size)
            step_size = 0.1
            num_steps = 10
            init_fun, predict_fun = ResNet50(num_classes)
            _, init_params = init_fun(rng_key, input_shape)
            resnet_pure = hk.transform(predict_fun)
            # inner_params = hk.experimental.lift(
            #     resnet_pure.init)(hk.next_rng_key(), x, is_training)
            vocab_size = num_classes

            if (DEQ_FLAG):
                from deq import deq
                max_iter = 10
                solver = 1  # 0: Broyden ; 1: Anderson ; 2: secant

                # Define a callable function for ease of access downstream
                def f(params, rng, x):
                    # print('params: {}'.format(params))
                    # print('rng: {}'.format(rng))
                    # print('x: {}'.format(x))
                    # print('input_mask: {}'.format(input_mask))
                    # print('is_training: {}'.format(is_training))
                    return resnet_pure.apply(params, rng, x)

                z_star = deq(
                    inner_params, hk.next_rng_key(), x, f, max_iter, solver
                )
            else:
                # print('x: {}'.format(x.shape))
                # print('init_params: {}'.format(init_params.shape))
                z_star = predict_fun(init_params, inputs=x, rng=rng_key)

        elif (MODE == 'seg'):
            # TODO import resnet model here
            # Use `single` and `multi-scale` approach here
            pass
        # elif (MODE=='clsTrans'):
        #     # TODO import resnet model here
        #     # Use `single` and `multi-scale` approach here
        #     continue
        # elif (MODE=='segTrans'):
        #     # TODO import resnet model here
        #     # Use `single` and `multi-scale` approach here
        #     continue

        # Reverse the embeddings (untied).
        return hk.Linear(vocab_size)(z_star)

    return forward_fn


def lm_loss_fn(forward_fn,
               vocab_size: int,
               params,
               rng,
               data: Mapping[str, jnp.ndarray],
               is_training: bool = True) -> jnp.ndarray:
    """Compute the loss on data wrt params."""
    #print('data.shape: {}'.format(data))
    logits = forward_fn(params, rng, data, is_training)
    targets = jax.nn.one_hot(data['target'], vocab_size)
    assert logits.shape == targets.shape

    mask = jnp.greater(data['obs'], 0)
    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * mask) / jnp.sum(mask)
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
            print('self._checkpoint_dir: {}'.format(self._checkpoint_dir))
            # print('self.data: {}'.format(data))
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
            with open(path, 'wb') as f:
                pickle.dump(checkpoint_state, f)

        state, out = self._inner.update(state, data)
        return state, out


# def one_hot(x, k, dtype=jnp.float32):
#     """Create a one-hot encoding of x of size k."""
#     return jnp.array(x[:, None] == jnp.arange(k), dtype)


# def random_layer_params(m, n, key, scale=1e-2):
#     w_key, b_key = random.split(key)
#     return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# def init_network_params(key):
#     layer_sizes = [784, 512, 512, 10]
#     keys = random.split(key, len(layer_sizes))
#     return [random_layer_params(m, n, k) for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]


# def relu(x):
#     return jnp.maximum(0, x)


# def predict(params, image):
#     # per-example predictions
#     activations = image
#     for w, b in params[:-1]:
#         outputs = jnp.dot(w, activations) + b
#         activations = relu(outputs)

#     final_w, final_b = params[-1]
#     logits = jnp.dot(final_w, activations) + final_b
#     return logits - logsumexp(logits)


def main(_):
    FLAGS.alsologtostderr = True  # Always log visibly.
    # Create the dataset.
    if (MODE == 'text'):
        train_dataset = dataset.AsciiDataset(
            FLAGS.dataset_path, FLAGS.batch_size, FLAGS.sequence_length)
        vocab_size = train_dataset.vocab_size
        # Set up the model, loss, and updater.
        forward_fn = build_forward_fn(vocab_size, FLAGS.d_model, FLAGS.num_heads,
                                      FLAGS.num_layers, FLAGS.dropout_rate)
        forward_fn = hk.transform(forward_fn)
        loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, vocab_size)
    elif (MODE == 'cls'):
        # train_dataset = dataset_resnet.Cifar10Dataset(
        #     FLAGS.dataset_path, FLAGS.batch_size)
        # mnist_dataset = MNIST('/tmp/mnist/', download=True,
        #                       transform=FlattenAndCast())
        # training_generator = NumpyLoader(
        #     mnist_dataset, batch_size=FLAGS.batch_size, num_workers=0)

        # train_images = np.array(mnist_dataset.train_data).reshape(
        #     len(mnist_dataset.train_data), -1)
        # train_labels = one_hot(
        #     np.array(mnist_dataset.train_labels), 10)

        # Get full test dataset
        # mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
        # test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(
        #     len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
        # test_labels = one_hot(
        #     np.array(mnist_dataset_test.test_labels), 10)
        # Set up the model, loss, and updater.
        forward_fn = build_forward_fn(10, FLAGS.d_model, FLAGS.num_heads,
                                      FLAGS.num_layers, FLAGS.dropout_rate)
        forward_fn = hk.transform(forward_fn)
        loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, 10)

    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.grad_clip_value),
        optax.adam(FLAGS.learning_rate, b1=0.9, b2=0.99))

    updater = Updater(forward_fn.init, loss_fn, optimizer)
    updater = CheckpointingUpdater(updater, FLAGS.checkpoint_dir)

    # Initialize parameters.
    logging.info('Initializing parameters...')
    rng = jax.random.PRNGKey(428)

    logging.info('Starting train loop...')
    prev_time = time.time()

    if (MODE == 'text'):
        data = next(train_dataset, mode='train')
        state = updater.init(rng, data)
        for step in range(MAX_STEPS):
            data = next(train_dataset, mode='test')
            state, metrics = updater.update(state, data)
            # We use JAX runahead to mask data preprocessing and JAX dispatch overheads.
            # Using values from state/metrics too often will block the runahead and can
            # cause these overheads to become more prominent.
            if step % LOG_EVERY == 0:
                steps_per_sec = LOG_EVERY / (time.time() - prev_time)
                prev_time = time.time()
                metrics.update({'steps_per_sec': steps_per_sec})
                logging.info({k: float(v) for k, v in metrics.items()})

    elif (MODE == 'cls'):

        # TODO add to config file
        config = {
            "path": "/home/skhalid/Documents/datalake/",
            "dataset": "CIFAR10",
            "batch_size": FLAGS.batch_size,
            "transform": None,
            "n_threads": 1,
            "epochs": 1,
            "classes": 10
        }

        # Get the dataset in the required format
        d = Datasets(config)
        ds_dict = d.get_datasets()
        print('\n\n\nds_dict: {}\n\n\n'.format(ds_dict))

        # Train the model
        for epoch in range(config["epochs"]):
            for step, (x, y) in enumerate(ds_dict['dl_trn']):
                #print('x.shape: {}, y.shape: {}'.format(x.shape, y.shape))
                x = np.transpose(x, (1, 2, 3, 0))
                data = {'obs': x, 'target': y}
                if (step < MAX_STEPS):
                    if (step == 0):
                        # Initialize state
                        state = updater.init(rng, data)

                    def accuracy(params, rng, x, y):
                        target_class = jnp.argmax(y, axis=1)
                        predicted_class = jnp.argmax(
                            forward_fn.apply(params, rng, data={'obs': x, 'target': y}, is_training=False), axis=1)
                        return jnp.mean(predicted_class == target_class)

                    state, metrics = updater.update(state, data)

                    # Training logs
                    if step % LOG_EVERY == 0:
                        steps_per_sec = LOG_EVERY / \
                            (time.time() - prev_time)
                        prev_time = time.time()
                        metrics.update({'steps_per_sec': steps_per_sec})
                        logging.info({k: float(v)
                                      for k, v in metrics.items()})

            # ============================ Evaluation logs ===========================
            eval_trn = []
            eval_tst = []
            # for i, (x, y) in enumerate(ds_dict['dl_trn']):
            #     train_acc = accuracy(state['params'],
            #                          rng,
            #                          x,
            #                          jax.nn.one_hot(y, config["classes"]))
            #     eval_trn.append(train_acc)
            for i, (x, y) in enumerate(tqdm(ds_dict['dl_tst'])):
                x = np.transpose(x, (1, 2, 3, 0))
                test_acc = accuracy(state['params'],
                                    rng,
                                    x,
                                    jax.nn.one_hot(y, config["classes"]))
                eval_tst.append(test_acc)
            print("epoch: {} - iter: {} - acc_trn {:.2f} - acc_tst: {:.2f}".format(epoch, i,
                  np.mean(eval_trn), np.mean(eval_tst)))
            # ============================ Evaluation logs ===========================


if __name__ == '__main__':
    flags.mark_flag_as_required('dataset_path')
    app.run(main)
