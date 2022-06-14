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
"""ResNet cifar10 dataset."""

import glob
import itertools
import random
import numpy as np


def _infinite_shuffle(iterable, buffer_size):
  """Infinitely repeat and shuffle data from iterable."""
  ds = itertools.cycle(iterable)
  buf = [next(ds) for _ in range(buffer_size)]
  random.shuffle(buf)
  while True:
    item = next(ds)
    idx = random.randint(0, buffer_size - 1)  # Inclusive.
    result, buf[idx] = buf[idx], item
    yield result


class Cifar10Dataset:
  def __init__(self, path: str, batch_size: int, sequence_length: int):
    """Load a single-file ASCII dataset in memory."""
    self.classes = 10
    self._batch_size = batch_size

    data = {'x': [], 'y': []. 'yb': []}
    for file in glob.glob(path+'/*data_batch*'): 
        with open(file, 'r') as f:
            batch = f.read()
            for i in range(len(batch)):
                r,g,b = np.array_split(dict[b'data'][i,:], 3)
                data['x'].append(np.array([v.reshape(32,32) for v in [r,g,b]]))
                data['y'].append(dict[b'labels'][i])
                data['yb'].append(dict[b'batch_label'][i])

    self._ds = _infinite_shuffle(data, batch_size * 10)

  def __next__(self):
    """Yield next mini-batch."""
    batch = [next(self._ds) for _ in range(self._batch_size)]
    batch = np.stack(batch)
    # Create the language modeling observation/target pairs.
    return dict(obs=batch['x'][:, :], target=batch['y'][:])

  def __iter__(self):
    return self
