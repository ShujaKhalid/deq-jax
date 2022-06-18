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
import pickle
import numpy as np


class Cifar10Dataset:
    def __init__(self, path: str, batch_size: int):
        """Load a single-file ASCII dataset in memory."""
        self.batch_train = 0
        self.batch_test = 0
        self.classes = 10
        self._batch_size = batch_size
        self.data_train = {'x': [], 'y': []}
        self.data_test = {'x': [], 'y': []}

        print('file: {}'.format(path+'/*data_batch*'))
        for file in glob.glob(path+'/*data_batch*'):
            with open(file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                for i in range(len(batch[b'data'])):
                    r, g, b = np.array_split(batch[b'data'][i, :], 3)
                    self.data_train['x'].append(np.array([v.reshape(32, 32)
                                                          for v in [r, g, b]]))
                    self.data_train['y'].append(batch[b'labels'][i])

        for file in glob.glob(path+'/*test_batch*'):
            with open(file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                for i in range(len(batch)):
                    r, g, b = np.array_split(batch[b'data'][i, :], 3)
                    self.data_test['x'].append(np.array([v.reshape(32, 32)
                                                         for v in [r, g, b]]))
                    self.data_test['y'].append(batch[b'labels'][i])

        # shuffle
        self.inds_train = [v for v in range(len(self.data_train['x']))]
        np.random.shuffle(self.inds_train)
        self.inds_test = [v for v in range(len(self.data_test['x']))]
        np.random.shuffle(self.inds_test)
        # print('self.inds_train: {}'.format(self.inds_train))
        # print('self.inds_test: {}'.format(self.inds_test))

        self._ds_train = {
            'x': np.array(self.data_train['x'])[self.inds_train].transpose(2, 3, 1, 0),
            'y': np.array(self.data_train['y'])[self.inds_train],
        }
        self._ds_test = {
            'x': np.array(self.data_test['x'])[self.inds_test].transpose(2, 3, 1, 0),
            'y': np.array(self.data_test['y'])[self.inds_test],
        }

    def __next__(self, mode):
        """Yield next mini-batch."""

        if (mode == 'train'):
            obs = np.array(self._ds_train['x'][:, :, :, (self.batch*self._batch_size):(
                (self.batch+1)*(self._batch_size))])
            tgt = np.array(self._ds_train['y'][(self.batch*self._batch_size):(
                (self.batch+1)*(self._batch_size))])
            batch = dict(obs=obs, target=tgt)
            self.batch_train += 1
        elif (mode == 'test'):
            obs = np.array(self._ds_test['x'][:, :, :, (self.batch*self._batch_size):(
                (self.batch+1)*(self._batch_size))])
            tgt = np.array(self._ds_test['y'][(self.batch*self._batch_size):(
                (self.batch+1)*(self._batch_size))])
            batch = dict(obs=obs, target=tgt)
            self.batch_test += 1

        return batch

    def __iter__(self):
        return self
