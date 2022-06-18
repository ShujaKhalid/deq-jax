'''
Create datasets and dataloader
'''

import torch
import torchvision
import numpy as np

CUSTOM_DATASETS = ['shakespeare_mini']


class Datasets():
    def __init__(self, path: str, dataset: str, config: dict, transform=None):
        self.dataset_path = path
        self.dataset_name = dataset
        self.config = config
        self.dataset_train = None
        self.dataset_test = None
        self.data_loader = None
        self.data_loader = None
        self.transform = transform

    def get_dataset_stats(self):
        if (self.dataset == None or self.data_loader == None):
            raise Exception('No dataset has been loaded...')
        return

    def get_dataset(self):

        if (self.dataset_name == 'imagenet'):
            self.dataset = torchvision.datasets.ImageNet(
                self.dataset_path)
            self.data_loader = torch.utils.data.DataLoader(self.dataset,
                                                           batch_size=self.config.batch_size,
                                                           shuffle=True,
                                                           num_workers=self.config.n_threads)
        elif ():

        else:
            if (self.dataset in CUSTOM_DATASETS):
                print("\nloading custom dataset\n")
                if (self.dataset == 'shakespear_mini'):
                    pass  # TODO
            else:
                raise Exception(
                    "Dataset not present in arsenal... Create schematic.")

        return self.dataset, self.data_loader
