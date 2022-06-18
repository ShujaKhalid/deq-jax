'''
Create datasets and dataloader
'''

import torch
import torchvision
import numpy as np

torch.manual_seed(1993)

CUSTOM_DATASETS = ['shakespeare_mini']


class Datasets():
    def __init__(self, config: dict, transform=None):
        self.config = config
        self.dataset_path = self.config.path
        self.dataset_name = self.config.dataset
        self.transform = config.transform
        self.dataset_train = None
        self.dataset_test = None
        self.data_loader_train = None
        self.data_loader_test = None

    def get_dataset_stats(self):
        if (self.dataset_train == None or self.data_loader == None):
            raise Exception('Error during dataset initialization...')

        print("self.dataset_train: {}".format(self.dataset_train))
        print("self.dataset_test: {}".format(self.dataset_test))

        return

    def get_dataloader(self, mode):
        return torch.utils.data.DataLoader(eval(self.dataset+eval("_"+mode)),
                                           batch_size=self.config.batch_size,
                                           shuffle=True,
                                           num_workers=self.config.n_threads)

    def get_dataset(self, train):
        if not self.dataset_name:
            raise Exception("No dataset has been set!?!")
        return getattr(torchvision.datasets, self.dataset_name)(
            root=self.dataset_path,
            train=train,
            download=True)

    def get_dataset(self):
        # Add new datasets to the schematic below.
        if (self.dataset_name is not None):
            self.dataset_train = self.get_dataset(train=True)
            self.dataset_test = self.get_dataset(train=False)
            self.data_loader_train = self.get_dataloader(mode="train")
            self.data_loader_test = self.get_dataloader(mode="test")

            # Print out dataset statistics
            self.get_dataset_stats()

        if (self.dataset in CUSTOM_DATASETS):
            print("\nloading custom dataset\n")
            if (self.dataset == 'shakespear_mini'):
                pass  # TODO
        else:
            raise Exception(
                "Dataset not present in arsenal... Create schematic.")

        return {
            "ds_trn": self.dataset_train,
            "ds_tst": self.dataset_test,
            "dl_trn": self.data_loader_train,
            "dl_tst": self.data_loader_test
        }
