'''
Create datasets and dataloader
'''

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import jax.numpy as jnp

torch.manual_seed(1993)

CUSTOM_DATASETS = ['shakespeare_mini']


class Datasets():
    def __init__(self, config: dict, transform=None):
        self.config = config
        self.dataset_path = self.config['dataset_path']
        self.dataset_name = self.config['dataset']
        self.transform = config['transform']
        self.dataset_train = None
        self.dataset_test = None
        self.data_loader_train = None
        self.data_loader_test = None

    def get_dataset_stats(self):
        if (self.dataset_train == None or
                self.data_loader_train == None or
                self.dataset_test == None or
                self.data_loader_test == None):
            raise Exception('Error during dataset initialization...')

        #print("\nself.dataset_train: {}".format(self.dataset_train))
        #print("\nself.dataset_test: {}".format(self.dataset_test))

        return

    def get_dataloader(self, mode):
        def collate_numpy(batch):
            # print([v[0].shape for v in batch])

            if isinstance(batch[0], np.ndarray):
                return np.stack(batch)
            elif isinstance(batch[0], (tuple, list)):
                transposed = zip(*batch)
                return [collate_numpy(samples) for samples in transposed]
            else:
                return np.array(batch)

        return torch.utils.data.DataLoader(eval("self.dataset_"+mode),
                                           batch_size=self.config["batch_size"],
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=self.config["n_threads"],
                                           collate_fn=collate_numpy)

    def get_dataset(self, train):
        def make_jax_friendly_tgt(pic):
            transform_cityscapes_friendly = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                    #     0.229, 0.224, 0.225])
                ])

            if (self.dataset_name == "Cityscapes"):
                pic = transform_cityscapes_friendly(pic)
            return np.squeeze(np.array(pic, jnp.float32), axis=0)

        def make_jax_friendly(pic):
            transform_imagenet_friendly = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225])
                ])
            transform_cityscapes_friendly = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225])
                ])

            if (self.dataset_name == "ImageNet"):
                pic = transform_imagenet_friendly(pic)

            if (self.dataset_name == "Cityscapes"):
                pic = transform_cityscapes_friendly(pic)
            return np.array(pic, jnp.float32)

        if not self.dataset_name:
            raise Exception("No dataset has been set!?!")

        if (self.dataset_name == "ImageNet"):
            return getattr(torchvision.datasets, self.dataset_name)(
                root=self.dataset_path,
                split="train" if train else "val",
                transform=make_jax_friendly)
        elif (self.dataset_name == "Cityscapes"):
            return getattr(torchvision.datasets, self.dataset_name)(
                root=self.dataset_path+"cityscapes",
                split="train" if train else "val",
                # ["coarse", "fine"]
                mode="coarse",
                # ["instance", "semantic", "polygon", "color"]
                target_type="semantic",
                transform=make_jax_friendly,
                target_transform=make_jax_friendly_tgt)
        else:
            return getattr(torchvision.datasets, self.dataset_name)(
                root=self.dataset_path,
                train=train,
                transform=make_jax_friendly,
                download=True)

    def get_datasets(self):
        # Add new datasets to the schematic below.
        if (self.dataset_name is not None):
            self.dataset_train = self.get_dataset(train=True)
            self.dataset_test = self.get_dataset(train=False)
            self.data_loader_train = self.get_dataloader(mode="train")
            self.data_loader_test = self.get_dataloader(mode="test")

            # Print out dataset statistics
            self.get_dataset_stats()

        elif (self.dataset_name in CUSTOM_DATASETS):
            print("\nloading custom dataset\n")
            if (self.dataset_name == 'shakespear_mini'):
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
