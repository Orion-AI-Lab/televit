from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
import xarray as xr
import xbatcher
import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .components.seasfire_dataset import sample_dataset, BatcherDS


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class SeasFireSpatialDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            ds_path: str = None,
            input_vars: list = None,
            positional_vars: list = None,
            target: str = 'BurntArea',
            target_shift: int = 1,
            task: str = 'regression',
            batch_size: int = 64,
            num_workers: int = 8,
            pin_memory: bool = False,
            debug: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.ds_path = ds_path
        self.input_vars = list(input_vars)
        self.target = target
        self.target_shift = target_shift
        self.ds = xr.open_zarr(ds_path, consolidated=True)
        # TODO remove when we have the new datacube
        self.ds['sst'] = self.ds['sst'].where(self.ds['sst'] >= 0)
        self.mean_std_dict = None
        self.positional_vars = positional_vars
        if debug:
            self.num_timesteps = 5
        else:
            self.num_timesteps = -1
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.task = task

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            print(self.ds[self.input_vars])
            # IMPORTANT! Call sample_dataset with ds.copy(). xarray Datasets are mutable
            train_batches, self.mean_std_dict = sample_dataset(self.ds.copy(), input_vars=self.input_vars,
                                                               target=self.target,
                                                               target_shift=-self.target_shift, split='train',
                                                               num_timesteps=self.num_timesteps)
            # Save mean std dict for normalization during inference time
            print(self.mean_std_dict)
            with open(f'mean_std_dict_{self.target_shift}.json', 'w') as f:
                f.write(json.dumps(self.mean_std_dict))

            val_batches, _ = sample_dataset(self.ds.copy(), input_vars=self.input_vars, target=self.target,
                                            target_shift=-self.target_shift, split='val',
                                            num_timesteps=self.num_timesteps)

            test_batches, _ = sample_dataset(self.ds.copy(), input_vars=self.input_vars, target=self.target,
                                             target_shift=-self.target_shift, split='test',
                                             num_timesteps=self.num_timesteps)

            self.data_train = BatcherDS(train_batches, input_vars=self.input_vars, positional_vars=self.positional_vars,
                                        target=self.target,
                                        mean_std_dict=self.mean_std_dict, task=self.task)
            self.data_val = BatcherDS(val_batches, input_vars=self.input_vars, positional_vars=self.positional_vars,
                                      target=self.target,
                                      mean_std_dict=self.mean_std_dict, task=self.task)
            self.data_test = BatcherDS(test_batches, input_vars=self.input_vars, positional_vars=self.positional_vars,
                                       target=self.target,
                                       mean_std_dict=self.mean_std_dict, task=self.task)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True
        )
