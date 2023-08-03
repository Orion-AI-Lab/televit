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
from .components.seasfire_dataset_oci import BatcherDS_with_ocis, \
    sample_dataset_with_ocis


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


class SeasFireSpatialOciDataModule(LightningDataModule):
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
            oci_vars: list = None,
            oci_lag: int = 10,
            target: str = 'BurntArea',
            target_shift: int = 1,
            random_crop: bool = False,
            patch_size: list = None,
            batch_size: int = 64,
            num_workers: int = 8,
            pin_memory: bool = False,
            debug: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        if patch_size is None:
            patch_size = [1, 128, 128]
        if positional_vars is None:
            self.positional_vars = []
        else:
            self.positional_vars = positional_vars
        self.save_hyperparameters(logger=False)
        self.ds_path = ds_path
        self.input_vars = list(input_vars)
        self.oci_vars = list(oci_vars)
        self.oci_lag = oci_lag
        self.target = target
        self.target_shift = target_shift
        self.ds = xr.open_zarr(ds_path, consolidated=True)
        # TODO remove when we have the new datacube
        self.ds['sst'] = self.ds['sst'].where(self.ds['sst'] >= 0)
        self.mean_std_dict = None
        if debug:
            self.num_timesteps = 5
        else:
            self.num_timesteps = -1
        self.random_crop = random_crop
        self.patch_size = tuple(patch_size)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

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
            # train_patch_size = 160 if self.random_crop else 128
            # val_patch_size = 128
            train_batches, train_oci_batches, self.mean_std_dict = sample_dataset_with_ocis(self.ds.copy(),
                                                                                            input_vars=self.input_vars,
                                                                                            oci_vars=self.oci_vars,
                                                                                            oci_lag=self.oci_lag,
                                                                                            target=self.target,
                                                                                            target_shift=-self.target_shift,
                                                                                            split='train',
                                                                                            num_timesteps=self.num_timesteps,
                                                                                            patch_size=self.patch_size)
            # Save mean std dict for normalization during inference time
            print(self.mean_std_dict)
            with open(f'mean_std_dict_{self.target_shift}.json', 'w') as f:
                f.write(json.dumps(self.mean_std_dict))

            val_batches, val_oci_batches, _ = sample_dataset_with_ocis(self.ds.copy(), input_vars=self.input_vars,
                                                                       target=self.target, oci_vars=self.oci_vars,
                                                                       oci_lag=self.oci_lag,
                                                                       target_shift=-self.target_shift, split='val',
                                                                       num_timesteps=self.num_timesteps,
                                                                       patch_size=self.patch_size)

            test_batches, test_oci_batches, _ = sample_dataset_with_ocis(self.ds.copy(), input_vars=self.input_vars,
                                                                         target=self.target, oci_vars=self.oci_vars,
                                                                         oci_lag=self.oci_lag,
                                                                         target_shift=-self.target_shift, split='test',
                                                                         num_timesteps=self.num_timesteps,
                                                                         patch_size=self.patch_size)

            self.data_train = BatcherDS_with_ocis(train_batches, input_vars=self.input_vars,
                                                  positional_vars=self.positional_vars, target=self.target,
                                                  oci_vars=self.oci_vars, oci_batches=train_oci_batches,
                                                  oci_lag=self.oci_lag, mean_std_dict=self.mean_std_dict,
                                                  random_crop=self.random_crop)
            self.data_val = BatcherDS_with_ocis(val_batches, input_vars=self.input_vars,
                                                positional_vars=self.positional_vars, target=self.target,
                                                oci_vars=self.oci_vars, oci_batches=val_oci_batches,
                                                oci_lag=self.oci_lag, mean_std_dict=self.mean_std_dict,
                                                random_crop=False)
            #
            # # function to filter val_batches based on mode value of gfed_region
            # def filter_by_region(batches, region):
            #     return [batch for batch in batches if batch['gfed_region'].median().item() == region]
            #
            # self.data_val = [
            #     BatcherDS_with_ocis(filter_by_region(val_batches, i), input_vars=self.input_vars, target=self.target,
            #                         oci_vars=self.oci_vars, oci_batches=val_oci_batches,
            #                         oci_lag=self.oci_lag, mean_std_dict=self.mean_std_dict,
            #                         random_crop=False) for i in range(1, 15)]
            self.data_test = BatcherDS_with_ocis(test_batches, input_vars=self.input_vars,
                                                 positional_vars=self.positional_vars, target=self.target,
                                                 oci_vars=self.oci_vars, oci_batches=test_oci_batches,
                                                 oci_lag=self.oci_lag, mean_std_dict=self.mean_std_dict,
                                                 random_crop=False)

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
