import xbatcher
import xarray as xr
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torchvision import transforms
import random


def sample_dataset_with_ocis(ds, input_vars, oci_vars, oci_lag, target, target_shift, patch_size=(2, 160, 160),
                             split='train', num_timesteps=-1):
    """
    :param ds: xarray dataset
    :param input_vars: list of input variables
    :param oci_vars: list of oci variables
    :param oci_lag: lag of oci variables
    :param target: target variable
    :param target_shift: shift of target variable
    :param patch_size: patch size
    :param split: train, val, test
    :param num_timesteps: number of timesteps
    :return: list of batches, list of oci_batches, mean_std_dict
    """
    print(f'Shifting inputs by {-target_shift}')
    for var in input_vars + oci_vars:
        if target_shift < 0:
            ds[var] = ds[var].shift(time=-target_shift)
    oci_ds = xr.Dataset()
    for var in oci_vars:
        # resample var to 1 month
        oci_ds[var] = ds[var].fillna(0).resample(time='1M').mean(dim='time')
    oci_ds.load()
    ds['oci_pdo'] = ds['oci_pdo'].where(ds['oci_pdo'] > -9).ffill(dim='time')
    ds['oci_epo'] = ds['oci_epo'].where(ds['oci_epo'] > -90).ffill(dim='time')
    print('Oceanic indices dataset loaded...')
    mean_std_dict = {}
    for var in oci_vars:
        mean_std_dict[var + '_mean'] = oci_ds[var].mean().values.item(0)
        mean_std_dict[var + '_std'] = oci_ds[var].std().values.item(0)
    print('Oci means calculated')
    # print(f'Shifting {target} by {target_shift}')
    # if target_shift < 0:
    #     ds[target] = ds[target].shift(time=target_shift)

    if split == 'train':
        ds = ds.sel(time=slice('2003-01-01', '2018-01-01'))
    elif split == 'val':
        ds = ds.sel(time=slice('2018-01-01', '2019-01-01'))
    elif split == 'test':
        ds = ds.sel(time=slice('2019-01-01', '2020-01-01'))

    if num_timesteps > 0:
        ds = ds.isel(time=slice(0, num_timesteps - 1))

    ds = ds[input_vars + [target]]
    ds = ds.load()
    print("Dataset loaded")
    for var in input_vars + [target]:
        mean_std_dict[var + '_mean'] = ds[var].mean().values.item(0)
        mean_std_dict[var + '_std'] = ds[var].std().values.item(0)
    print('Means calculated')
    bgen = xbatcher.BatchGenerator(
        ds=ds,
        input_dims={'longitude': patch_size[1], 'latitude': patch_size[2], 'time': patch_size[0]},
        input_overlap={'time': patch_size[0] - 1} if (patch_size[0] - 1) else {}
    )

    # compute positional embedding from longitude and latitude
    lon = ds.longitude.values
    lat = ds.latitude.values
    lon = np.expand_dims(lon, axis=0)
    lat = np.expand_dims(lat, axis=1)
    lon = np.tile(lon, (lat.shape[0], 1))
    lat = np.tile(lat, (1, lon.shape[1]))

    ds['cos_lon'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.cos(lon * np.pi / 180))
    ds['cos_lat'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.cos(lat * np.pi / 180))
    ds['sin_lon'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.sin(lon * np.pi / 180))
    ds['sin_lat'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.sin(lat * np.pi / 180))
    # if log_tp:
    ds['tp'] = np.log(ds['tp'] + 1)

    ds[target] = np.log(1+ ds[target])

    n_batches = 0
    n_pos = 0
    batches = []
    oci_batches = []
    # negatives = []
    # batches = []
    # mean_std_dict = {}
    # for var in input_vars + [target]:
    #     mean_std_dict[var + '_mean'] = ds[var].mean().values.item(0)
    #     mean_std_dict[var + '_std'] = ds[var].std().values.item(0)
    for batch in tqdm(bgen):
        if batch.isel(time=-1)[target].sum() > 0:
            batches.append(batch)
            # select oci_ds from lag months before until the batch time
            oci_batch = oci_ds.sel(time=slice(batch.time[-1] - np.timedelta64(oci_lag * 31, 'D'), batch.time[-1]))
            oci_batches.append(oci_batch)
            n_pos += 1
        #         else:
        #             if not np.isnan(batch.isel(time=-1)['NDVI']).sum()>0:
        #                 negatives.append(batch)
        n_batches += 1
    print('# of batches', n_batches)
    print('# of positives', len(batches))
    print('# of oci_batches', len(oci_batches))
    return batches, oci_batches, mean_std_dict


# ['random crop numpy array'] of shape (C, H, W) to (C, H', W') and apply same transformation to label
def random_crop_fn(img, label, crop_size):
    h, w = img.shape[1:]
    new_h, new_w = crop_size
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    img = img[:, top: top + new_h, left: left + new_w]
    label = label[top: top + new_h, left: left + new_w]
    return img, label


class BatcherDS_with_ocis(Dataset):
    """Dataset from Xbatcher"""

    def __init__(self, batches, input_vars, positional_vars, oci_batches, oci_vars, oci_lag, target, mean_std_dict,
                 random_crop=True, task='classification', nanfill = -1.0, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.task = task
        assert self.task in ['classification', 'regression']
        self.positional_vars = positional_vars
        self.batches = batches
        self.target = target
        self.input_vars = input_vars
        self.oci_batches = oci_batches
        self.oci_vars = oci_vars
        self.mean_std_dict = mean_std_dict
        self.random_crop = random_crop
        self.transform = lambda img, label: random_crop_fn(img, label, (128, 128))
        self.oci_lag = oci_lag
        self.nanfill = nanfill
        self.mean = np.stack([mean_std_dict[f'{var}_mean'] for var in input_vars])
        self.std = np.stack([mean_std_dict[f'{var}_std'] for var in input_vars])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        def _normalize(x, var, mean_std_dict):
            return (x - mean_std_dict[f'{var}_mean']) / mean_std_dict[f'{var}_std']

        # function to divide by std
        def _divide_by_std(x, var, mean_std_dict):
            return x / mean_std_dict[f'{var}_std']

        batch = self.batches[idx]
        # stack postional variables
        pos_vars = np.stack([batch[var].values for var in self.positional_vars], axis=0)
        # stack input variables
        inputs = np.stack([_normalize(batch[var], var, self.mean_std_dict) for var in self.input_vars], axis=0)
        c, t, h, w = inputs.shape
        inputs = inputs.reshape((t * c, h, w))
        # concatenate inputs with pos_vars
        inputs = np.concatenate([inputs, pos_vars], axis=0).astype(np.float32)

        t_batch = self.oci_batches[idx].isel(time=slice(-self.oci_lag, None))
        t_inputs = np.stack([_divide_by_std(t_batch[var], var, self.mean_std_dict) for var in self.oci_vars])

        target = batch.isel(time=-1)[self.target].values
        inputs = np.nan_to_num(inputs, nan=self.nanfill)
        target = np.nan_to_num(target, nan=0)
        if self.random_crop:
            inputs, target = self.transform(inputs, target)
        if self.task == 'classification':
            target = np.where(target > 0, 1, 0)

        return inputs, t_inputs, target

