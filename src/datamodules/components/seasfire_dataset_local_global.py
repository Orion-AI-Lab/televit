import xbatcher
import xarray as xr
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torchvision import transforms
import random

# save dictionary to json file in path
import json


def load_global_ds(ds_path, input_vars, log_transform_vars, target, target_shift):
    ds = xr.open_zarr(ds_path)
    for var in log_transform_vars:
        ds[var] = np.log(ds[var] + 1)
    ds = ds[input_vars + [target]].load()
    for var in input_vars:
        if target_shift < 0:
            ds[var] = ds[var].shift(time=-target_shift)
    for var in input_vars:
        ds[var] = (ds[var] - ds[var].mean()) / ds[var].std()
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
    return ds


def save_dict_to_json(d, path):
    with open(path, 'w') as fp:
        json.dump(d, fp)


def sample_dataset_with_ocis(ds, input_vars, oci_vars, oci_lag, target, log_transform_vars, target_shift,
                             patch_size=(1, 80, 80),
                             split='train', num_timesteps=-1, stats_dir=None):
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
    
    flag_mean_calculate = False
    mean_std_dict = {}
    mean_std_dict_path = Path(stats_dir) / f'mean_std_dict_{target_shift}.json'
    if mean_std_dict_path.exists():
        with open(mean_std_dict_path, 'r') as fp:
            mean_std_dict = json.load(fp)
        # check if all variables are in mean_std_dict
        for var in input_vars + oci_vars + [target]:
            if var + '_mean' not in mean_std_dict:
                mean_std_dict = {}
                flag_mean_calculate = True
                break
    else:
        flag_mean_calculate = True
    
    if flag_mean_calculate:
        for var in oci_vars:
            mean_std_dict[var + '_mean'] = oci_ds[var].mean().values.item(0)
            mean_std_dict[var + '_std'] = oci_ds[var].std().values.item(0)
    print('Oci means calculated')
    # print(f'Shifting {target} by {target_shift}')
    # if target_shift < 0:
    #     ds[target] = ds[target].shift(time=target_shift)

    if split == 'train':
        ds = ds.sel(time=slice('2002-01-01', '2018-01-01'))
    elif split == 'val':
        ds = ds.sel(time=slice('2018-01-01', '2019-01-01'))
    elif split == 'test':
        ds = ds.sel(time=slice('2019-01-01', '2020-01-01'))

    if num_timesteps > 0:
        ds = ds.isel(time=slice(0, num_timesteps - 1))

    for var in log_transform_vars:
        ds[var] = np.log(ds[var] + 1)

    ds = ds[input_vars + [target]]
    ds = ds.load()
    print("Dataset loaded")
    if flag_mean_calculate:
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


class BatcherDS_global_local(Dataset):
    """Dataset from Xbatcher"""

    def __init__(self, ds_global, batches, input_vars, positional_vars, oci_batches, oci_vars, oci_lag, target,
                 mean_std_dict, task='classification', nanfill=-1.):
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
        self.oci_lag = oci_lag
        self.mean = np.stack([mean_std_dict[f'{var}_mean'] for var in input_vars])
        self.std = np.stack([mean_std_dict[f'{var}_std'] for var in input_vars])
        self.ds_global = ds_global
        self.nanfill = nanfill

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

        # stack input variables
        inputs = np.stack([_normalize(batch[var], var, self.mean_std_dict) for var in self.input_vars], axis=0)
        c, t, h, w = inputs.shape
        inputs = inputs.reshape((t * c, h, w))
        # concatenate inputs with pos_vars
        if self.positional_vars:
            pos_vars = np.stack([batch[var].values for var in self.positional_vars], axis=0)
            inputs = np.concatenate([inputs, pos_vars], axis=0).astype(np.float32)

        if self.ds_global:
            global_ds = self.ds_global.sel(time=batch.isel(time=-1).time.values)
            global_inputs = np.stack([global_ds[var] for var in self.input_vars], axis=0)
            global_target = global_ds[self.target].values
            global_inputs = np.nan_to_num(global_inputs, nan=self.nanfill)
            if self.positional_vars:
                global_pos_vars = np.stack([global_ds[var].values for var in self.positional_vars], axis=0)
                global_inputs = np.concatenate([global_inputs, global_pos_vars], axis=0).astype(np.float32)
        else:
            global_inputs = 0.
            global_target = 0.    


        t_batch = self.oci_batches[idx].isel(time=slice(-self.oci_lag, None))
        t_inputs = np.stack([_divide_by_std(t_batch[var], var, self.mean_std_dict) for var in self.oci_vars])

        target = batch.isel(time=-1)[self.target].values
        inputs = np.nan_to_num(inputs, nan=-1)
        target = np.nan_to_num(target, nan=0)
        mask = np.isnan(batch.isel(time=-1)['ndvi']).values

        if self.task == 'classification':
            target = np.where(target > 0, 1, 0)
            global_target = np.where(global_target > 0, 1, 0)

        return {
            'x_local': inputs,
            'x_local_mask': mask,
            'x_oci': t_inputs,
            'x_global': global_inputs,
            'y_local': target,
            'y_global': global_target
        }
