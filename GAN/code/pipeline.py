# ##################################################################### #
# Modifications authored by:
# Peikai Li
# Ipek Ilayda Onur
#
# Most of the code is borrowed from:
# https://github.com/JeongHyunJin/Jeong2020_SolarFarsideMagnetograms/blob/master/pipeline.py
# ##################################################################### #

import os
from astropy.io import fits
from os.path import split, splitext
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate
from random import randint

"""
    loads from npz file
"""
def loadDataset(opt, filename):
    maps      = np.load(filename)
    img_shape = opt.img_shape
    num_train, num_val, num_test = opt.splits
    train_label_maps = np.concatenate(
        (maps['q_obs'][num_train:num_train + num_val].reshape((num_train,) + img_shape),
         maps['u_obs'][num_train:num_train + num_val].reshape((num_train,) + img_shape)), xis=1)
    val_label_maps = np.concatenate(
        (maps['q_obs'][num_train:num_train + num_val].reshape((num_val,) + img_shape),
         maps['u_obs'][num_train:num_train + num_val].reshape((num_val,) + img_shape)), xis=1)
    test_label_maps = np.concatenate(
        (maps['q_obs'][num_train + num_val:].reshape((num_test,) + img_shape),
         maps['u_obs'][num_train + num_val:].reshape((num_test,) + img_shape)), axis=1)

    train_target_maps = maps['tru_kappa'][:num_train]
    val_target_maps = maps['tru_kappa'][num_train:num_train + num_val]
    test_target_maps = maps['tru_kappa'][num_train + num_val:]

    q_obs_mean = np.mean(train_label_maps[:, 0, :, :])
    u_obs_mean = np.mean(train_label_maps[:, 1, :, :])

    q_obs_std = np.std(train_label_maps[:, 0, :, :])
    u_obs_std = np.std(train_label_maps[:, 1, :, :])

    tru_kappa_std = np.std(train_target_maps[:, :, :])

    data = {
        'train_target_maps': train_target_maps,
        'train_label_maps': train_label_maps,
        'val_target_maps': val_target_maps,
        'val_label_maps': val_label_maps,
        'test_target_maps': test_target_maps,
        'test_label_maps': test_label_maps,
        'q_obs_mean': q_obs_mean,
        'u_obs_mean': u_obs_mean,
        'q_obs_std': q_obs_std,
        'u_obs_std': u_obs_std,
        'tru_kappa_std': tru_kappa_std

    }
    return data

class CustomDataset(Dataset):
    def __init__(self, opt, dataset = None):
        super(CustomDataset, self).__init__()
        self.opt         = opt
        self.dataset     = dataset
        self.dataset_dir = opt.dataset_dir
        self.input_format = opt.data_format_input
        self.target_format = opt.data_format_target

        if dataset is not None:
            self.maps = dataset
        if opt.is_train:
            self.label_path_list = np.arange(36000)
            self.target_path_list = np.arange(36000)

        elif opt.is_val:
            self.label_path_list = np.arange(2000)
            self.target_path_list = np.arange(2000)
        else:
            self.label_path_list = np.arange(2000)
            self.target_path_list = np.arange(2000)

    def __getitem__(self, index):
        list_transforms = []
        list_transforms += []

        # [ Training data ] ==============================================================================================
        if self.opt.is_train:
            self.angle = randint(-self.opt.max_rotation_angle, self.opt.max_rotation_angle)

            self.offset_x = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0
            self.offset_y = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0

            # [ Input ] ==================================================================================================

            IMG_A0 = self.maps.train_label_maps[index]

            IMG_A0[np.isnan(IMG_A0)] = 1
            label_array = IMG_A0
            label_tensor = torch.tensor(label_array)

            if len(label_tensor.shape) == 2:
                label_tensor = label_tensor.unsqueeze(dim=0)

            # [ Target ] ==================================================================================================

            IMG_B0 = self.maps.train_target_maps[index]

            IMG_B0[np.isnan(IMG_B0)] = 0
            target_array = IMG_B0
            target_tensor = torch.tensor(target_array, dtype=torch.float32)

            if len(target_tensor.shape) == 2:
                target_tensor = target_tensor.unsqueeze(dim=0)  # Add channel dimension.

        # [ Test data ] ===================================================================================================
        else:
            # [ Input ] ==================================================================================================
            if self.opt.is_val:
                IMG_A0 = self.maps.val_label_maps[index]
            else:
                IMG_A0 = self.maps.test_label_maps[index]

            IMG_A0[np.isnan(IMG_A0)] = 1
            label_array = IMG_A0
            label_tensor = torch.tensor(label_array, dtype=torch.float32)

            if len(label_tensor.shape) == 2:
                label_tensor = label_tensor.unsqueeze(dim=0)

            # [ Target ] ==================================================================================================
            if self.opt.is_val:
                IMG_B0 = self.maps.val_target_maps[index]
            else:
                IMG_B0 = self.maps.test_target_maps[index]

            IMG_B0[np.isnan(IMG_B0)] = 0
            target_array = IMG_B0
            target_tensor = torch.tensor(target_array, dtype=torch.float32)

            if len(target_tensor.shape) == 2:
                target_tensor = target_tensor.unsqueeze(dim=0)  # Add channel dimension.
            return label_tensor, target_tensor
        return label_tensor, target_tensor

    def __random_crop(self, x):
        x = np.array(x)
        x = x[self.offset_x: self.offset_x + 1024, self.offset_y: self.offset_y + 1024]
        return x

    @staticmethod
    def __pad(x, padding_size):
        if type(padding_size) == int:
            if len(x.shape) == 3:
                padding_size = ((0, 0), (padding_size, padding_size), (padding_size, padding_size))
            else:
                padding_size = ((padding_size, padding_size), (padding_size, padding_size))
        return np.pad(x, pad_width=padding_size, mode="constant", constant_values=0)

    def __rotate(self, x):
        return rotate(x, self.angle, reshape=False)

    @staticmethod
    def __to_numpy(x):
        return np.array(x, dtype=np.float32)

    def __len__(self):
        return len(self.label_path_list)