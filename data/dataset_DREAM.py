import glob
import os
import random

import imageio.v2 as imageio
import numpy as np
import skimage.color as sc
import torch
import torch.utils.data as data


def set_channel(img, n_channels=3):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    channels = img.shape[2]
    if n_channels == 1 and channels == 3:
        img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
    elif n_channels == 3 and channels == 1:
        img = np.concatenate([img] * n_channels, 2)
    return img


def np2tensor(img, rgb_range=255):
    array = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(array).float()
    tensor.mul_(rgb_range / 255)
    return tensor


class SRData(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.train = args['phase'] == 'train'
        self.dir_hr = args['dataroot_H']
        self.dir_lr = args['dataroot_L']

        lr_paths = sorted(glob.glob(os.path.join(self.dir_lr, '*.tif')))
        self.images_lr = []
        self.images_hr = []
        for lr_path in lr_paths:
            paired_lr = lr_paths[random.randint(0, len(lr_paths) - 1)]
            self.images_lr.append([lr_path, paired_lr])
            self.images_hr.append(
                os.path.join(self.dir_hr, os.path.basename(lr_path))
            )

    def __getitem__(self, index):
        if self.train:
            lr_images = [imageio.imread(path) for path in self.images_lr[index]]
            lr_tensors = [
                np2tensor(
                    set_channel(image, self.args['n_colors']),
                    self.args['rgb_range'],
                )
                for image in lr_images
            ]
            hr = imageio.imread(self.images_hr[index])
            hr_tensor = np2tensor(
                set_channel(hr, self.args['n_colors']),
                self.args['rgb_range'],
            )
            return {'L': torch.stack(lr_tensors, 0), 'H': hr_tensor}

        lr = imageio.imread(self.images_lr[index][0])
        hr = imageio.imread(self.images_hr[index])
        lr_tensor = np2tensor(
            set_channel(lr, self.args['n_colors']),
            self.args['rgb_range'],
        )
        hr_tensor = np2tensor(
            set_channel(hr, self.args['n_colors']),
            self.args['rgb_range'],
        )
        return {
            'L': lr_tensor,
            'H': hr_tensor,
            'L_path': self.images_lr[index][0],
            'H_path': self.images_hr[index],
        }

    def __len__(self):
        return len(self.images_hr)
