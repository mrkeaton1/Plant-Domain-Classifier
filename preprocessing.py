"""
Module for data augmentation and any other preprocessing

Created by Matthew Keaton on 4/14/2020
"""

import torchvision
import numpy as np

from loaddataset import DomainData
import pickle
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class RandomCropToScale(object):
    """Randomly square crop image based on proportion of smaller dimension size.

    Args:
        crop_ratio (float): Ratio of crop in relation to smaller dimension. Should be 0 < crop_ratio < 1.
    """

    def __init__(self, crop_ratio):
        assert isinstance(crop_ratio, float)
        self.crop_ratio = crop_ratio

    def __call__(self, sample):
        h = sample.shape[1]
        w = sample.shape[2]
        min_dim = min(h, w)
        new_dim = int(min_dim * self.crop_ratio)
        top = np.random.randint(0, h - new_dim)
        left = np.random.randint(0, w - new_dim)
        sample = sample[:, top: top+new_dim, left: left+new_dim]
        return sample


base_transform = torchvision.transforms.Compose([
                                RandomCropToScale(0.8),
                                torchvision.transforms.ToPILImage(),
                                torchvision.transforms.Resize((224, 224)),
                                torchvision.transforms.RandomHorizontalFlip(0.5),
                                torchvision.transforms.ToTensor()
                                ])

distillation_transform1 = torchvision.transforms.Compose([
                                torchvision.transforms.ToPILImage(),
                                torchvision.transforms.RandomHorizontalFlip(1.0),
                                torchvision.transforms.ToTensor()
                                ])

distillation_transform2 = RandomCropToScale(0.8)

distillation_transform3 = RandomCropToScale(0.6)

dist_transforms = [distillation_transform1, distillation_transform2, distillation_transform3]


# if __name__ == '__main__':
#
# data_dir = "/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)"
# partition = pickle.load(open(os.path.join(data_dir, 'partition_dict.p'), 'rb'))
# labels = pickle.load(open(os.path.join(data_dir, 'label_list.p'), 'rb'))
# dataset = DomainData(partition['train'], labels, data_dir, transform=None)
# generator = DataLoader(dataset, batch_size=128, shuffle=True)
# sample = dataset[1][0]
# plt.figure()
# sample = sample.transpose(0,2)
# sample = sample.transpose(0,1)
# plt.imshow(sample)
# plt.show()
#
# dataset_t1 = DomainData(partition['train'], labels, data_dir, transform=base_transform)
# sample_t1 = dataset_t1[1][0]
# plt.figure()
# sample_t1 = sample_t1.transpose(0,2).transpose(0,1)
# plt.imshow(sample_t1)
# plt.show()
#
# dataset_t2 = DomainData(partition['train'], labels, data_dir, transform=distillation_transform1)
# sample_t2 = dataset_t2[1][0]
# plt.figure()
# sample_t2 = sample_t2.transpose(0,2).transpose(0,1)
# plt.imshow(sample_t2)
# plt.show()
#
# dataset_t3 = DomainData(partition['train'], labels, data_dir, transform=distillation_transform2)
# sample_t3 = dataset_t3[1][0]
# plt.figure()
# sample_t3 = sample_t3.transpose(0,2).transpose(0,1)
# plt.imshow(sample_t3)
# plt.show()
#
# dataset_t4 = DomainData(partition['train'], labels, data_dir, transform=distillation_transform3)
# sample_t4 = dataset_t4[1][0]
# plt.figure()
# sample_t4 = sample_t4.transpose(0,2).transpose(0,1)
# plt.imshow(sample_t4)
# plt.show()
