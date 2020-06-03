"""
Data Loader implementation for loading domain data in plant dataset

Created by Matthew Keaton on 4/14/2020
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os

categories = ['leaves', 'branches', 'trees']


def get_labelspace_size():
    return len(categories)


# Currently unused
def randomize(data, labels):
    size = len(data)
    new_data = []
    new_labels = []
    for i in np.random.permutation(size):
        new_data.append(data[i])
        new_labels.append(labels[i])
    return new_data, new_labels


class DomainData(Dataset):
    def __init__(self, list_ids, labels, data_dir, transform=None):
        self.labels = labels
        self.list_ids = list_ids
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        data_id = self.list_ids[index]
        X = torch.load(os.path.join(self.data_dir, data_id))
        y = self.labels[data_id]
        return X, y

    def get_sample_jpg_id(self, index):
        return self.list_ids[index][:-3] + '.jpg'
