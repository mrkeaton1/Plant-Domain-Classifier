"""
Data Loader implementation for loading domain data in plant dataset
Created by Matthew Keaton on 4/14/2020
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import imghdr

categories = ['leaves', 'branches', 'trees']


def get_labelspace_size():
    return len(categories)


# Assumes input is less than the size of the label space
def one_hot(in_label):
    output = torch.zeros([1, get_labelspace_size()])
    output[0][in_label] = 1
    return output


def randomize(data, labels):
    size = len(data)
    new_data = []
    new_labels = []
    for i in np.random.permutation(size):
        new_data.append(data[i])
        new_labels.append(labels[i])
    return new_data, new_labels


# split data into train and test set
def split(data, labels, train_ratio):
    size = len(data)
    train_split = int(size*train_ratio)
    train_data = data[0:train_split]
    test_data = data[train_split:]
    train_labels = labels[0:train_split]
    test_labels = labels[train_split:]
    return train_data, train_labels, test_data, test_labels


# Use for specifics for this dataset and extraction method
def load_data(data_path):
    data = []
    labels = []
    os.chdir(data_path)
    for species_dir in os.listdir():
        if os.path.isdir(species_dir):

            for domain in categories:
                for im in os.listdir(os.path.join(species_dir, domain)):
                    if im.endswith('.jpg'):
                        data.append(os.path.join(data_path, species_dir, domain, im))
                        labels.append(torch.tensor([categories.index(domain)]))

    # print('Data loaded.')
    return data, labels


# Make as generalizable as possible
class DomainData(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data, self.labels = data, labels
        self.size = len(data)  # Unsure if this is best way to implement size
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.data[index])
        if imghdr.what(self.data[index]) != 'jpeg':
            print(self.data[index])
            print(imghdr.what(self.data[index]))
        if self.transform is not None:
            return self.transform(image), self.labels[index]
        return image, self.labels[index]


if __name__ == '__main__':
    d, l = load_data('/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset')
    d, l = randomize(d, l)
    train_d, train_l, test_d, test_l = split(d, l, 0.7)
    train = DomainData(train_d, train_l)
    test = DomainData(test_d, test_l)
