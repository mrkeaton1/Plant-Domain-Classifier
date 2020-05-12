"""
Code to generate .pt files of dataset and export list of partition and IDs
Created by Matthew Keaton on 4/22/2020
"""

import os
import torch
import random
from PIL import Image
from torchvision.transforms import ToTensor
import pickle
from time import time
from utils import elapsed_time

categories = ['leaves', 'branches', 'trees']


def generate_tensor_data(data_path):

    # First remove all .pt files, in case .jpg files have been added, moved, or removed
    print('Clearing current .pt files:')
    os.chdir(data_path)
    for species_dir in os.listdir():
        if os.path.isdir(species_dir):
            for domain in categories:
                for file in os.listdir(os.path.join(species_dir, domain)):
                    if file.endswith('.pt'):
                        os.remove(os.path.join(data_path, species_dir, domain, file))

    data_ids = []
    labels = {}

    count = 1
    print('Generating tensor data:')
    for species_dir in os.listdir():
        if os.path.isdir(species_dir):
            print(str(count) + ": " + species_dir)
            count += 1
            for domain in categories:
                for im in os.listdir(os.path.join(species_dir, domain)):
                    if im.endswith('.jpg'):
                        data_id = os.path.join(species_dir, domain, im[:-4] + '.pt')
                        data_ids.append(data_id)
                        labels[data_id] = categories.index(domain)

                        image = Image.open(os.path.join(species_dir, domain, im))
                        image = image.resize((224, 224))
                        data = ToTensor()(image)
                        pt_filename = os.path.join(data_path, data_id)
                        torch.save(data, pt_filename)

    pickle.dump(labels, open(os.path.join(data_path, 'label_list.p'), 'wb'))
    pickle.dump(data_ids, open(os.path.join(data_path, 'pre-partition_ids.p'), 'wb'))
    return data_ids


def generate_partition_data(data_ids, train_ratio):
    random.shuffle(data_ids)  # Another way to shuffle?
    partition = {'train': data_ids[0:round(len(data_ids)*train_ratio)],
                 'test': data_ids[round(len(data_ids)*train_ratio):]}
    pickle.dump(partition, open('partition_dict.p', 'wb'))
    print('Completed.')


if __name__ == '__main__':
    start = time()
    data_dir = '/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)'
    d_ids = generate_tensor_data(data_dir)
    generate_partition_data(d_ids, 0.8)
    print('Elapsed time: {}'.format(elapsed_time((time() - start))))
    # partition = pickle.load(open(os.path.join(data_dir, 'partition_dict.p'), 'rb'))
    # labels = pickle.load(open(os.path.join(data_dir, 'label_list.p'), 'rb'))
