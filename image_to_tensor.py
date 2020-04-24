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

categories = ['leaves', 'branches', 'trees']


def generate_tensor_data(data_path):

    data_ids = []
    labels = {}
    os.chdir(data_path)

    count = 1
    for species_dir in os.listdir():
        print(str(count) + ": " + species_dir)
        count += 1

        if os.path.isdir(species_dir):
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
    data_dir = '/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset'
    d_ids = generate_tensor_data(data_dir)
    generate_partition_data(d_ids, 0.8)
    # partition = pickle.load(open(os.path.join(data_dir, 'partition_dict.p'), 'rb'))
    # labels = pickle.load(open(os.path.join(data_dir, 'label_list.p'), 'rb'))
