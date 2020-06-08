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


def generate_tensor_data(data_path, cd):

    # First remove all .pt files, in case .jpg files have been added, moved, or removed
    print('Clearing current .pt files:')
    os.chdir(data_path)
    for species_dir in os.listdir():
        if os.path.isdir(species_dir):
            for domain in categories:
                for file in os.listdir(os.path.join(species_dir, domain)):
                    if file.endswith('.pt'):
                        os.remove(os.path.join(data_path, species_dir, domain, file))
    if os.path.exists('label_list.p'):
        os.remove('label_list.p')
    if os.path.exists('partition_dict.p'):
        os.remove('partition_dict.p')
    if os.path.exists('pre-partition_ids.p'):
        os.remove('pre-partition_ids.p')

    if cd.lower() == 'y':
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
                            # Removed in order for augmentation to occur on entire image
                            # image = image.resize((224, 224))
                            data = ToTensor()(image)
                            pt_filename = os.path.join(data_path, data_id)
                            torch.save(data, pt_filename)

        pickle.dump(labels, open(os.path.join(data_path, 'label_list.p'), 'wb'))
        pickle.dump(data_ids, open(os.path.join(data_path, 'pre-partition_ids.p'), 'wb'))
        return data_ids
    else:
        print('Skipping Data Creation.')


def generate_partition_data(data_ids, train_ratio):
    random.shuffle(data_ids)  # Another way to shuffle?
    partition = {'train': data_ids[0:round(len(data_ids)*train_ratio)],
                 'test': data_ids[round(len(data_ids)*train_ratio):]}
    pickle.dump(partition, open('partition_dict.p', 'wb'))
    print('Completed.')


if __name__ == '__main__':
    create_data = input('Type \'Y\' if you would like to create data,'
                        ' or \'n\' if you would like to just clear all.\n>>> ')
    start = time()
    data_dir = '/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)'
    d_ids = generate_tensor_data(data_dir, create_data)
    if create_data.lower() == 'y':
        generate_partition_data(d_ids, 0.8)
    print('Elapsed time: {}'.format(elapsed_time((time() - start))))
