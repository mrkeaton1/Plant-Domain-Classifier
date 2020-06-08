"""
Module for data augmentation and any other preprocessing

Created by Matthew Keaton on 4/14/2020
"""

import torchvision

from loaddataset import DomainData
import pickle
import os
from torch.utils.data import DataLoader

# class Crop_To_Scale(object):
#     """Randomly crop image based on smaller dimension size.
#
#     Args:
#         crop_ratio (float): Ratio of crop in relation to smaller dimension. Should be 0 < crop_ratio < 1.
#     """
#
#     def __init__(self, crop_ratio):
#         assert isinstance(crop_ratio, float)
#         self.crop_ratio = crop_ratio
#
#     # def __call__(self, sample):
#     #     image,


base_transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToPILImage(),
                                torchvision.transforms.RandomCrop((112, 112)),
                                torchvision.transforms.Resize((224, 224)),
                                torchvision.transforms.RandomHorizontalFlip(0.5),
                                torchvision.transforms.ToTensor()
                                ])

distillation_transform1 = torchvision.transforms.Compose([
                                torchvision.transforms.ToPILImage(),
                                torchvision.transforms.RandomCrop((112, 112)),
                                torchvision.transforms.RandomHorizontalFlip(0.5),
                                torchvision.transforms.ToTensor()
                                ])

distillation_transform2 = torchvision.transforms.Compose([
                                torchvision.transforms.ToPILImage(),
                                torchvision.transforms.RandomCrop((112, 112)),
                                torchvision.transforms.RandomHorizontalFlip(0.5),
                                torchvision.transforms.ToTensor()
                                ])

distillation_transform3 = torchvision.transforms.Compose([
                                torchvision.transforms.ToPILImage(),
                                torchvision.transforms.RandomCrop((112, 112)),
                                torchvision.transforms.RandomHorizontalFlip(0.5),
                                torchvision.transforms.ToTensor()
                                ])

dist_transforms = [distillation_transform1, distillation_transform2, distillation_transform3]


# if __name__ == '__main__':
#
#     data_dir = "/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)"
#     partition = pickle.load(open(os.path.join(data_dir, 'partition_dict.p'), 'rb'))
#     labels = pickle.load(open(os.path.join(data_dir, 'label_list.p'), 'rb'))
#     dataset = DomainData(partition['train'], labels, data_dir, transform=None)
#     generator = DataLoader(dataset, batch_size=128, shuffle=True)

