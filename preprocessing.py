"""
Module for data augmentation and any other preprocessing
Created by Matthew Keaton on 4/14/2020
"""

import torchvision

base_transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToPILImage(),
                                torchvision.transforms.RandomCrop((112, 112)),
                                # torchvision.transforms.Resize((224, 224)),
                                torchvision.transforms.RandomHorizontalFlip(0.5),
                                torchvision.transforms.ToTensor()
                                ])
