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