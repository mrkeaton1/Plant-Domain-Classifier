"""
Utilizes model trained on manually labelled dataset to generate labels for unlabelled data.

See Data Distillation: Towards Omni-Supervised Learning - https://arxiv.org/abs/1712.04440
Created by Matthew Keaton on 6/1/2020
"""

import sys
import os
import torch
import numpy as np
import pickle
from model_init import init_model
from torch.utils.data import DataLoader
from loaddataset import DomainData
from preprocessing import dist_transforms

### Subject to change: Initial use only
data_dir = "/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)"
# model_path = sys.argv[1]
model_path = "/home/mrkeaton/Documents/Plant_Code/Plant-Domain-Classifier/Results/abc1/resnet-18_pretrained_epochs=1_lr=0.02_mom=0.5_batchsize=32-32/resnet-18_model.pt"
modelname = "resnet-18"
pretrained = True
batch_size = 128
n_epochs = 1
learning_rate = 0.02
momentum = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = init_model(modelname, pretrained, n_epochs, learning_rate, momentum, device)
model.load_state_dict(torch.load(model_path))
model.eval()

prediction_list = []
for transform in dist_transforms:
    # Initial setup: using same dataset with transforms for prototyping
    partition = pickle.load(open(os.path.join(data_dir, 'partition_dict.p'), 'rb'))
    labels = pickle.load(open(os.path.join(data_dir, 'label_list.p'), 'rb'))

    unlabelled_dataset = DomainData(partition['train'], labels, data_dir, transform=transform)
    unlabelled_dataset_generator = DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False)

    predictions = np.array([]).reshape(0, 3)
    with torch.no_grad():
        for batch_idx, batch_info in enumerate(unlabelled_dataset_generator):
            batch_data, batch_labels = batch_info[0].to(device), batch_info[1].to(device)
            output = model(batch_data).cpu().numpy()
            predictions = np.concatenate((predictions, output))

    prediction_list.append(predictions)

with open('predictions.pkl', 'wb') as f:
    pickle.dump(prediction_list, f)

# Ensembling task: right now, just taking the max of the sum from each model, for each data point
ensembled_labels = []
for i in range(len(prediction_list[0])):
    ensembled_labels.append(np.argmax(prediction_list[0][i] + prediction_list[1][i] + prediction_list[2][i]))

pickle.dump(ensembled_labels, open('ensembled_labels.pkl', 'wb'))
