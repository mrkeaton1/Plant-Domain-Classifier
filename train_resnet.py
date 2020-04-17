"""
Trains resnet-18 model on iNaturalist dataset domains
Created by Matthew Keaton on 4/14/2020
"""

from loaddataset import get_labelspace_size, randomize, split, load_data, DomainData
from preprocessing import base_transform
from train_test import train, test
import torch
from torchvision.models.resnet import resnet18
import matplotlib.pyplot as plt

train_batch_size = 128
test_batch_size = 32
train_percent = 0.7
learning_rate = 0.01
momentum = 0.5
n_epochs = 5
log_interval = 10

d, l = load_data('/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset')
d, l = randomize(d, l)
train_data_list, train_labels_list, test_data_list, test_labels_list = split(d, l, train_percent)
train_dataset = DomainData(train_data_list, train_labels_list, transform=base_transform)
test_dataset = DomainData(test_data_list, test_labels_list, transform=base_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)
train_losses = []
test_losses = []
train_counter = []
test_counter = [i*train_dataset.size for i in range(1, n_epochs+1)]

resnet18_base = resnet18(pretrained=True)
resnet18_base.fc = torch.nn.Linear(512, get_labelspace_size())  # May or may not be best method
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18_base.parameters(), lr=learning_rate, momentum=momentum)

train(resnet18_base, train_dataset, train_losses, train_counter, test_dataset, test_losses, optimizer, n_epochs, ce_loss, train_batch_size)

fig = plt.figure()
plt.plot(train_counter, train_losses, color = 'blue')
plt.scatter(test_counter, test_losses, color = 'red')
plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('cross entropy loss')
fig.show()
fig.savefig('/home/mrkeaton/Documents/Plant_Code/Domain Classifier/init_loss_results.png')
