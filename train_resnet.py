"""
Trains resnet-18 model on iNaturalist dataset domains
Created by Matthew Keaton on 4/14/2020
To Do:
1) make train () return all needed values, rather than passing in as params
2) train accuracy?
"""

from loaddataset import get_labelspace_size, randomize, split, load_data, DomainData
from preprocessing import base_transform
from train_test import train
import sys
import torch
from torchvision.models.resnet import resnet18
import matplotlib.pyplot as plt

train_batch_size = 128
test_batch_size = 256
train_percent = 0.7
learning_rate = 0.01
momentum = 0.9
n_epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_init, labels_init = load_data('/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset')
data_init, labels_init = randomize(data_init, labels_init)
train_data_list, train_labels_list, test_data_list, test_labels_list = split(data_init, labels_init, train_percent)
train_dataset = DomainData(train_data_list, train_labels_list, transform=base_transform)
test_dataset = DomainData(test_data_list, test_labels_list, transform=base_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)
test_counter = [i * train_dataset.size for i in range(1, n_epochs + 1)]
test_losses = []

if sys.argv[1] == 'True':
    print('Beginning with pretrained resnet-18 architecture.')
    resnet18_base = resnet18(pretrained=True)
else:
    print('Beginning with untrained resnet-18 architecture.')
    resnet18_base = resnet18(pretrained=False)

resnet18_base.fc = torch.nn.Linear(512, get_labelspace_size())  # May or may not be best method
resnet18_base.to(device)
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18_base.parameters(), lr=learning_rate, momentum=momentum)

train_losses, train_counter, test_losses, test_accuracies = train(resnet18_base, train_dataset, train_batch_size,
                                                                  test_dataset, test_batch_size, optimizer, n_epochs, ce_loss, device)

fig1 = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('Number of training examples seen')
plt.ylabel('Cross entropy loss')
fig1.show()
if sys.argv[1] == 'True':
    fig1.savefig('/home/mrkeaton/Documents/Plant_Code/Domain Classifier/init_loss_results_pretrained.png')
else:
    fig1.savefig('/home/mrkeaton/Documents/Plant_Code/Domain Classifier/init_loss_results_untrained.png')

fig2 = plt.figure()
plt.plot(range(1, n_epochs+1), test_accuracies, color='red')
plt.title('Accuracy across each epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
fig2.show()
if sys.argv[1] == 'True':
    fig2.savefig('/home/mrkeaton/Documents/Plant_Code/Domain Classifier/init_accuracy_results_pretrained.png')
else:
    fig2.savefig('/home/mrkeaton/Documents/Plant_Code/Domain Classifier/init_accuracy_results_untrained.png')
