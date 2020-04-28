"""
Trains resnet-18 model on iNaturalist dataset domains
Created by Matthew Keaton on 4/14/2020
"""

from loaddataset import get_labelspace_size, DomainData
from utils import elapsed_time
from preprocessing import base_transform
import pickle
import sys
import os
from math import ceil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from time import time
import matplotlib.pyplot as plt

# train_percent = 0.8

data_dir = sys.argv[1]
pretrained = bool(sys.argv[2])
train_batch_size = int(sys.argv[3])
test_batch_size = int(sys.argv[4])
n_epochs = int(sys.argv[5])
learning_rate = float(sys.argv[6])
momentum = float(sys.argv[7])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ?
# cudnn.benchmark = True

partition = pickle.load(open(os.path.join(data_dir, 'partition_dict.p'), 'rb'))
labels = pickle.load(open(os.path.join(data_dir, 'label_list.p'), 'rb'))

training_dataset = DomainData(partition['train'], labels, data_dir, transform=base_transform)
training_generator = DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True)
test_dataset = DomainData(partition['test'], labels, data_dir, transform=base_transform)
test_generator = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

train_dom_count = np.zeros(3, dtype='int')
for i in range(len(training_dataset)):
    train_dom_count[training_dataset[i][1]] += 1
test_dom_count = np.zeros(3, dtype='int')
for i in range(len(test_dataset)):
    test_dom_count[test_dataset[i][1]] += 1
print('Number of leaves/branches/trees in training set: {}/{}/{}'
      .format(train_dom_count[0], train_dom_count[1], train_dom_count[2]))
print('Number of leaves/branches/trees in testing set: {}/{}/{}'
      .format(test_dom_count[0], test_dom_count[1], test_dom_count[2]))
print('Directory: {}'.format(data_dir))

if pretrained:
    print('Beginning with pretrained resnet-18 architecture. Epochs = {}; Learning rate = {}; momentum = {}\n'
          .format(n_epochs, learning_rate, momentum))
    resnet18_base = resnet18(pretrained=True)
else:
    print('Beginning with untrained resnet-18 architecture. Epochs = {}; Learning rate = {}; momentum = {}\n'
          .format(n_epochs, learning_rate, momentum))
    resnet18_base = resnet18(pretrained=False)

resnet18_base.fc = torch.nn.Linear(512, get_labelspace_size())
resnet18_base = torch.nn.DataParallel(resnet18_base)
resnet18_base.to(device)
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18_base.parameters(), lr=learning_rate, momentum=momentum)

train_counter = []
train_losses = []
train_accs = []
test_counter = [i * len(training_dataset) for i in range(1, n_epochs + 1)]
test_losses = []
test_accs = []
start_time = time()

for e in range(1, n_epochs + 1):

    print('\nTraining...')
    resnet18_base.train()
    start_train = time()
    train_corrects = 0
    for batch_idx, batch_info in enumerate(training_generator):
        batch_data, batch_labels = batch_info[0].to(device), batch_info[1].to(device)
        optimizer.zero_grad()
        output = resnet18_base(batch_data)
        train_predictions = torch.argmax(output, 1)
        loss = ce_loss(output, batch_labels)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if len(train_counter) == 0:
            train_counter.append(len(train_predictions))
        else:
            train_counter.append(train_counter[-1] + len(train_predictions))
        for i in range(len(train_predictions)):
            train_corrects += (train_predictions[i].item() == batch_labels[i].item())
        current_epoch_tc = train_counter[-1]-((e-1)*len(training_dataset))
        running_train_accuracy = float(train_corrects / current_epoch_tc * 100)
        print('Train Epoch: {}/{}; Batch: {}/{} Accuracy: {}/{} ({:.2f}%)   Time elapsed: {}'
              .format(e, n_epochs, batch_idx + 1, ceil(len(training_dataset) / train_batch_size),
                      train_corrects, current_epoch_tc, running_train_accuracy,
                      (elapsed_time(time() - start_train))))
    train_accuracy = float(train_corrects / len(training_dataset) * 100)
    train_accs.append(train_accuracy)
    print('\nTotal overall time: {}'.format(elapsed_time(time() - start_time)))


    print('\nTesting...')
    test_avg_loss = 0.0
    test_corrects = 0
    start_test = time()
    resnet18_base.eval()
    with torch.no_grad():
        for batch_idx, batch_info in enumerate(test_generator):
            batch_data, batch_labels = batch_info[0].to(device), batch_info[1].to(device)
            output = resnet18_base(batch_data)
            test_predictions = torch.argmax(output, 1)
            loss = ce_loss(output, batch_labels)
            test_avg_loss += (loss.item() * len(test_predictions) / len(test_dataset))
            for i in range(len(test_predictions)):
                test_corrects += (test_predictions[i].item() == batch_labels[i].item())
            print('Epoch {} - Test Batch: {}/{}   Time elapsed: {}'
                  .format(e, batch_idx + 1, ceil(len(test_dataset) / test_batch_size),
                          (elapsed_time(time() - start_test, short=True))))
        test_losses.append(test_avg_loss)
        test_accuracy = float(test_corrects / len(test_dataset) * 100)
        test_accs.append(test_accuracy)
        print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_avg_loss, test_corrects, len(test_dataset), test_accuracy))
        print('Total test time: {}'.format(elapsed_time(time() - start_test)))
    print('\nTotal overall time: {}'.format(elapsed_time(time() - start_time)))

fig1 = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.title('Training and testing losses')
plt.xlabel('Number of training examples seen by model')
plt.ylabel('Cross entropy loss')
fig1.show()
pt = 'pretrained' if pretrained else 'untrained'
fig1.savefig('Results/init_results_{}_lr={}_mom={}_losses.png'
             .format(pt, learning_rate, momentum))

fig2 = plt.figure()
plt.plot(range(1, n_epochs+1), train_accs, color='blue')
plt.plot(range(1, n_epochs+1), test_accs, color='red')
plt.xlim(0.75, n_epochs+0.25)  # Assumes n_epochs > 1
plt.ylim(0, 100)
plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
plt.title('Accuracy across each epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
fig2.show()
fig2.savefig('Results/init_results_{}_lr={}_mom={}_accuracy.png'
             .format(pt, learning_rate, momentum))
