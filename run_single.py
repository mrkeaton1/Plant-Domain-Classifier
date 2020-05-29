import sys
import os
from train_test import train_test

data_dir = sys.argv[1]
modelname = sys.argv[2]
pretrained = bool(sys.argv[3])
train_batch_size = int(sys.argv[4])
test_batch_size = int(sys.argv[5])
n_epochs = int(sys.argv[6])
learning_rate = float(sys.argv[7])
momentum = float(sys.argv[8])

new_dir = input("Input new folder name\n>>> ")
new_path = os.path.join('Results', new_dir)
os.mkdir(new_path)
os.chdir(new_path)
os.mkdir('Accuracies')
os.mkdir('Losses')

train_test(data_dir, modelname, pretrained, train_batch_size, test_batch_size, n_epochs, learning_rate, momentum)