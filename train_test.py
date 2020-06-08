"""
Trains resnet-18 model on iNaturalist dataset domains

Created by Matthew Keaton on 4/14/2020
"""

from loaddataset import DomainData, categories
from utils import elapsed_time, create_confusion_matrix
from preprocessing import base_transform
from model_init import init_model
import pickle
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from time import time
import matplotlib.pyplot as plt


def train_test(data_dir, modelname, pretrained, train_batch_size, test_batch_size, n_epochs, learning_rate, momentum):

    print('Starting experiment.\n')
    pt = 'pretrained' if pretrained else 'untrained'
    partition = pickle.load(open(os.path.join(data_dir, 'partition_dict.p'), 'rb'))
    labels = pickle.load(open(os.path.join(data_dir, 'label_list.p'), 'rb'))

    print('Creating Domain Train Dataset')
    training_dataset = DomainData(partition['train'], labels, data_dir, transform=base_transform)
    print('Creating training generator')
    training_generator = DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True)
    print('Creating Domain Test Dataset')
    test_dataset = DomainData(partition['test'], labels, data_dir, transform=base_transform)
    print('Creating testing generator')
    test_generator = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    # Commented out because of massive slowdown
    # train_dom_count = np.zeros(3, dtype='int')
    # for i in range(len(training_dataset)):
    #     train_dom_count[training_dataset[i][1]] += 1
    # test_dom_count = np.zeros(3, dtype='int')
    # for i in range(len(test_dataset)):
    #     test_dom_count[test_dataset[i][1]] += 1
    # print('Number of leaves/branches/trees in training set: {}/{}/{}'
    #       .format(train_dom_count[0], train_dom_count[1], train_dom_count[2]))
    # print('Number of leaves/branches/trees in testing set: {}/{}/{}'
    #       .format(test_dom_count[0], test_dom_count[1], test_dom_count[2]))
    # print('Directory: {}'.format(data_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Model Selection and Initialization
    model = init_model(modelname, pretrained, n_epochs, learning_rate, momentum, device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_counter = []
    train_losses = []
    train_accs = []
    test_counter = [i * len(training_dataset) for i in range(1, n_epochs + 1)]
    test_losses = []
    test_accs = []
    start_time = time()
    train_misses = {}
    test_misses = {}

    new_dir = '{}_{}_epochs={}_lr={}_mom={}_batchsize={}-{}'.format(modelname, pt, n_epochs, learning_rate, momentum,
                                                                    train_batch_size, test_batch_size)
    os.mkdir(new_dir)
    os.chdir(new_dir)
    cm_basic = 'Confusion Matrices - Non-Normalized'
    cm_all = 'Confusion Matrices - Normalized on All'
    cm_true = 'Confusion Matrices - Normalized on True Values'
    cm_pred = 'Confusion Matrices - Normalized on Predictions'
    os.mkdir(cm_basic)
    os.mkdir(cm_all)
    os.mkdir(cm_true)
    os.mkdir(cm_pred)

    for e in range(1, n_epochs + 1):
        print('Epoch {}/{}'.format(e, n_epochs))
        print('Training...')
        model.train()
        start_train = time()
        train_corrects = 0
        for batch_idx, batch_info in enumerate(training_generator):

            batch_data, batch_labels = batch_info[0].to(device), batch_info[1].to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            train_predictions = torch.argmax(output, 1)
            loss = loss_fn(output, batch_labels)
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
            for i in range(len(train_predictions)):
                if train_predictions[i].item() != batch_labels[i].item():
                    im_id = training_dataset.get_sample_jpg_id(batch_idx * train_batch_size + i)
                    if train_misses.get(im_id, None) is None:
                        train_misses[im_id] = [e]
                    else:
                        train_misses[im_id].append(e)
        train_accuracy = float(train_corrects / len(training_dataset) * 100)
        train_accs.append(train_accuracy)
        print('Accuracy: {}/{} ({:.2f}%)   Time elapsed: {}'
              .format(train_corrects, current_epoch_tc, running_train_accuracy,
                      (elapsed_time(time() - start_train))))

        print('Testing...')
        test_avg_loss = 0.0
        test_corrects = 0
        test_labels = []  # Used for confusion matrix
        test_predictions = []  # Used for confusion matrix
        start_test = time()
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_info in enumerate(test_generator):
                batch_data, batch_labels = batch_info[0].to(device), batch_info[1].to(device)
                output = model(batch_data)
                batch_predictions = torch.argmax(output, 1)
                loss = loss_fn(output, batch_labels)
                test_avg_loss += (loss.item() * len(batch_predictions) / len(test_dataset))
                for i in range(len(batch_predictions)):
                    test_corrects += (batch_predictions[i].item() == batch_labels[i].item())
                for i in range(len(batch_predictions)):
                    if batch_predictions[i].item() != batch_labels[i].item():
                        im_id = test_dataset.get_sample_jpg_id(batch_idx * test_batch_size + i)
                        if test_misses.get(im_id, None) is None:
                            test_misses[im_id] = [e]
                        else:
                            test_misses[im_id].append(e)
                test_labels += batch_labels.tolist()
                test_predictions += batch_predictions.tolist()

            cmfig1 = plt.figure()
            plt.title('Confusion Matrix of Predicted vs. Ground Truth Labels')
            pivot1, heatmap1 = create_confusion_matrix(test_labels, test_predictions, categories)
            cmfig1.savefig(os.path.join(cm_basic, 'CM_Epoch_{}_basic.png'.format(e)))
            cmfig2 = plt.figure()
            plt.title('Normalized Confusion Matrix')
            pivot2, heatmap2 = create_confusion_matrix(test_labels, test_predictions, categories, normalize='all')
            cmfig2.savefig(os.path.join(cm_all, 'CM_Epoch_{}_normalized_all.png'.format(e)))
            cmfig3 = plt.figure()
            plt.title('Confusion Matrix Normalized Across True Labels')
            pivot3, heatmap3 = create_confusion_matrix(test_labels, test_predictions, categories, normalize='True')
            cmfig3.savefig(os.path.join(cm_true, 'CM_Epoch_{}_normalized_true.png'.format(e)))
            cmfig4 = plt.figure()
            plt.title('Confusion Matrix Normalized Across Predicted Labels')
            pivot4, heatmap4 = create_confusion_matrix(test_labels, test_predictions, categories, normalize='Pred')
            cmfig4.savefig(os.path.join(cm_pred, 'CM_Epoch_{}_normalized_prediction.png'.format(e)))
            plt.close('all')


            test_losses.append(test_avg_loss)
            test_accuracy = float(test_corrects / len(test_dataset) * 100)
            test_accs.append(test_accuracy)
            print('Avg. loss: {:.3f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_avg_loss, test_corrects, len(test_dataset), test_accuracy))
            print('Test time: {}'.format(elapsed_time(time() - start_test)))
            print('Train/test time: {}'.format(elapsed_time(time() - start_train)))
        print('Total overall time: {}\n'.format(elapsed_time(time() - start_time)))

    top_acc = max(test_accs)
    top_ind = test_accs.index(top_acc) + 1
    with open("../top_accuracies.txt", "a") as t:
        t.write('{}_{}_epochs={}_lr={}_mom={}_batchsize={}-{} - Top Accuracy: {}; Epoch Number: {}\n'
                .format(modelname, pt, n_epochs, learning_rate, momentum,
                        train_batch_size, test_batch_size, top_acc, top_ind))

    fig1 = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.title('Training and Testing Losses')
    plt.xlabel('Number of training examples seen by model')
    plt.ylabel('Cross entropy loss')
    fig1.savefig('Loss.png')
    fig1.savefig('../Losses/{}_{}_epochs={}_lr={}_mom={}_batchsize={}-{}.png'
                 .format(modelname, pt, n_epochs, learning_rate, momentum, train_batch_size, test_batch_size))

    fig2 = plt.figure()  # Code assumes n_epochs > 1
    plt.plot(range(1, n_epochs+1), train_accs, color='blue')
    plt.plot(range(1, n_epochs+1), test_accs, color='red')
    plt.xlim(0.75, n_epochs+0.25)
    plt.ylim(0, 100)
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
    plt.title('Accuracy Across Each Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    fig2.savefig('Accuracy.png')
    fig2.savefig('../Accuracies/{}_{}_epochs={}_lr={}_mom={}_batchsize={}-{}.png'
                 .format(modelname, pt, n_epochs, learning_rate, momentum, train_batch_size, test_batch_size))

    torch.save(model.state_dict(), '{}_model.pt'.format(modelname))

    os.chdir('..')

    # Used in python console for analysis of missed predictions in final epoch
    miss_5 = []
    for key in test_misses.keys():
        temp = set(test_misses[key])
        if n_epochs in temp:
            miss_5.append(key)
    miss_5.sort()
