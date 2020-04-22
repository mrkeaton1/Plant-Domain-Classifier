"""
Created by Matthew Keaton on 4/16/2020
"""

from utils import elapsed_time
import time
import torch
from torchvision.transforms import ToTensor
from loaddataset import randomize

def train(t_model, trn_dataset, trn_batch_size, tst_dataset, tst_batch_size, optim, n_epochs, lss, dev):

    trn_losses = []
    trn_counter = []
    corrects = 0
    trn_accs = []
    tst_losses = []
    tst_accs = []
    start_train = time.time()

    for e in range(1, n_epochs + 1):
        t_model.train()

        for idx in range(trn_dataset.size):  # TTT "- 7000"
            optim.zero_grad()
            image, label = trn_dataset[idx]
            data = ToTensor()(image).unsqueeze(0).to(dev)
            label = label.to(dev)
            output = t_model(data)
            pred = torch.argmax(output)
            corrects += int(pred == label.item())
            loss = lss(output, label)
            loss.backward()
            optim.step()

            if idx % trn_batch_size == 0:  # NOTE: This is not actually handled as a batch - needs to be improved later
                print('Train Epoch: {}/{}\t[{} / {}]\tTime elapsed: {}'
                      .format(e, n_epochs, idx, trn_dataset.size,
                              (elapsed_time(time.time() - start_train))))
                trn_losses.append(loss.item())
                trn_counter.append(idx + (e - 1) * trn_dataset.size)

        trn_accs.append(corrects/trn_dataset.size)
        print('Accuracy: {}'.format(corrects/trn_dataset.size))
        tst_loss, tst_acc = test(t_model, tst_dataset, tst_batch_size, lss, dev)
        tst_losses.append(tst_loss)
        tst_accs.append(tst_acc)

    print('Total time: {}'.format(elapsed_time(time.time() - start_train)))
    return trn_counter, trn_losses, trn_accs, tst_losses, tst_accs


def test(t_model, tst_dataset, tst_batch_size, lss, dev):

    start_test = time.time()
    print('Testing...')
    t_model.eval()
    test_avg_loss = 0
    correct = 0

    with torch.no_grad():

        for idx in range(tst_dataset.size):  # TTT "- 2900"
            image, label = tst_dataset[idx]
            data = ToTensor()(image).unsqueeze(0).to(dev)
            label = label.to(dev)
            output = t_model(data)
            pred = torch.argmax(output)
            test_avg_loss += lss(output, label).item()
            correct += int(pred == label.item())

    test_avg_loss /= tst_dataset.size
    accuracy = 100. * correct / tst_dataset.size
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_avg_loss, correct, tst_dataset.size, accuracy))
    print('Total test time: {}'.format(elapsed_time(time.time() - start_test)))
    return test_avg_loss, accuracy
