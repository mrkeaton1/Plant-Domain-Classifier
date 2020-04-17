"""
Created by Matthew Keaton on 4/16/2020
"""

from utils import elapsed_time
import sys
import time
import torch
from torchvision.transforms import ToTensor


def train(t_model, trn_dataset, trn_losses, trn_counter, tst_dataset, tst_losses, optim, n_epochs, lss, t_batch_size):
    start_train = time.time()
    t_model.train()
    for e in range(1, n_epochs + 1):
        for idx in range(trn_dataset.size):
            optim.zero_grad()
            image, label = trn_dataset[idx]
            data = ToTensor()(image).unsqueeze(0)
            # Try catch block implemented to remove any files in dataset that end with '.jpg' but are of type '.png'
            try:
                output = t_model(data)
                loss = lss(output, label)
                loss.backward()
                optim.step()
                if idx % t_batch_size == 0:  # NOTE: This is not actually handed as a batch - needs to be improved later
                    sys.stdout.write("\033[F")  # back to previous line
                    sys.stdout.write("\033[K")  # clear line
                    print('Train Epoch: {}\t[{} / {}]\tTime elapsed: {}'
                          .format(e, idx, trn_dataset.size,
                                  (elapsed_time(time.time() - start_train))))  # should I use train_data here?
                    trn_losses.append(loss.item())
                    trn_counter.append(idx + (n_epochs-1) * trn_dataset.size)
            except:
                print()
        # Test here
        test(t_model, tst_dataset, tst_losses, lss)
    print('Total time: {:.3f}'.format(time.time() - start_train))


def test(t_model, t_dataset, tst_losses, lss):
    start_test = time.time()
    print('Testing...')
    t_model.eval()
    test_avg_loss = 0
    correct = 0
    with torch.no_grad():
        for idx in range(t_dataset.size):
            image, label = t_dataset[idx]
            data = ToTensor()(image).unsqueeze(0)
            output = t_model(data)
            test_avg_loss += lss(output, label)
            pred = torch.max(output)
            if idx % 100 == 0:
                print('[{} / {}]\t{}'
                      .format(idx, t_dataset.size,
                              (elapsed_time(time.time() - start_test, short=True))))  # should I use train_data here?
    test_avg_loss /= t_dataset.size
    tst_losses.append(test_avg_loss.item())
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_avg_loss, correct, t_dataset.size,
        100. * correct / t_dataset.size))
    print('Total time: {}'.format(elapsed_time(time.time() - start_test)))

# sys.stdout.write("\033[F")  # back to previous line
#     sys.stdout.write("\033[K")  # clear line
