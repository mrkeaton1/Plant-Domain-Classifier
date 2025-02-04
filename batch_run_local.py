import os
from train_test import train_test

new_dir = input("Input new folder name\n>>> ")
new_path = os.path.join('Results', new_dir)
os.mkdir(new_path)
os.chdir(new_path)
os.mkdir('Accuracies')
os.mkdir('Losses')

for lr in (0.01, 0.025, 0.05):
    for mom in (0.1, 0.25, 0.5):

        train_test("/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)",
                   "resnet-18", True, 128, 128, 10, lr, mom)
        train_test("/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)",
                   "resnet-34", True, 64, 64, 10, lr, mom)
        train_test("/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)",
                   "fbnet_a", True, 64, 64, 10, lr, mom)
        train_test("/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)",
                   "fbnet_c", True, 64, 64, 10, lr, mom)
