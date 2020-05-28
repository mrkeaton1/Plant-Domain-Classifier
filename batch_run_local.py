import os
from train_test import train_test

new_dir = input("Input new folder name\n>>> ")
new_path = os.path.join('Results', new_dir)
os.mkdir(new_path)
os.chdir(new_path)

for pt in (True, False):
    for lr in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75):
        for mom in (0.5, 0.7, 0.9, 0.95):

            train_test("/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)",
                       "resnet-18", pt, 128, 128, 1, lr, mom)
            train_test("/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)",
                       "resnet-34", pt, 64, 64, 1, lr, mom)
            # train_test("/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)",
            #            "efficientnet-b0", pt, 32, 32, 10, lr, mom)
            # train_test("/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)",
            #            "efficientnet-b7", pt, 4, 4, 10, lr, mom)
            train_test("/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)",
                       "fbnet_a", pt, 64, 64, 10, lr, mom)
            train_test("/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)",
                       "fbnet_c", pt, 64, 64, 10, lr, mom)
