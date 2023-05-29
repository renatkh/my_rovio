from os import listdir
from os.path import isfile, join

import numpy as np
from torch.utils.data import Dataset

REVERSE_COMMANDS = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 9, 7: 8, 8: 7, 9: 6}


class ImageDataSet(Dataset):
    def __init__(self, data_files, transform=None):
        self.data_files = data_files
        self.im1 = None
        self.im2 = None
        self.vector = None
        for f in self.data_files:
            with np.load(f) as data:
                if self.im1 is not None:
                    self.im1 = np.concatenate(
                        (self.im1, np.moveaxis(data['im1'], -1, 0)))
                    self.im2 = np.concatenate(
                        (self.im2, np.moveaxis(data['im2'], -1, 0)))
                    self.vector = np.concatenate((self.vector, data['vector']))
                else:
                    self.im1 = np.moveaxis(data['im1'], -1, 0)
                    self.im2 = np.moveaxis(data['im2'], -1, 0)
                    self.vector = data['vector']
        self.transform = transform
        self.rc = np.random.RandomState(42)

    def __len__(self):
        return self.im1.shape[0]

    def __getitem__(self, idx):
        im1 = self.im1[idx]
        im2 = self.im2[idx]
        vector = self.vector[idx]
        vector[1] = vector[1] / 10.  # speed normalization
        vector[2] = vector[2] / 4.  # rovio units value normalization
        if self.rc.rand() < 0.5:
            img = np.moveaxis(np.array([im1, im2, im2-im1]), 0, -1)
        else:
            img = np.moveaxis(np.array([im2, im1, im1-im2]), 0, -1)
            vector[0] = REVERSE_COMMANDS[vector[0]]
        if self.transform:
            img = self.transform(img)
        return img, vector


def get_datasets(
        store_path='/Volumes/Documents/datasets/rovio/',
        train_fraction=0.8):
    data_files = [join(store_path, f)
                  for f in listdir(store_path) if isfile(join(store_path, f))]
    random_state = np.random.RandomState(42)
    train_set = random_state.choice(data_files, max(1, int(
        len(data_files) * train_fraction)), replace=False)
    val_set = [f for f in data_files if f not in train_set]
    return train_set, val_set
