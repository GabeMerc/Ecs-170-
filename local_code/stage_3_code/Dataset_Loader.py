'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None  # 'ORL', 'MNIST', or 'CIFAR'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _to_arrays(self, instances):
        #Shape notation used below
        #  N = number of images in this split
        #  C = number of color channels
        #  H = image height in pixels (rows), W = image width in pixels (columns)
        #Conv2d expects X as (N, C, H, W)

        #Stack every image into one array
        X = np.array([inst['image'] for inst in instances])
        #One integer label per image
        y = np.array([inst['label'] for inst in instances], dtype=np.int64)

        if self.dataset_source_file_name == 'MNIST':
            #No channel dim in data insert one, we need to add it in
            #(num_images, 28 height, 28 width) -> (num_images, 1 channel, 28 height, 28 width)
            X = X[:, np.newaxis, :, :]
        elif self.dataset_source_file_name == 'ORL':
            # R=G=B so all are qual, so keep only channel 0, which drops the channel dim
            #(num_images, 112 height, 92 width, 3 channels) -> (num_images, 112 height, 92 width)
            #Then insert our own
            #(num_images, 112 height, 92 width) -> (num_images, 1 channel, 112 height, 92 width)
            X = X[:, :, :, 0][:, np.newaxis, :, :]
            #Labels are 1-40, shift to 0-39 for CrossEntropyLoss
            y = y - 1
        elif self.dataset_source_file_name == 'CIFAR':
            #Move channels from last to second position
            #(num_images, 32 height, 32 width, 3 channels) -> (num_images, 3 channels, 32 height, 32 width)
            X = X.transpose(0, 3, 1, 2)

        #Pixel values are 0-255
        X = X.astype(np.float32)
        return {'X': X, 'y': y}

    def load(self):
        print('loading data...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            raw = pickle.load(f)
        return {'train': self._to_arrays(raw['train']), 'test': self._to_arrays(raw['test'])}
