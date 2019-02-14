from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave

from .bases import BaseImageDataset, BaseClassificationDataset


class Cub200(BaseClassificationDataset):
    """
    CUB 200

    """
    dataset_dir = 'cub200/raw/CUB_200_2011'

    def __init__(self, root='data',
            verbose=True, **kwargs):
        super(Cub200, self).__init__(root)

        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.images_dir = osp.join(self.dataset_dir, 'images')

        self._images_txt = osp.join(self.dataset_dir, 'images.txt')
        self._split_txt = osp.join(self.dataset_dir, 'train_test_split.txt' )
        self._process_label_file()

        # self._split_train()

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

        if verbose:
            print("=> Cub200 loaded")
            self.print_dataset_statistics(self.train, self.test)

    def _process_label_file(self):

        # Format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(self._images_txt, dtype=str)
        # Format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(self._split_txt, dtype=int)

        train_data = []
        test_data = []

        for id_ in range(id2name.shape[0]):
            image_path = os.path.join(self.images_dir, id2name[id_, 1])
            label = int(id2name[id_, 1][:3]) - 1  # Label starts with 0

            entry = (image_path, label)

            if id2train[id_, 1] == 1:
                train_data.append(entry)
            else:
                test_data.append(entry)

        self.train = train_data
        self.test = test_data
