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
from collections import defaultdict
import os.path as osp
from scipy.io import loadmat
import pdb
import numpy as np
import h5py
from scipy.misc import imsave

from .bases import BaseImageDataset


class Boxcars21k(BaseImageDataset):
    ''' Boxcars21k
    '''

    dataset_dir = 'boxcars21k'

    def __init__(self, root='data', verbose=True, **kwargs):

        super(Boxcars21k, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.images_dir = osp.join(self.dataset_dir, 'images')

        self.train_label_csv = osp.join(self.dataset_dir, 'boxcars21k_train.txt')

        self._check_before_run()

        self.train = self._process_train_file_list(self.train_label_csv)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        if verbose:
            print("=> BoxCars21k loaded, Can only be used as train source")
            self.print_dataset_statistics(self.train, self.train, self.train)

    def _read_csv(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
            lines = [l.strip().split(' ') for l in lines]

        return lines

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        # if not osp.exists(self.dataset_dir):
        #     raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.images_dir):
            raise RuntimeError("'{}' is not available".format(self.images_dir))

    def _process_train_file_list(self, fpath):

        file_list = self._read_csv(fpath)
        file_list = [l[0] for l in file_list]
        train = []

        id_counter = -1
        for idx, path in enumerate(file_list):
            # there are 3 images per ID
            if idx % 3 is 0:
                id_counter += 1

            train.append( (osp.join(self.images_dir, path), id_counter, -1) )

        return train
