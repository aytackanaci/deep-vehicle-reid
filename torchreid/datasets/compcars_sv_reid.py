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
import pdb
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave

from .bases import BaseImageDataset


class CompcarsSurvReid(BaseImageDataset):
    ''' CompCars sv_data each model_color combination as ID
    '''

    dataset_dir = 'compcars'

    def __init__(self, root='data', verbose=True,  **kwargs):

        super(CompcarsSurvReid, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.images_dir = osp.join(self.dataset_dir, 'sv_data/image')

        self.train_color_txt = osp.join(self.dataset_dir, 'sv_data/train_color.txt')

        self._check_before_run()

        self.train = self._process_train_file_list(self.train_color_txt)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        if verbose:
            print("=> CompcarsSurvReid loaded, can onyl be used as train source")
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
        train = []

        id_counter = -1

        models = [[[] for _ in range(10)] for _ in range(281)]

        for item in file_list:
            model_id  = int(item[0].split('/')[0]) -1
            color = int(item[1])
            if color < 0: pass

            models[model_id][color].append(item[0])

        for idx, model in enumerate(models):
            for idx2, items in enumerate(model):
                if items:
                    id_counter += 1
                    train.extend( [(osp.join(self.images_dir, item), id_counter, -1 ) for item in items])

        return train
