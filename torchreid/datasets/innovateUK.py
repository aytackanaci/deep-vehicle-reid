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
import pandas

from .bases import BaseImageDataset


class innovateUK(BaseImageDataset):

    dataset_dir = 'innovateUK'

    def __init__(self, root='data',
                 verbose=True,
                 **kwargs):
        super(innovateUK, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.gallery_dir = self.dataset_dir

        self._check_before_run()

        gallery = self._process_dir(self.gallery_dir)

        self.gallery = gallery
        # Set dummy queries
        self.query = gallery

        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        if verbose:
            print("=> innovateUK loaded. Only loaded for test feature extraction.")
            self.print_dataset_statistics(gallery, gallery, gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        # if not osp.exists(self.dataset_dir):
        #     raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))


    def _process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*/*/*.jpg'))
        pattern = re.compile(r'(\d\d)/\d_(\d+)/\d+_\d+.jpg')

        dataset = []
        for img_path in img_paths:
            camid, pid = map(int, pattern.search(img_path).groups())
            dataset.append((img_path, pid, camid))

        return dataset
