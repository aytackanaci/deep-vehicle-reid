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


class CompCars(BaseClassificationDataset):
    """
    VeRi-776

    """
    dataset_dir = 'compcars'

    def __init__(self, root='data',
            verbose=True, **kwargs):
        super(CompCars, self).__init__(root)

        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')

        self.xml_path = osp.join(self.dataset_dir, 'train_label.xml')

        self.train = self._process_xml(self.xml_path, 'type')
        self._split_train()

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)

        if verbose:
            print("=> CompCars loaded")
            self.print_dataset_statistics(self.train, self.test)

    def _process_xml(self, xml_path, label, gather_classes=False):
        from lxml import etree
        tree = etree.parse(xml_path)
        root = tree.getroot()
        items = root.getchildren()[0]

        if label == 'type':
            def select_classes(c):
                type_dict = {1:1, 2:2, 3:3, 4:4, 5:3, 6:5, 7:6, 8:7, 9:1}
                return type_dict[c]
            if gather_classes:
                train = [ (osp.join(self.train_dir, item.attrib['imageName']), select_classes(int(item.attrib['typeID'])) ) for item in items]
            else:

                train = [ (osp.join(self.train_dir, item.attrib['imageName']), int(item.attrib['typeID']) ) for item in items]

        return train
