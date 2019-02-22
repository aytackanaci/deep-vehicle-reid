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


class VeRiType(BaseClassificationDataset):
    """
    VeRi-776

    """
    dataset_dir = 'veri/raw/VeRi'

    def __init__(self, root='data',
            verbose=True, **kwargs):
        super(VeRiType, self).__init__(root)

        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')

        self.xml_path = osp.join(self.dataset_dir, 'train_label.xml')

        self.train = self._process_xml(self.xml_path, 'type')
        self._split_train()

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)

        if verbose:
            print("=> VeRi776 loaded")
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

class VeRi776(BaseImageDataset):
    """
    VeRi-776

    """
    dataset_dir = 'veri/raw/VeRi'

    def __init__(self, root='data',
            verbose=True, **kwargs):
        super(VeRi776, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.xml_path = osp.join(self.dataset_dir, '')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> VeRi776 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        # if not osp.exists(self.dataset_dir):
        #     raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d\d\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
