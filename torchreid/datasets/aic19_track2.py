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

from .bases import BaseImageDataset


class Aic19Track2(BaseImageDataset):
    """
    VeRi-776

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 776
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'aic19-track2-reid'

    def __init__(self, root='data',
            verbose=True, **kwargs):
        super(Aic19Track2, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self.train_label_csv = osp.join(self.dataset_dir, 'train_label_xml.csv')

        self._check_before_run()

        train = self._process_label_csv(self.train_label_csv, relabel=True)
        query = self._process_dir(self.query_dir, 0, relabel=False)
        gallery = self._process_dir(self.gallery_dir, 1, relabel=False)

        if verbose:
            print("=> Aic19_track2 loaded")
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

    def _process_label_csv(self, fpath, relabel=False):
        def getClasses(imlist):
            classesList = [x[1] for x in imlist]
            classes = list(set(classesList))
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            return classes, class_to_idx

        def pluck(imlist_file):

            imlist = []
            with open(imlist_file, 'r') as rf:
                for line in rf.readlines():
                    impath, imlabel, cam_label = line.strip().split()
                    impath = osp.join(self.train_dir, impath)
                    imlist.append( (impath, int(imlabel), int(cam_label)) )

            return imlist

        imlist = pluck(fpath)
        train_classes, classes2idx = getClasses(imlist)
        if relabel:
            imlist = [(fpath, classes2idx[pid], cid) for (fpath, pid, cid) in imlist]

        return imlist

    def _process_dir(self, dir_path, camid, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d\d\d)')

        # pid_container = set()
        # for img_path in img_paths:
        #     pid, _ = map(int, pattern.search(img_path).groups())
        #     if pid == -1: continue  # junk images are just ignored
        #     pid_container.add(pid)
        # pid2label = {pid:label for label, pid in enumerate(pid_container)}

        # dataset = []
        # for img_path in img_paths:
        #     pid, camid = map(int, pattern.search(img_path).groups())
        #     if pid == -1: continue  # junk images are just ignored
        #     # assert 1 <= pid <= 776  # pid == 0 means background
        #     # assert 1 <= camid <= 20
        #     camid -= 1 # index starts from 0
        #     if relabel: pid = pid2label[pid]
        #     dataset.append((img_path, pid, camid))

        en = enumerate(sorted(img_paths))
        dataset = [(fpath, idx, camid) for (idx, fpath) in en]

        return dataset
