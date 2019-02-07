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
import pandas>

from .bases import BaseImageDataset


class VeRi776(BaseImageDataset):
    """
    VeRi-776

    Reference:
    Liu X., Liu W., Ma H., Fu H.: Large-scale vehicle re-identification in urban surveillance videos. 
    In: IEEE International Conference on Multimedia and Expo. (2016)

    Dataset statistics:
    # identities: 776
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'veri/raw/VeRi'

    def __init__(self, root='data',
                 verbose=True, keypoints_dir=None, **kwargs):
        super(VeRi776, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        if keypoints_dir:
            keypoints_train=keypoints_dir+'keypoint_train.txt'
            keypoints_query=keypoints_dir+'keypoint_test.txt'
            keypoints_gallery=keypoints_dir+'keypoint_test.txt'
        else:
            keypoints_train=None
            keypoints_query=None
            keypoints_gallery=None

        train = self._process_dir(self.train_dir, relabel=True, keypoints=keypoints_train)
        query = self._process_dir(self.query_dir, relabel=False, keypoints=keypoints_query)
        gallery = self._process_dir(self.gallery_dir, relabel=False, keypoints=keypoints_gallery)

        if verbose:
            print("=> VeRi776 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        if keypoints_dir:
            self.num_train_orients, self.num_train_landmarks = self.get_imagelandmark_info(self.train)
            self.num_query_orients, self.num_query_landmarks = self.get_imagelandmark_info(self.query)
            self.num_gallery_orients, self.num_gallery_landmarks = self.get_imagelandmark_info(self.gallery)
        
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


    def _process_dir(self, dir_path, relabel=False, keypoints=None):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d\d\d)')

        if keypoints:
            # If keypoints file given then process this to get landmarks and orients
            keypoints_f = pandas.read_csv(keypoints, header=None, sep=' ')
            kp_image_pat = re.compile('\/([\d]+_c\d\d\d_.*\.jpg)')
            
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

            if keypoints:
                img_name = kp_image_pat.match(img_path).group(1)
                row = keypoints_f[keypoints_f[0] == 'VeRi/image_train/'+img_name]
                orient = row[41]
                landmarks = [0]*20
                for i in range(0:20):
                    if row[2*i+1] > -1:
                        landmarks[i] = 1
                dataset.append((img_path, pid, camid, orient, landmarks))
            else:
                dataset.append((img_path, pid, camid))

        
        return dataset

    def _get_imagelandmark_info(self, dataset):
        uniq_orients = set([o for (_, _, _, o, _) in dataset])
        num_orients = len(uniq_orients)
        num_landmarks = len(dataset[0][4])

        return num_orients, num_landmarks
