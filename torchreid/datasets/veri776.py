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
    dataset_dir = 'VeRi'

    def __init__(self, root='data',
                 verbose=True, keypoints_dir=None,
                 regress_landmarks=False,
                 **kwargs):
        super(VeRi776, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.regress_landmarks = regress_landmarks
        
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

        self.train = train
        self.query = query
        self.gallery = gallery

        train_info = self.get_imagedata_info(self.train)
        query_info = self.get_imagedata_info(self.query)
        gallery_info = self.get_imagedata_info(self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = train_info[0:3]
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = query_info[0:3]
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = gallery_info[0:3]
        
        if len(train_info) > 3:
            self.num_train_orients, self.num_train_landmarks = train_info[3:5]
            self.num_query_orients, self.num_query_landmarks = query_info[3:5]
            self.num_gallery_orients, self.num_gallery_landmarks = gallery_info[3:5]

        if verbose:
            print("=> VeRi776 loaded")
            self.print_dataset_statistics(train, query, gallery)

    def get_imagedata_info(self, data):
        if len(data[0]) == 3:
            uniq_pids = set([i for (_, i, _) in data])
            uniq_cams = set([c for (_, _, c) in data])
        else:
            # Also have orientation and landmark info
            uniq_pids = set([i for (_, i, _, _, _) in data])
            uniq_cams = set([c for (_, _, c, _, _) in data])
            uniq_orients = set([o for (_, _, _, o, _) in data])
            num_orients = len(uniq_orients)
            num_landmarks = len(data[0][4])
        num_pids = len(uniq_pids)
        num_cams = len(uniq_cams)
        num_imgs = len(data)

        if len(data[0]) == 3:
            return num_pids, num_imgs, num_cams
        else:
            return num_pids, num_imgs, num_cams, num_orients, num_landmarks
    
    def print_dataset_statistics(self, train, query, gallery):
        if len(train[0]) == 3:
            return super(VeRi776, self).print_dataset_statistics(train, query, gallery)
        else:            
            print("Dataset statistics:")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # cameras | # orients | # landmarks")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:8d} | {:9d} | {:9d} | {:11d}".format(self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_orients, self.num_train_landmarks))
            print("  query    | {:5d} | {:8d} | {:9d} | {:9d} | {:11d}".format(self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_orients, self.num_query_landmarks))
            print("  gallery  | {:5d} | {:8d} | {:9d} | {:9d} | {:11d}".format(self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_orients, self.num_gallery_landmarks))
            print("  ----------------------------------------")

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
            kp_image_pat = re.compile('.*\/([\d]+_c\d\d\d_.*\.jpg)')
            
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
                if len(row) == 0:
                    orient = 0
                    landmarks = np.zeros(20)
                else:
                    row = row.iloc[0]
                    orient = row[41]

                    # TODO get this to work instead
                    # landmarks = np.array([min(0,int(x))+1 for x in row[list(range(1,len(row)-1,2))]]) 
                    if self.regress_landmarks:
                        # TODO these need to be normalised according to the image resizing
                        landmarks = np.array(row[1:41])
                    else:
                        landmarks = np.zeros(20)
                        for i in range(20):
                            if int(row[2*i+1]) > -1:
                                landmarks[i] = 1

                dataset.append((img_path, pid, camid, orient, landmarks))
            else:
                dataset.append((img_path, pid, camid))
        
        return dataset
