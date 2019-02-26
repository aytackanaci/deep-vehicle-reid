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
import numpy as np
import h5py
from scipy.misc import imsave

from .bases import BaseImageDataset


class VehicleID(BaseImageDataset):
    ''' PKU VehicleID

    Liu H, Tian Y, Yang Y, et al.
    Deep relative distance learning: Tell the difference between similar vehicles
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2167-2175.
    '''

    dataset_dir = 'veid/raw/VehicleID_V1.0/'

    def __init__(self, root='data', verbose=True, vehicleid_test_size='large', **kwargs):

        super(VehicleID, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.images_dir = osp.join(self.dataset_dir, 'images')

        self.train_label_csv = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')

        sizes = {'small': 'train_test_split/test_list_800.txt',
                'medium': 'train_test_split/test_list_1600.txt',
                'large': 'train_test_split/test_list_2400.txt'}

        if not vehicleid_test_size in sizes.keys():
            raise RuntimeError("'--vehicleid-test-size' must be one of {}, given {}.".format(sizes, vehicleid_test_size))
        else:
            self.test_label_csv = osp.join(self.dataset_dir, sizes[vehicleid_test_size])

        self._check_before_run()

        self.train = self._process_label_csv(self.train_label_csv, relabel=True)
        test = self._process_label_csv(self.test_label_csv, relabel=False)
        self.query, self.gallery = self._split_test(test)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        if verbose:
            print("=> Aic19_track2 loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

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

    def _split_test(self, test):
        from random import randint, seed
        seed(1337)

        test_dict = defaultdict(list)
        for i, item in enumerate(test):
            test_dict[item[1]].append(i)

        query_imlist = []
        gallery_imlist = []
        for ID, idxs in test_dict.items():
            r = randint(0, len(idxs)-1)
            g = test[idxs[r]]
            gallery_imlist.append((g[0], g[1], 1)) # camID 1 for gallery images
            idxs.pop(r)
            qs = [(test[i][0], test[i][1], 0) for i in idxs] # camID 0 for query images
            query_imlist.extend(qs)

        return query_imlist, gallery_imlist

    def _split_train(self, val=0.3):
        print('Splitting Train')
        import random
        train_pids = set()
        num_pids, num_imgs, num_cams = self.get_imagedata_info(self.train)
        identities = [[[] for _ in range(num_cams+6)] for _ in range(num_pids)]

        for im_path, pid, camid in self.train:
            identities[pid][camid].append(im_path)
            train_pids.add(pid)

        trainval_pids = list(train_pids)
        random.seed(1337)
        random.shuffle(trainval_pids)
        num_val = int(round(num_pids * val))

        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])

        train = []
        for idx, pid in enumerate(train_pids):
            for camid, paths in enumerate(identities[pid]):
                for im_path in paths:
                    train.append( (im_path, idx, camid) )


        query, gallery = [], []
        for idx, pid in enumerate(val_pids):

            q_cam = None
            cams = []
            for camid, paths in enumerate(identities[pid]):
                # print(pid, camid, paths)
                if len(paths) != 0:
                    cams.append(camid)

            q_cam = random.choice(cams)

            for path in identities[pid][q_cam]:
                query.append( (path, idx+num_val, camid))

            for camid, paths in enumerate(identities[pid]):
                if camid is not q_cam:
                    for im_path in paths:
                        gallery.append( (im_path, idx+num_val, camid))

        self.train = train
        self.query = query
        self.gallery = gallery


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
                    impath, imlabel = line.strip().split()
                    impath = osp.join(self.images_dir, impath)
                    imlist.append( (impath+'.jpg', int(imlabel), -1 ) )

            return imlist

        imlist = pluck(fpath)
        train_classes, classes2idx = getClasses(imlist)
        if relabel:
            imlist = [(fpath, classes2idx[pid], cid) for (fpath, pid, cid) in imlist]

        return imlist
