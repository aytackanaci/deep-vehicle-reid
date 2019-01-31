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


class Aic19Track2(BaseImageDataset):
    """
    AIC 2019 Track2 (Re-ID)

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 666
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'aic19-track2-reid'

    def __init__(self, root='data', verbose=True, aic19_manual_labels=False, val=None, **kwargs):

        super(Aic19Track2, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self.train_label_csv = osp.join(self.dataset_dir, 'train_label_xml.csv')

        self.train_tracks_csv = osp.join(self.dataset_dir,  'train_track_id.txt')
        self.gallery_tracks_csv = osp.join(self.dataset_dir, 'test_track_id.txt')
        self.gallery_manual_id_csv = osp.join(self.dataset_dir, 'test_track_manual_ids.txt')
        self.query_manual_csv = osp.join(self.dataset_dir, 'name_query_manual.txt')

        self._check_before_run()

        train = self._process_label_csv(self.train_label_csv, relabel=True)
        if aic19_manual_labels:
            query, gallery = self._process_manual_labels(
                    self.gallery_dir,
                    self.query_dir,
                    self.gallery_tracks_csv,
                    self.gallery_manual_id_csv,
                    self.query_manual_csv)

        else:
            query = self._process_dir(self.query_dir, 0, relabel=False)
            gallery = self._process_dir(self.gallery_dir, 1, relabel=False)


        self.train = train
        self.query = query
        self.gallery = gallery

        if 0.0 < val < 1.0:
            # self.train_tracks = _read_csv(self.train_tracks_csv)
            self._split_train(val=val)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        if verbose:
            print("=> Aic19_track2 loaded")
            self.print_dataset_statistics(train, query, gallery)

    def _read_csv(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
            lines = [l.strip().split(' ') for l in lines]

        return lines

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
                for im_path in paths:
                    if camid is not q_cam:
                        gallery.append( (im_path, idx+num_val, camid))

        self.train = train
        self.query = query
        self.gallery = gallery

    def _process_manual_labels(self, gallery_dir, query_dir, tracks_path, gallery_csv_path, query_csv_path):

        tracks = self._read_csv(tracks_path)
        gallery_ids = self._read_csv(gallery_csv_path)
        query_ids = self._read_csv(query_csv_path)

        gallery = self._process_dir(gallery_dir, 1)
        num_gallery_images = len(gallery)

        for (track_id, man_idx, _, _, _) in gallery_ids:
            track = tracks[int(track_id)-1]
            for im in track:
                gallery[int(im)-1] = (
                        gallery[int(im)-1][0],
                        num_gallery_images + int(man_idx),
                        gallery[int(im)-1][2])

        labelled_query = filter(lambda x: len(x) == 2, query_ids)

        # sort query images by ID
        labelled_query.sort(key=lambda x: x[1])

        query = [(osp.join(self.query_dir, query), num_gallery_images + int(man_idx), 0) for query, man_idx in labelled_query]

        return query, gallery


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
