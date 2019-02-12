from __future__ import absolute_import
from __future__ import print_function

import os.path as osp
import numpy as np

class BaseClassificationDataset(object):

    def __init__(self, root):
        self.root = osp.expanduser(root)

    def get_imagedata_info(self, data):
        pids = []
        for _, pid in data:
            pids += [pid]
        pids = set(pids)
        num_pids = len(pids)
        num_imgs = len(data)

        return num_pids, num_imgs

    def _split_train(self, split=0.9):
        assert self.train is not None, 'Error: train images are not initialized'
        print('Splitting Train')
        import random
        random.seed(1337)
        train_pids = set()
        num_pids, _ = self.get_imagedata_info(self.train)
        classes = [[] for _ in range(num_pids)]

        for im_path, pid in self.train:
            classes[pid].append(im_path)
            train_pids.add(pid)

        train, val = [], []
        for pid in train_pids:
            random.shuffle(classes[pid])

            len_images = len(classes[pid])
            len_train = len_images - int(len_images*split)
            imlist = [(item, pid) for item in classes[pid]]

            train.extend(imlist[:len_train])
            val.extend(imlist[len_train:])

        self.train = train
        self.test = val

    def print_dataset_statistics(self, train, test):
        num_train_pids, num_train_imgs= self.get_imagedata_info(train)
        num_test_pids, num_test_imgs= self.get_imagedata_info(test)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images ")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} ".format(num_train_pids, num_train_imgs))
        print("  test     | {:5d} | {:8d} ".format(num_test_pids, num_test_imgs))
        print("  ----------------------------------------")


class BaseDataset(object):
    """
    Base class of reid dataset
    """
    def __init__(self, root):
        self.root = osp.expanduser(root)

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def get_videodata_info(self, data, return_tracklet_stats=False):
        pids, cams, tracklet_stats = [], [], []
        for img_paths, pid, camid in data:
            pids += [pid]
            cams += [camid]
            tracklet_stats += [len(img_paths)]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_tracklets = len(data)
        if return_tracklet_stats:
            return num_pids, num_tracklets, num_cams, tracklet_stats
        return num_pids, num_tracklets, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class BaseVideoDataset(BaseDataset):
    """
    Base class of video reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_tracklets, num_train_cams, train_tracklet_stats = \
            self.get_videodata_info(train, return_tracklet_stats=True)

        num_query_pids, num_query_tracklets, num_query_cams, query_tracklet_stats = \
            self.get_videodata_info(query, return_tracklet_stats=True)

        num_gallery_pids, num_gallery_tracklets, num_gallery_cams, gallery_tracklet_stats = \
            self.get_videodata_info(gallery, return_tracklet_stats=True)

        tracklet_stats = train_tracklet_stats + query_tracklet_stats + gallery_tracklet_stats
        min_num = np.min(tracklet_stats)
        max_num = np.max(tracklet_stats)
        avg_num = np.mean(tracklet_stats)

        print("Dataset statistics:")
        print("  -------------------------------------------")
        print("  subset   | # ids | # tracklets | # cameras")
        print("  -------------------------------------------")
        print("  train    | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_cams))
        print("  query    | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_cams))
        print("  gallery  | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets, num_gallery_cams))
        print("  -------------------------------------------")
        print("  number of images per tracklet: {} ~ {}, average {:.2f}".format(min_num, max_num, avg_num))
        print("  -------------------------------------------")
