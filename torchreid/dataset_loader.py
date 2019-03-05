from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import io

import torch
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transforms is not None:
            assert isinstance(self.transforms, (list, tuple))
            imgs = [transform(img) for transform in self.transforms]

        return imgs, pid, camid, img_path

class ImageLandmarksDataset(Dataset):
    """Image Person ReID Dataset with Landmarks"""
    def __init__(self, dataset, num_landmarks, transforms=None, regress_landmarks=False):
        self.dataset = dataset
        self.transforms = transforms
        self.num_landmarks = num_landmarks
        if regress_landmarks:
            self.landmarks_type=np.double
        else:
            self.landmarks_type=np.long

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img_path, pid, camid = data[0:3]

        img = read_image(img_path)
        im_size = img.size

        if len(data) > 3:
            orient, landmarks = data[3:5]
        else:
            orient, landmarks = -1, np.ones(self.num_landmarks,dtype=self.landmarks_type)*-1

        if self.transforms is not None:
            assert isinstance(self.transforms, (list, tuple))
            imgs, orients, landmark_sets = (), (), ()
            for transform in self.transforms:
                img_new, orient_new, landmarks_new = transform((img, orient, landmarks))
                imgs = imgs + (img_new,)
                orients = orients + (orient_new,)
                landmark_sets = landmark_sets + (landmarks_new,)

        return imgs, pid, camid, orients, landmark_sets, img_path

class ClassficationDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    _sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample_method='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample_method = sample_method
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample_method == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)

        elif self.sample_method == 'evenly':
            """
            Evenly sample seq_len items from num items.
            """
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num/self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32)*(num-1)])
            assert len(indices) == self.seq_len

        elif self.sample_method == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)

        else:
            raise ValueError("Unknown sample method: {}. Expected one of {}".format(self.sample_method, self._sample_methods))

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid
