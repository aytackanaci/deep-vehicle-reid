from __future__ import absolute_import
from __future__ import print_function

import os
from torch.utils.data import DataLoader

from .dataset_loader import ImageDataset, ImageLandmarksDataset, VideoDataset
from .datasets import init_imgreid_dataset, init_vidreid_dataset
from .transforms import build_transforms
from .samplers import RandomIdentitySampler


class BaseDataManager(object):

    @property
    def num_train_pids(self):
        return self._num_train_pids

    @property
    def num_train_cams(self):
        return self._num_train_cams

    @property
    def num_train_orients(self):
        return self._num_train_orients

    @property
    def num_train_landmarks(self):
        return self._num_train_landmarks

    def return_dataloaders(self, scale=None, landmarks=False):
        """
        Return trainloader and testloader dictionary
        """
        return self.trainloader, self.testloader_dict


    def return_testdataset_by_name(self, name):
        """
        Return query and gallery, each containing a list of (img_path, pid, camid).
        """
        return self.testdataset_dict[name]['query'], self.testdataset_dict[name]['gallery']


class ImageDataManager(BaseDataManager):
    """
    Image-ReID data manager
    """

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 root,
                 split_id=0,
                 val=None,
                 height=256,
                 width=128,
                 train_batch_size=32,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 num_instances=4, # number of instances per identity (for RandomIdentitySampler)
                 cuhk03_labeled=False, # use cuhk03's labeled or detected images
                 cuhk03_classic_split=False, # use cuhk03's classic split or 767/700 split
                 aic19_manual_labels=False,
                 vehicleid_test_size='large',
                 scales=None,
                 keypoints_dirs=None,
                 regress_landmarks=False,
                 grayscale=False
                 ):
        super(ImageDataManager, self).__init__()
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.split_id = split_id
        self.val = val
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.num_instances = num_instances
        self.cuhk03_labeled = cuhk03_labeled
        self.cuhk03_classic_split = cuhk03_classic_split
        self.aic19_manual_labels = aic19_manual_labels
        self.vehicleid_test_size = vehicleid_test_size
        self.keypoints_dirs = keypoints_dirs
        if len(self.keypoints_dirs) != len(self.source_names):
            print('Warning! Keypoint directories given do not match number of source directories - keypoints not being used!')
            self.keypoints_dirs = ['']*len(self.source_names)

        print("=> Initializing TRAIN (source) datasets")
        self.train = []
        self.train_lm = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        self._num_train_orients = 1
        self._num_train_landmarks = 0

        for idx, name in enumerate(self.source_names):

            use_keypoints = False
            keypoints_dir = self.keypoints_dirs[idx]
            if os.path.exists(keypoints_dir):
                use_keypoints = True

            dataset = init_imgreid_dataset(
                root=self.root, name=name, split_id=self.split_id, cuhk03_labeled=self.cuhk03_labeled,
                cuhk03_classic_split=self.cuhk03_classic_split, val=self.val, keypoints_dir=keypoints_dir, regress_landmarks=regress_landmarks
            )

            for data in dataset.train:
                img_path, pid, camid = data[0:3]
                pid += self._num_train_pids
                camid += self._num_train_cams
                if use_keypoints:
                    orient, landmarks = data[3:5]
                    self.train.append((img_path, pid, camid, orient, landmarks))
                else:
                    self.train.append((img_path, pid, camid))

            # Need to keep ids across both landmark and typical dataset separate
            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

            if use_keypoints:
                # Assume that orientations and landmarks from different datasets will match
                self._num_train_orients = max(dataset.num_train_orients, self._num_train_orients)
                self._num_train_landmarks = max(dataset.num_train_landmarks, self._num_train_landmarks)

        inc_orient_lm = True if self._num_train_landmarks > 0 else False

        # Build train and test transform functions
        if scales is None:
            transform_train = [build_transforms(self.height, self.width, is_train=True)]
            transform_train_lm = [build_transforms(self.height, self.width, is_train=True, inc_orient_lm=inc_orient_lm, regress_landmarks=regress_landmarks)]
            transform_test = [build_transforms(self.height, self.width, is_train=False)]
            height = self.height
            width = self.width
        else:
            transform_train = [build_transforms(scale, scale, is_train=True) for scale in scales]
            transform_train_lm = [build_transforms(scale, scale, is_train=True, inc_orient_lm=inc_orient_lm, regress_landmarks=regress_landmarks) for scale in scales]
            transform_test = [build_transforms(scale, scale, is_train=False) for scale in scales]
            height = scales[0]
            width = scales[0]

        if grayscale:
            transform_train.append(build_transforms(height, width, is_train=True, grayscale=True))
            transform_train_lm.append(build_transforms(height, width, is_train=True, inc_orient_lm=inc_orient_lm, regress_landmarks=regress_landmarks, grayscale=True))
            transform_test.append(build_transforms(height, width, is_train=False, grayscale=True))

        if self._num_train_landmarks > 0:
            print('Create an image landmarks dataset.')

            imageDataset = ImageLandmarksDataset(self.train, self._num_train_landmarks, transforms=transform_train_lm, regress_landmarks=regress_landmarks)
        else:
            print('Create an image dataset')
            imageDataset = ImageDataset(self.train, transforms=transform_train)

        if self.train_sampler == 'RandomIdentitySampler':
            self.trainloader = DataLoader(
                imageDataset,
                sampler=RandomIdentitySampler(self.train, self.train_batch_size, self.num_instances),
                batch_size=self.train_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=True
            )
        else:
            self.trainloader = DataLoader(
                imageDataset,
                batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=True
            )

        print("=> Initializing TEST (target) datasets")
        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}

        for name in self.target_names:
            dataset = init_imgreid_dataset(
                root=self.root, name=name, split_id=self.split_id, cuhk03_labeled=self.cuhk03_labeled,
                cuhk03_classic_split=self.cuhk03_classic_split, aic19_manual_labels=self.aic19_manual_labels, val=self.val, vehicleid_test_size=self.vehicleid_test_size,
            )

            queryImageDataset =   ImageDataset(dataset.query, transforms=transform_test)
            galleryImageDataset = ImageDataset(dataset.gallery, transforms=transform_test)

            self.testloader_dict[name]['query'] = DataLoader(
                queryImageDataset,
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False
            )

            self.testloader_dict[name]['gallery'] = DataLoader(
                galleryImageDataset,
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False
            )

            self.testdataset_dict[name]['query'] = dataset.query
            self.testdataset_dict[name]['gallery'] = dataset.gallery

        print("\n")
        print("  **************** Summary ****************")
        print("  train names      : {}".format(self.source_names))
        print("  # train datasets : {}".format(len(self.source_names)))
        print("  # train ids      : {}".format(self._num_train_pids))
        print("  # train images   : {}".format(len(self.train)))
        print("  # train cameras  : {}".format(self._num_train_cams))
        if self._num_train_landmarks > 0:
            print("  # train orients  : {}".format(self._num_train_orients))
            print("  # train landmarks  : {}".format(self._num_train_landmarks))
        print("  test names       : {}".format(self.target_names))
        print("  *****************************************")
        print("\n")


class VideoDataManager(BaseDataManager):
    """
    Video-ReID data manager
    """

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 root,
                 split_id=0,
                 height=256,
                 width=128,
                 train_batch_size=32,
                 test_batch_size=100,
                 workers=4,
                 seq_len=15,
                 sample_method='evenly',
                 image_training=True # train the video-reid model with images rather than tracklets
                 ):
        super(VideoDataManager, self).__init__()
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.split_id = split_id
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.seq_len = seq_len
        self.sample_method = sample_method
        self.image_training = image_training

        # Build train and test transform functions
        transform_train = build_transforms(self.height, self.width, is_train=True)
        transform_test = build_transforms(self.height, self.width, is_train=False)

        print("=> Initializing TRAIN (source) datasets")
        self.train = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        for name in self.source_names:
            dataset = init_vidreid_dataset(root=self.root, name=name, split_id=self.split_id)

            for img_paths, pid, camid in dataset.train:
                pid += self._num_train_pids
                camid += self._num_train_cams
                if self.image_training:
                    # decompose tracklets into images
                    for img_path in img_paths:
                        self.train.append((img_path, pid, camid))
                else:
                    self.train.append((img_paths, pid, camid))

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

        if self.image_training:
            # each batch has image data of shape (batch, channel, height, width)
            self.trainloader = DataLoader(
                ImageDataset(self.train, transform=transform_train),
                batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=True
            )
        else:
            # each batch has image data of shape (batch, seq_len, channel, height, width)
            # note: this requires new training scripts
            self.trainloader = DataLoader(
                VideoDataset(self.train, seq_len=self.seq_len, sample_method=self.sample_method, transform=transform_test),
                batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=True
            )

        print("=> Initializing TEST (target) datasets")
        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}

        for name in self.target_names:
            dataset = init_vidreid_dataset(root=self.root, name=name, split_id=self.split_id)

            self.testloader_dict[name]['query'] = DataLoader(
                VideoDataset(dataset.query, seq_len=self.seq_len, sample_method=self.sample_method, transform=transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False,
            )

            self.testloader_dict[name]['gallery'] = DataLoader(
                VideoDataset(dataset.gallery, seq_len=self.seq_len, sample_method=self.sample_method, transform=transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False,
            )

            self.testdataset_dict[name]['query'] = dataset.query
            self.testdataset_dict[name]['gallery'] = dataset.gallery

        print("\n")
        print("  **************** Summary ****************")
        print("  train names       : {}".format(self.source_names))
        print("  # train datasets  : {}".format(len(self.source_names)))
        print("  # train ids       : {}".format(self._num_train_pids))
        if self.image_training:
            print("  # train images   : {}".format(len(self.train)))
        else:
            print("  # train tracklets: {}".format(len(self.train)))
        print("  # train cameras   : {}".format(self._num_train_cams))
        print("  test names        : {}".format(self.target_names))
        print("  *****************************************")
        print("\n")
