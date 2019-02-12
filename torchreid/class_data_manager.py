from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader

from .dataset_loader import ClassificationDataset
from .datasets import init_class_dataset
from .transforms import build_transforms
from .samplers import RandomIdentitySampler

class CDataManager(object):

    @property
    def num_train_pids(self):
        return self._num_train_pids

    def return_dataloaders(self, scale=None):
        """
        Return trainloader and testloader dictionary
        """
        return self.trainloader, self.testloader_dict


    def return_testdataset_by_name(self, name):
        """
        Return query and gallery, each containing a list of (img_path, pid).
        """
        return self.testdataset_dict[name]['test']



class ClassDataManager(CDataManager):
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
                 height=224,
                 width=224,
                 train_batch_size=32,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 # num_instances=4, # number of instances per identity (for RandomIdentitySampler)
                 scales=None,
                 **kwargs
                 ):
        super(CDataManager, self).__init__()
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
        # self.num_instances = num_instances

        # Build train and test transform functions
        if scales is None:
            transform_train = build_transforms(self.height, self.width, is_train=True)
            transform_test = build_transforms(self.height, self.width, is_train=False)
        else:
            transform_train = [build_transforms(scale, scale, is_train=True) for scale in scales]
            transform_test = [build_transforms(scale, scale, is_train=False) for scale in scales]


        print("=> Initializing TRAIN (source) datasets")
        self.train = []
        self._num_train_pids = 0

        for name in self.source_names:
            dataset = init_class_dataset(
                root=self.root, name=name, split_id=self.split_id
            )

            for img_path, pid in dataset.train:
                pid += self._num_train_pids
                self.train.append((img_path, pid))

            self._num_train_pids += dataset.num_train_pids

        if scales is None:
            imageDataset = ClassificationDataset(self.train, transform=transform_train)
        else:
            imageDataset = MultiScaleImageDataset(self.train, transforms=transform_train)

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
        self.testloader_dict = {name: {'test': None} for name in self.target_names}
        self.testdataset_dict = {name: {'test': None} for name in self.target_names}

        for name in self.target_names:
            dataset = init_class_dataset(
                root=self.root, name=name, split_id=self.split_id
            )

            if scales is None:
                testImageDataset =   ClassificationDataset(dataset.test, transform=transform_test)
            else:
                queryImageDataset =   MultiScaleImageDataset(dataset.query, transforms=transform_test)

            self.testloader_dict[name]['test'] = DataLoader(
                testImageDataset,
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False
            )

            self.testdataset_dict[name]['test'] = dataset.test

        print("\n")
        print("  **************** Summary ****************")
        print("  train names      : {}".format(self.source_names))
        print("  # train datasets : {}".format(len(self.source_names)))
        print("  # train ids      : {}".format(self._num_train_pids))
        print("  # train images   : {}".format(len(self.train)))
        print("  test names       : {}".format(self.target_names))
        print("  *****************************************")
        print("\n")
