from __future__ import absolute_import
from __future__ import division

from torchvision.transforms import *
import torch

from PIL import Image
import random
import numpy as np


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR, return_trans=False):
        self.height = height
        self.width = width
        self.resize_ratio = 1.125
        self.p = p
        self.interpolation = interpolation
        self.return_trans = return_trans

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            if self.return_trans:
                return img.resize((self.width, self.height), self.interpolation), 0, 0
            else:
                return img.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * self.resize_ratio)), int(round(self.height * self.resize_ratio))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        if self.return_trans:
            return croped_img, x1, y1
        else:
            return croped_img

class RandomHorizontalFlipReturn(object):
    """Horizontally flip the given PIL Image randomly with a given probability 
    and return the image (flipped or not) with whether the flipping happened

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return functional.hflip(img), True
        return img, False

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class ComposeReturn(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        returns = ()
        for t in self.transforms:
            r_t = t(img)
            if isinstance(r_t, tuple):
                img = r_t[0]
                if len(r_t) > 1:
                    returns = returns + r_t[1:]
            else:
                img = r_t
        return img, returns

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
def build_transforms(height, width, is_train, return_trans=False, **kwargs):
    """Build transforms

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - is_train (bool): train or test phase.
    """
    
    # use imagenet mean and std as default
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

    transforms = []

    if is_train:
        if return_trans:
            transforms += [Random2DTranslation(height, width, return_trans=True)]
            transforms += [RandomHorizontalFlipReturn()]
        else:
            transforms += [Random2DTranslation(height, width)]
            transforms += [RandomHorizontalFlip()]
    else:
        transforms += [Resize((height, width))]
    
    transforms += [ToTensor()]
    transforms += [normalize]

    if return_trans:
        transforms = ComposeReturn(transforms)
    else:
        transforms = Compose(transforms)

    return transforms
