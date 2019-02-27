from __future__ import absolute_import
from __future__ import division

from torchvision.transforms import *
from torchvision.transforms import functional as F
import torch

from PIL import Image
import random
import numpy as np

def crop_image(img, width, height, resize_ratio, interpolation):
    new_width, new_height = int(round(width * resize_ratio)), int(round(height * resize_ratio))
    resized_img = img.resize((new_width, new_height), interpolation)
    x_maxrange = new_width - width
    y_maxrange = new_height - height
    x1 = int(round(random.uniform(0, x_maxrange)))
    y1 = int(round(random.uniform(0, y_maxrange)))
    cropped_img = resized_img.crop((x1, y1, x1 + width, y1 + height))
    return cropped_img, x1, y1

# Transform the landmarks in the same way the image is cropped
def stretch_and_crop_landmarks(landmarks, im_size, new_width, new_height, resize_ratio, x_start, y_start):
    landmarks_t = landmarks
    landmarks_t[range(0,len(landmarks),2)] = landmarks[range(0,len(landmarks),2)]/im_size[0]*new_width*resize_ratio
    landmarks_t[range(1,len(landmarks)+1,2)] = landmarks[range(1,len(landmarks)+1,2)]/im_size[1]*new_height*resize_ratio
    
    for idx in range(0,len(landmarks_t),2):
        l_x = landmarks_t[idx]
        l_y = landmarks_t[idx+1]
        if landmarks[idx] == -1 or \
           l_x < x_start or l_y < y_start or \
           l_x > (x_start+new_width) or l_y > (y_start+new_height):
            landmarks_t[idx:idx+2] = 0

    return landmarks_t

def flip_orient(orient):
    if orient < 2:
        return orient
    elif orient < 5:
        return (orient+3)
    else:
        return (orient-3)

def flip_landmarks(landmarks, im_size):
    im_centre = np.array(im_size)/2

    #TODO!!!
    return landmarks

def class_landmarks(landmarks):
    return np.array(list(map(lambda x: 1 if x>0 else 0, landmarks[range(0,len(landmarks),2)])))
    
class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, resize_ratio=1.125, interpolation=Image.BILINEAR, return_trans=False):
        self.height = height
        self.width = width
        self.resize_ratio = resize_ratio
        self.p = p
        self.interpolation = interpolation
        self.return_trans = return_trans

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        cropped_img, _, _ = crop_image(img, self.width, self.height, self.resize_ratio, self.interpolation)
        return cropped_img

class Random2DTranslationLabels(Random2DTranslation):

    def __call__(self, data):

        img, orient, landmarks = data
        
        if random.uniform(0, 1) > self.p:
            cropped_img = img.resize((self.width, self.height), self.interpolation)
            landmarks_t = stretch_and_crop_landmarks(landmarks, img.size, self.width, self.height, 0, 0, 0)
        else:
            cropped_img, x, y = crop_image(img, self.width, self.height, self.resize_ratio, self.interpolation)
            landmarks_t = stretch_and_crop_landmarks(landmarks, img.size, self.width, self.height, self.resize_ratio, x, y)
            
        return cropped_img, orient, landmarks_t

class RandomHorizontalFlipLabels(RandomHorizontalFlip):

    def __call__(self, data):

        img, orient, landmarks = data

        if random.random() < self.p:
            return F.hflip(img), flip_orient(orient), flip_landmarks(landmarks, img.size)
        return img, orient, landmarks
    
class ToClassificationLabels(object):

    def __call__(self, data):
        img, orient, landmarks = data
        return img, orient, class_landmarks(landmarks)

class ToTensorImage(object):

    def __call__(self, data):
        img, orient, landmarks = data
        return F.to_tensor(img), orient, landmarks

class NormalizeImage(Normalize):

    def __call__(self, data):
        tensor, orient, landmarks = data
        return F.normalize(tensor, self.mean, self.std), orient, landmarks

def build_transforms(height, width, is_train, inc_orient_lm=False, regress_landmarks=False, **kwargs):
    """Build transforms

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - is_train (bool): train or test phase.
    """
    
    # use imagenet mean and std as default
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transforms = []

    if is_train and inc_orient_lm:
        print('Build transform for both image and labels')
        transforms += [Random2DTranslationLabels(height, width)]
        transforms += [RandomHorizontalFlipLabels()]
        transforms += [ToTensorImage()]
        transforms += [NormalizeImage(mean=imagenet_mean, std=imagenet_std)]
        if not regress_landmarks:
            transforms += [ToClassificationLabels()]
    else:
        if is_train:
            transforms += [Random2DTranslation(height, width)]
            transforms += [RandomHorizontalFlip()]
        else:
            transforms += [Resize((height, width))]
    
        transforms += [ToTensor()]
        transforms += [Normalize(mean=imagenet_mean, std=imagenet_std)]

    transforms = Compose(transforms)

    return transforms
