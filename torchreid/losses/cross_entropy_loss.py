from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes
    - epsilon (float): weight
    - use_gpu (bool): whether to use gpu devices
    - label_smooth (bool): whether to apply label smoothing, if False, epsilon = 0
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True, multiclass=False, soft_targets=False, multilabel=False, weighting=1, use_batch_subset=False):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.use_batch_subset = use_batch_subset
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.multiclass = multiclass
        self.soft_targets = soft_targets
        self.multilabel = multilabel
        self.weighting = weighting
        self.use_batch_subset = use_batch_subset

        if self.soft_targets and not self.multiclass:
            print('Warning: soft targets should also be multiclass. Softmax will not be applied otherwise')

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """

        if self.use_batch_subset:
            indices = targets.ne(-1).nonzero().transpose(0,1)[0].unique(sorted=True)
            inputs = torch.index_select(inputs,0,indices)
            targets = torch.index_select(targets,0,indices)

        log_probs = self.logsoftmax(inputs)

        if not self.multiclass:
            targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        elif self.soft_targets:
            targets = F.softmax(targets,dim=1)

        if self.use_gpu: targets = targets.cuda()

        if self.multilabel:
            loss = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            loss = (- targets * log_probs).mean(0).sum()

        loss = loss*self.weighting

        return loss
