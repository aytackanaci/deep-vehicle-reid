from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class SelectedMSELoss(nn.Module):

    "Mean squared error loss calculated from selected elements."

    def __init__(self, use_gpu=True):
        super(MSELoss, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        if self.use_gpu: targets = targets.cuda()
        losses = nn.MSELoss(inputs, targets, reduction='none')
        loss = torch.mean(losses.index_select(targets == 0))

        return loss
