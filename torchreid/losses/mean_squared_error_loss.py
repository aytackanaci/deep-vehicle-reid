from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class SelectedMSELoss(nn.Module):

    "Mean squared error loss calculated from selected elements."

    def __init__(self, batch_size, im_size, use_gpu=True):
        super(SelectedMSELoss, self).__init__()
        self.use_gpu = use_gpu
        self.mse_loss = nn.MSELoss(reduction='none')
        self.batch_size = batch_size
        self.im_size = im_size

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        if self.use_gpu: targets = targets.cuda()
        losses = self.mse_loss(inputs, targets)

        # Only sum losses for landmarks that are present in the target
        # Divide through by the average number of examples present for each landmark
        mask = targets.gt(0.0)
        loss = torch.masked_select(losses,mask).sum()/(mask.sum(0).float().mean()*self.im_size**2)

        return loss
