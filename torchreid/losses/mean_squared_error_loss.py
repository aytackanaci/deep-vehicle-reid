from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class SelectedMSELoss(nn.Module):

    "Mean squared error loss calculated from selected elements."

    def __init__(self, batch_size, im_size, use_gpu=True, use_batch_subset=False):
        super(SelectedMSELoss, self).__init__()
        self.use_gpu = use_gpu
        self.mse_loss = nn.MSELoss(reduction='none')
        self.batch_size = batch_size
        self.im_size = im_size
        self.use_batch_subset = use_batch_subset

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
            batch_size = len(indices)
        else:
            batch_size = self.batch_size

        if self.use_gpu: targets = targets.cuda()
        losses = self.mse_loss(inputs, targets)

        # Only sum losses for landmarks that are present in the target
        mask = targets.gt(0.0)
        loss = torch.masked_select(losses,mask).sum()/(batch_size*self.im_size**2)

        return loss
