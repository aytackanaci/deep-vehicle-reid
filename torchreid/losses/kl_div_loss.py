from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class KLDivLoss(nn.Module):
    """KL Div loss with label smoothing regularizer.
    
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    
    Equation: y = (1 - epsilon) * y + epsilon / K.
    
    Args:
    - num_classes (int): number of classes
    - epsilon (float): weight
    - use_gpu (bool): whether to use gpu devices
    - label_smooth (bool): whether to apply label smoothing, if False, epsilon = 0
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True, multiclass=False):
        super(KLDivLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.multiclass = multiclass
        self.loss = nn.KLDivLoss()
        
    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        log_probs_inputs = self.logsoftmax(inputs)
        probs_targets = self.softmax(targets)
        
        if not self.use_gpu: probs_targets = probs_targets.data.cpu()
        probs_targets = (1 - self.epsilon) * probs_targets + self.epsilon / self.num_classes
        return self.loss(log_probs_inputs, probs_targets)
