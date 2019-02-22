from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import pdb

from .mobilenetv2_pre import mobilenetv2ws, MobileNetV2wS
from .resnet import resnet18_class

__all__ = ['bilinear']

class Bilinear(nn.Module):
    """DPFL Implementation
    """


    def __init__(self, num_classes,
                 loss,
                 pretrained=None,
                 **kwargs):

        super(Bilinear, self).__init__()

        self.loss = loss
        self.num_classes = num_classes
        # self.dropout_p = dropout_p

        self.backbone = resnet18_class(num_classes, loss, pretrained)
        self.backbone.backbone_mode = True

        self.feature_dim = self.backbone.feature_dim ** 2
        self.fc = torch.nn.Linear(self.feature_dim, num_classes)

        self.init_params()

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), "fc_dims must be either list or tuple, but got {}".format(type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # print(x.size())
        x = self.backbone(x)
        # print(x.size())
        (N, D, H, W) = x.size() # last layer dimensions

        x = x.view(N, D, H**2)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (H**2)  # Bilinear
        assert x.size() == (N, D, D)
        x = x.view(N, D**2)
        x = torch.sqrt(x + 1e-5)
        f = torch.nn.functional.normalize(x) # normalized features
        # print(f.size())

        # if not self.training:
        #     return f

        y = self.fc(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def bilinear(num_classes, loss, pretrained='imagenet', **kwargs):

    model = Bilinear(num_classes, loss, pretrained, **kwargs)

    return model
