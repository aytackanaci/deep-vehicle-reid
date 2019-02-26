from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.nn import init

from .mobilenetv2_pre import mobilenetv2ws, MobileNetV2wS

__all__ = ['dpfl']

class DPFL(nn.Module):
    """DPFL Implementation
    """


    def __init__(self, num_classes,
                 loss,
                 scales=[1.0, 0.5],
                 dropout_p=0.001,
                 **kwargs):

        super(DPFL, self).__init__()

        self.loss = loss
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        #scale 1.0

        self.branch_large = mobilenetv2ws(num_classes=num_classes,
                                loss=loss,
                                input_size=224,
                                pretrained='imagenet',
                                )

        self.branch_small = mobilenetv2ws(num_classes=num_classes,
                                loss=loss,
                                input_size=160,
                                pretrained='imagenet',
                                )

        self.branch_large.feature_extract_mode = True
        self.branch_small.feature_extract_mode = True

        self.dropout_small = nn.Dropout(p=dropout_p, inplace=True)
        self.dropout_large = nn.Dropout(p=dropout_p, inplace=True)
        self.dropout_joint = nn.Dropout(p=dropout_p, inplace=True)


        self.fc_large = nn.Linear(self.branch_large.last_conv_out_ch, self.num_classes)
        self.fc_small = nn.Linear(self.branch_small.last_conv_out_ch, self.num_classes)

        self.fc_joint = nn.Linear(
                self.branch_large.last_conv_out_ch + self.branch_small.last_conv_out_ch,
                self.num_classes)

        # self.fc_consensus = self._construct_fc_layer(
        #         [self.num_classes],
        #         self.scale10.last_conv_out_ch + self.scale05.last_conv_out_ch,
        #         dropout_p=0.2)

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

    def forward(self, x_large, x_small):

        f_large = self.branch_large(x_large)
        f_small = self.branch_small(x_small)

        f_large = self.dropout_large(f_large)
        f_small = self.dropout_small(f_small)

        y_large = self.fc_large(f_large)
        y_small = self.fc_small(f_small)

        f_fusion = torch.cat([f_large, f_small], 1)
        f_fusion = self.dropout_joint(f_fusion)

        if not self.training:
            return f_fusion

        y_joint = self.fc_joint(f_fusion)

        if self.loss == {'xent'}:
            return y_large, y_small, y_joint
        elif self.loss == {'xent', 'htri'}:
            return y_large, y_small, y_joint, f_fusion
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def dpfl(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DPFL(num_classes, loss, pretrained, **kwargs)

    return model
