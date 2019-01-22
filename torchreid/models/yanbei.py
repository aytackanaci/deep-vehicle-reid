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
                 **kwargs):

        super(DPFL, self).__init__()

        self.loss = loss
        self.num_classes = num_classes

        #scale 1.0

        self.scale10 = mobilenetv2ws(num_classes=num_classes,
                                loss=loss,
                                input_size=224,
                                pretrained='imagenet',
                                )

        self.scale05 = mobilenetv2ws(num_classes=num_classes,
                                loss=loss,
                                input_size=160,
                                pretrained='imagenet',
                                )

        self.scale10.feature_extract_mode = True
        self.scale05.feature_extract_mode = True


        self.fc10 = nn.Linear(self.scale10.last_conv_out_ch, self.num_classes)
        self.fc05 = nn.Linear(self.scale05.last_conv_out_ch, self.num_classes)

        self.fc_consensus = self._construct_fc_layer(
                [self.num_classes],
                self.scale10.last_conv_out_ch + self.scale05.last_conv_out_ch,
                dropout_p=0.2)

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

    def forward(self, x10, x05):

        f10 = self.scale10(x10)
        f05 = self.scale05(x05)

        y10 = self.fc10(f10)
        y05 = self.fc05(f05)

        f_fusion = torch.cat([f10, f05], 1)

        if not self.training:
            return f_fusion

        y_concensus = self.fc_consensus(f_fusion)

        if self.loss == {'xent'}:
            return y10, y05, y_concensus
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def dpfl(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DPFL(num_classes, loss, pretrained, **kwargs)

    return model
