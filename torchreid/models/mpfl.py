from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.nn import init

from .mobilenetv2_pre import mobilenetv2ws, MobileNetV2wS

__all__ = ['mpfl']

class MPFL(nn.Module):
    """Multi-pose Feature Learner Implementation
    """


    def __init__(self, num_classes, num_orients, num_landmarks,
                 loss,
                 scales=[1.0, 0.5],
                 dropout_p=0.001,
                 train_orient=True,
                 train_landmarks=True,
                 regress_landmarks=False,
                 **kwargs):

        super(MPFL, self).__init__()

        self.train_orient = train_orient
        self.train_landmarks = train_landmarks
        self.regress_lms = regress_landmarks
        
        self.loss = loss
        self.num_classes = num_classes
        self.num_orients = num_orients
        self.num_landmarks = num_landmarks

        self.dropout_p = dropout_p

        print('Model created with num pids:',self.num_classes,'num orients:',self.num_orients,'num_landmarks:',self.num_landmarks)

        self.id_branch = mobilenetv2ws(num_classes=num_classes,
                                loss=loss,
                                input_size=224,
                                pretrained='imagenet',
                                )

        self.orient_branch = mobilenetv2ws(num_classes=num_orients,
                                loss=loss,
                                input_size=224,
                                pretrained='imagenet',
                                )

        self.landmarks_branch = mobilenetv2ws(num_classes=num_landmarks,
                                loss=loss,
                                input_size=224,
                                pretrained='imagenet',
                                )

        self.id_branch.feature_extract_mode = True
        self.orient_branch.feature_extract_mode = True
        self.landmarks_branch.feature_extract_mode = True

        self.dropout_id = nn.Dropout(p=dropout_p, inplace=True)
        self.dropout_orient = nn.Dropout(p=dropout_p, inplace=True)
        self.dropout_landmarks = nn.Dropout(p=dropout_p, inplace=True)
        self.dropout_consensus = nn.Dropout(p=dropout_p, inplace=True)


        self.fc_id = nn.Linear(self.id_branch.last_conv_out_ch, self.num_classes)
        self.fc_orient = nn.Linear(self.orient_branch.last_conv_out_ch, self.num_orients)
        self.fc_landmarks = nn.Linear(self.landmarks_branch.last_conv_out_ch, self.num_landmarks)
        self.fc_orient_id = nn.Linear(self.orient_branch.last_conv_out_ch, self.num_classes)
        self.fc_landmarks_id = nn.Linear(self.landmarks_branch.last_conv_out_ch, self.num_classes)

        self.fusion_last_conv_out = self.id_branch.last_conv_out_ch
        if (train_orient):
            self.fusion_last_conv_out += self.orient_branch.last_conv_out_ch
        if (train_landmarks):
            self.fusion_last_conv_out += self.landmarks_branch.last_conv_out_ch
        self.fc_consensus = nn.Linear(self.fusion_last_conv_out, self.num_classes)

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

    def forward(self, x):

        f_id = self.id_branch(x)
        f_orient = self.orient_branch(x)
        f_landmarks = self.landmarks_branch(x)

        f_id = self.dropout_id(f_id)
        f_orient = self.dropout_orient(f_orient)
        f_landmarks = self.dropout_landmarks(f_landmarks)

        y_id = self.fc_id(f_id)
        y_orient = self.fc_orient(f_orient)
        y_landmarks = self.fc_landmarks(f_landmarks)
        y_orient_id = self.fc_orient_id(f_orient)
        y_landmarks_id = self.fc_landmarks_id(f_landmarks)
        
        f_fusion = f_id
        #print(f_fusion)
        if self.train_orient:
            f_fusion = torch.cat([f_fusion, f_orient], 1)
        if self.train_landmarks:
            f_fusion = torch.cat([f_fusion, f_landmarks], 1)
        #print(f_fusion)
#        f_fusion = torch.cat(f_fusion, 1)
        f_fusion = self.dropout_consensus(f_fusion)

        if not self.training:
            return f_fusion

        y_concensus = self.fc_consensus(f_fusion)

        if self.loss == {'xent'}:
            return y_id, y_orient, y_landmarks, y_orient_id, y_landmarks_id, y_concensus
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def mpfl(num_classes, num_orients, num_landmarks, loss, pretrained='imagenet', **kwargs):

    model = MPFL(num_classes, num_orients, num_landmarks, loss, pretrained, **kwargs)

    return model
