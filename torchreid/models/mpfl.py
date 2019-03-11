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
                 loss, pretrained,
                 scales=[224],
                 dropout_p=0.001,
                 train_orient=True,
                 train_landmarks=True,
                 train_grayscale=False,
                 regress_landmarks=False,
                 **kwargs):

        super(MPFL, self).__init__()

        print('scales:',scales)
        self.train_scales = len(scales) > 1
        print(self.train_scales)
        self.train_orient = train_orient
        self.train_landmarks = train_landmarks
        self.train_grayscale = train_grayscale
        self.regress_lms = regress_landmarks

        self.loss = loss
        self.num_classes = num_classes
        self.num_orients = num_orients
        self.num_landmarks = num_landmarks

        if self.train_orient and self.num_orients <= 1:
            print('Error! Require more than one orient to train model with orient branch')
        if self.train_landmarks and self.num_landmarks == 0:
            print('Error! Require at least one landmark to train model with landmarks branch')

        self.dropout_p = dropout_p

        print('Model created with num pids:',self.num_classes,'num orients:',self.num_orients,'num_landmarks:',self.num_landmarks)

        self.id_branch = mobilenetv2ws(num_classes=num_classes,
                                       loss=loss,
                                       input_size=scales[0],
                                       pretrained=pretrained)
        self.id_branch.feature_extract_mode = True
        self.dropout_id = nn.Dropout(p=dropout_p, inplace=True)
        self.fc_id = nn.Linear(self.id_branch.last_conv_out_ch, self.num_classes)

        if self.train_scales:
            self.id_small_branch = mobilenetv2ws(num_classes=num_classes,
                                                 loss=loss,
                                                 input_size=scales[1],
                                                 pretrained=pretrained)
            self.id_small_branch.feature_extract_mode = True
            self.dropout_id_small = nn.Dropout(p=dropout_p, inplace=True)
            self.fc_id_small = nn.Linear(self.id_small_branch.last_conv_out_ch, self.num_classes)

        if self.train_grayscale:
            self.id_grayscale_branch = mobilenetv2ws(num_classes=num_classes,
                                                     loss=loss,
                                                     input_size=scales[0],
                                                     pretrained=pretrained)
            self.id_grayscale_branch.feature_extract_mode = True
            self.dropout_id_grayscale = nn.Dropout(p=dropout_p, inplace=True)
            self.fc_id_grayscale = nn.Linear(self.id_grayscale_branch.last_conv_out_ch, self.num_classes)

        if self.train_orient:
            self.orient_branch = mobilenetv2ws(num_classes=num_orients,
                                               loss=loss,
                                               input_size=scales[0],
                                               pretrained=pretrained)
            self.orient_branch.feature_extract_mode = True
            self.dropout_orient = nn.Dropout(p=dropout_p, inplace=True)
            self.fc_orient = nn.Linear(self.orient_branch.last_conv_out_ch, self.num_orients)
            self.fc_orient_id = nn.Linear(self.orient_branch.last_conv_out_ch, self.num_classes)

        if self.train_landmarks:
            self.landmarks_branch = mobilenetv2ws(num_classes=num_landmarks,
                                                  loss=loss,
                                                  input_size=scales[1],
                                                  pretrained=pretrained)
            self.landmarks_branch.feature_extract_mode = True
            self.fc_landmarks = nn.Linear(self.landmarks_branch.last_conv_out_ch, self.num_landmarks)
            self.dropout_landmarks = nn.Dropout(p=dropout_p, inplace=True)
            self.fc_landmarks_id = nn.Linear(self.landmarks_branch.last_conv_out_ch, self.num_classes)

        self.dropout_consensus = nn.Dropout(p=dropout_p, inplace=True)


        self.fusion_last_conv_out = self.id_branch.last_conv_out_ch
        if self.train_scales:
            self.fusion_last_conv_out += self.id_small_branch.last_conv_out_ch
        if self.train_grayscale:
            self.fusion_last_conv_out += self.id_grayscale_branch.last_conv_out_ch
        if self.train_orient:
            self.fusion_last_conv_out += self.orient_branch.last_conv_out_ch
        if self.train_landmarks:
            self.fusion_last_conv_out += self.landmarks_branch.last_conv_out_ch
        self.fc_consensus = nn.Linear(self.fusion_last_conv_out, self.num_classes)

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

    def forward(self, x1, x2=None, x3=None):

        f_id = self.id_branch(x1)
        f_id = self.dropout_id(f_id)
        y_id = self.fc_id(f_id)

        if self.train_scales:
            assert(x2 is not None, "Small image required for training scaled branch")
            f_id_small = self.id_small_branch(x2)
            f_id_small = self.dropout_id_small(f_id_small)
            y_id_small = self.fc_id_small(f_id_small)
        else:
            y_id_small = 0

        if self.train_grayscale:
            assert(x3 is not None, "Grayscale image required for training grayscale branch")
            f_id_grayscale = self.id_grayscale_branch(x3)
            f_id_grayscale = self.dropout_id_grayscale(f_id_grayscale)
            y_id_grayscale = self.fc_id_grayscale(f_id_grayscale)
        else:
            y_id_grayscale = 0

        if self.train_orient:
            f_orient = self.orient_branch(x1)
            f_orient = self.dropout_orient(f_orient)
            y_orient = self.fc_orient(f_orient)
            y_orient_id = self.fc_orient_id(f_orient)
        else:
            y_orient = 0
            y_orient_id = 0

        if self.train_landmarks:
            f_landmarks = self.landmarks_branch(x1)
            f_landmarks = self.dropout_landmarks(f_landmarks)
            y_landmarks = self.fc_landmarks(f_landmarks)
            y_landmarks_id = self.fc_landmarks_id(f_landmarks)
        else:
            y_landmarks = 0
            y_landmarks_id = 0

        f_fusion = f_id

        if self.train_scales:
            f_fusion = torch.cat([f_fusion, f_id_small], 1)
        if self.train_grayscale:
            f_fusion = torch.cat([f_fusion, f_id_grayscale], 1)
        if self.train_orient:
            f_fusion = torch.cat([f_fusion, f_orient], 1)
        if self.train_landmarks:
            f_fusion = torch.cat([f_fusion, f_landmarks], 1)

        f_fusion = self.dropout_consensus(f_fusion)

        if not self.training:
            return f_fusion

        y_consensus = self.fc_consensus(f_fusion)

        if self.loss == {'xent'}:
            return y_id, y_id_small, y_id_grayscale, y_orient, y_landmarks, y_orient_id, y_landmarks_id, y_consensus
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def mpfl(num_classes, num_orients, num_landmarks, loss, pretrained='imagenet', **kwargs):

    model = MPFL(num_classes, num_orients, num_landmarks, loss, pretrained,  **kwargs)

    return model
