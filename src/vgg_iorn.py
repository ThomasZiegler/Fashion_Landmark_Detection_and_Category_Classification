'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
from iorn.modules import ORConv2d
from iorn.functions import oralign2d
from iorn.functions import oralign1d

from src import const

__all__ = [
    'VGG', 'vgg16','orn_align1d_vgg16','orn_align2d_vgg16','orn_vgg16'
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, end_mode, nr_orientations=4):
        super(VGG, self).__init__()
        self.end_mode = end_mode
        self.features = nn.ModuleList(features)
        self.nr_orientations = nr_orientations

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # might need to be 3*3
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

         # Initialize weights
        for m in self.modules():
            if isinstance(m,ORConv2d):
                continue
            else:
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

    def forward(self, x):
        if self.end_mode == 2:
            for k in range(len(self.features)):
                x = self.features[k](x)

                if k == 22: # add align block after Conv4-3
                    x = oralign2d(x, self.nr_orientations)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, end_mode, nr_orientations=4, batch_norm=False):
    layers = []
    in_channels = 3

    if end_mode == 0:
        rotation_invariance = False
    else:
        rotation_invariance = True

    for index,v in enumerate(cfg):
        if v == 'R' : # reset flag use normal Conv layers after this flag
            rotation_invariance = False
            if end_mode != 0: # rescale nr of input channels when iorn is used
                in_channels = in_channels * nr_orientations
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            if not rotation_invariance:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                v = v // nr_orientations
                if index == 0:
                    conv2d = ORConv2d(in_channels, v, arf_config=(1,nr_orientations), kernel_size=3, padding=1)
                else:
                    conv2d = ORConv2d(in_channels, v, arf_config=nr_orientations, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v

    return layers


cfg = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 'R', 512, 512, 512, 'M'],
}


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D'],0),0)

def vgg16_iorn(nr_orientations=4, pretrained=False):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers(cfg['D'], 2, nr_orientations), 2, nr_orientations)

    if pretrained:
        checkpoint = torch.load(const.VGG_INIT_MODEL, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    return model


