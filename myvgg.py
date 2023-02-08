import math
from typing import Callable, List, Tuple
from common import *


import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

__all__ = ['vgg11','vgg16_linear', 'vgg16']


defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class BuildingBlock(nn.Module):
    def config(self):
        pass

class VGGBlock(BuildingBlock):
    def __init__(self, conv: nn.Conv2d, batch_norm: bool, input_channel: int, output_channel: int):
        super().__init__()
        self.conv = conv      
        if isinstance(self.conv, nn.Conv2d):
            self.batch_norm = nn.BatchNorm2d(output_channel)
        elif isinstance(self.conv, nn.Linear):
            self.batch_norm = nn.BatchNorm1d(output_channel)
        self.relu = nn.ReLU(inplace=True)       

    def forward(self, x):      
       
        conv_out = self.conv(x)
        bn_out = self.batch_norm(conv_out)
        out = self.relu(bn_out)

        return out
    


class VGG(nn.Module):
    def __init__(self, num_classes, depth, cfg = None, init_weights=True, linear=False, bn_init_value=1, width_multiplier=1.):
        super(VGG, self).__init__()
        
        self._linear = linear

        if cfg is None:
            cfg: List[int] = defaultcfg[depth].copy()  # do not change the content of defaultcfg!
        
        self.feature = self.make_layers(cfg, True)

        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights(bn_init_value)

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
       
        for index, v in enumerate(cfg):
            if v == 'M':
                layers +=[nn.MaxPool2d(kernel_size=1, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                layers.append(MaskVGGBlock(conv=conv2d, batch_norm=batch_norm, input_channel=in_channels, output_channel=v))
                in_channels = v              
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)   
        x = x.view(x.size(0), -1)  
        x = self.classifier(x)        
        return x
    

       
    def _initialize_weights(self, bn_init_value=1):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_init_value)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias != None:
                    m.bias.data.zero_()

    @property
    def building_block(self):
        return VGGBlock

    def config(self) -> List[int]:
        config = []
        for submodule in self.modules():
            if isinstance(submodule, self.building_block):
                for c in submodule.config():
                    config.append(c)
            elif isinstance(submodule, MaxPool):
                config.append('M')

        return config

def vgg11(num_classes,cfg=None):
    depth = 11
    return VGG(num_classes, depth, cfg=cfg)

def vgg16_linear(num_classes):
    depth = 16
    return VGG(num_classes, depth=16, init_weights=True, linear=True)


def vgg16(num_classes):
    depth = 16
    return VGG(num_classes, depth)



