#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np
import os, sys, time

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import random



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class MaskBlock(nn.Module):
    def __init__(self, in_channels, out_channels, args=None): 
        super(MaskBlock, self).__init__() 
   
       
       

        if out_channels < 80:
            squeeze_rate = args.squeeze_rate // 2
        else:
            squeeze_rate = args.squeeze_rate
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc1 = nn.Linear(in_channels, out_channels // squeeze_rate, bias=False)
        self.fc2 = nn.Linear(out_channels // squeeze_rate, out_channels, bias=True)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 1.0) 
        
        self.register_buffer('mask_sum', torch.zeros(out_channels))
        self.register_buffer('thre',torch.zeros(1))
        #self.thre.fill_(args.thre_init)
         
    def forward(self, x):

        batch_size = x.shape[0]
        x_averaged = self.avg_pool(x).squeeze(-1).squeeze(-1)
        y = self.fc1(x_averaged)
        y = F.relu(y,inplace=True)
        y = self.fc2(y)

        mask_before = F.sigmoid(y) 
      
        _lasso=mask_before.mean(dim=-1)
        
        if self.training:
            self.mask_sum.add_(mask_before.data.sum(dim=0)) #
        thre = torch.quantile(mask_before, 0.75, dim=1).reshape(batch_size, -1)
        tmp=torch.ones_like(mask_before)
        tmp[mask_before.data<thre]=0 
        mask=mask_before*tmp
        
        return mask,_lasso,mask_before

 



class MaskedBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,args=None):
        super(MaskedBasicblock, self).__init__()
        
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample
        
  
  
        self.mb1=MaskBlock(inplanes, planes, args=args)
        self.mb2=MaskBlock(planes, planes, args=args)
    
        
    def forward(self, x):
        x,_mask_list,_lasso_loss,_mask_before_list,_avg_fea_list=x
        
        residual = x
 
 
        mask1,_lasso1,mask1_before=self.mb1(x)
        _mask_list.append(mask1)
        _lasso_loss.append(_lasso1)
        _mask_before_list.append(mask1_before)
            
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        

        _avg_fea_list.append(F.adaptive_avg_pool2d(basicblock,1))
        basicblock=basicblock* mask1.unsqueeze(-1).unsqueeze(-1) 
        mask2,_lasso2,mask2_before=self.mb2(basicblock)
        _mask_list.append(mask2)
        _lasso_loss.append(_lasso2)
        _mask_before_list.append(mask2_before)

        
        basicblock = self.conv_b(basicblock)  
        basicblock = self.bn_b(basicblock)
        
   
            
        _avg_fea_list.append(F.adaptive_avg_pool2d(basicblock,1))
        basicblock=basicblock* mask2.unsqueeze(-1).unsqueeze(-1)
  
     
        if self.downsample is not None:
            residual = self.downsample(x)
        
        return [F.relu(residual + basicblock, inplace=True),_mask_list,_lasso_loss,_mask_before_list,_avg_fea_list]


class CifarResNet(nn.Module):


    def __init__(self, block, depth, num_classes,args=None):
  
        super(CifarResNet, self).__init__()

        
        assert (depth - 2) % 6 == 0 
        
        
        layer_blocks = (depth - 2) // 6
        print('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1,args=args)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2,args=args)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2,args=args)
        #self.avgpool = nn.AvgPool2d(4)
        self.classifier = nn.Linear(1024 * block.expansion, num_classes)
       
        
        for name,m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
         
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and 'classifier' in name:
                print('init classifier')
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,args=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,args=args))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,args=args))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1([x,[],[],[],[]])
        x = self.stage_2(x)
        x = self.stage_3(x)
        
        _mask_list=x[1]
        _lasso_loss = x[2]
        _mask_before_list=x[3]
        _avg_fea_list=x[4]

        x = x[0].view(x[0].size(0), -1)
        
        x=self.classifier(x)
     
        return x,_mask_list,_lasso_loss,_mask_before_list,_avg_fea_list

def resnet20(num_classes=10,args=None):

    model = CifarResNet(MaskedBasicblock, 20, num_classes,args=args)
    return model


def resnet32(num_classes=10,args=None):

    model = CifarResNet(MaskedBasicblock, 32, num_classes,args=args)
    return model

def resnet56(num_classes=10,args=None):

    model = CifarResNet(MaskedBasicblock, 56, num_classes,args=args)
    return model