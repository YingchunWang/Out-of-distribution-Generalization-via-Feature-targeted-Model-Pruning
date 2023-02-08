import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd
import torch.utils.data as Data
import torchvision 
from maskvgg import maskvgg11
import  cv2
import os
import logging
import math
from logging import FileHandler
from logging import StreamHandler
import torch.optim.lr_scheduler as lr_scheduler
from common import  *
from prun import *
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

use_cuda = torch.cuda.is_available()
path_list=["/home/lthpc/wyc/wyc/Prun_For_OOD/colored_mnist/checkpoint/rex_pruned_2.pth.tar","/home/lthpc/wyc/wyc/Prun_For_OOD/colored_mnist/checkpoint/cop_pruned_2.pth.tar"]
for p in range(len(path_list)):
  model = torch.load(path_list[p])
  for n,p in model.named_parameters():
      print(n,p.size())