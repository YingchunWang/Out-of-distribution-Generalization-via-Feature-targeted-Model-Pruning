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


class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


class BuildingBlock(nn.Module):
    def do_pruning(self, in_channel_mask: np.ndarray, pruner: Callable[[np.ndarray], float], prune_mode: str):
        pass

    def get_conv_flops_weight(self, update: bool, scaling: bool):
        pass

    def get_sparse_modules(self):
        pass

    def config(self):
        pass

class MaxPool(nn.Module):
    def __init__(self): 
        super(MaxPool, self).__init__() 
        self.maxpool =  nn.MaxPool2d(kernel_size=1, stride=2)
    def forward(self,x):
        x,_mask_list,_lasso_loss,_mask_before_list,_avg_fea_list=x 
        x=self.maxpool(x)
        return [x,_mask_list,_lasso_loss,_mask_before_list,_avg_fea_list]
class MaskBlock(nn.Module):
    def __init__(self, in_channels, out_channels, args=None): 
        super(MaskBlock, self).__init__() 
          
        self.clamp_max=1000.0
        if out_channels < 80:
            squeeze_rate = 1
        else:
            squeeze_rate = 2            
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc1 = nn.Linear(in_channels, out_channels // squeeze_rate, bias=False)
        self.fc2 = nn.Linear(out_channels // squeeze_rate, out_channels, bias=True)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 1.0) 
        
        self.register_buffer('mask_sum', torch.zeros(out_channels))
        self.register_buffer('thre',torch.zeros(1))
        self.thre.fill_(-10000.0)
         
    def forward(self, x):
      
        x_averaged = self.avg_pool(x).squeeze(-1).squeeze(-1)
        y = self.fc1(x_averaged)
        y = F.relu(y,inplace=True)
        y = self.fc2(y)
        mask_before=F.sigmoid(y)       
        _lasso=mask_before.mean(dim=-1)#[batch]_lasso represent complexity for each sample,its length is batch num
        '''
        if self.training:
            self.mask_sum.add_(mask_before.data.sum(dim=0)) #[channel1'score,channel2'score...channeln'score]
        
        #Mean to compute mask
        mask=torch.ones_like(mask_before)        
        mask[mask_before.data<mask_before.mean()]=0 
        masked_score=mask_before*mask
        '''
        #Tred to compute mask
        batch_size = mask_before.shape[0]
        score, sort =torch.sort(mask_before.reshape(batch_size,-1),descending =True)
        thred = score[:, int(mask_before.shape[1] * 0.7)].reshape(batch_size,-1)
    
        mask = (mask_before > thred).float().to(mask_before.device)
        masked_score = mask_before * mask
        return masked_score,_lasso,mask_before, mask

class MaskVGGBlock(BuildingBlock):
    def __init__(self, conv: nn.Conv2d, batch_norm: bool, input_channel: int, output_channel: int):
        super().__init__()
        self.conv = conv      
        if isinstance(self.conv, nn.Conv2d):
            self.batch_norm = nn.BatchNorm2d(output_channel)
        elif isinstance(self.conv, nn.Linear):
            self.batch_norm = nn.BatchNorm1d(output_channel)

        self.relu = nn.ReLU(inplace=True)
        self.mb1=MaskBlock(input_channel, output_channel)
        
    @property
    def is_batch_norm(self):
        return not isinstance(self.batch_norm, Identity)
    def forward(self, x):
        x,_mask_list,_lasso_loss,_mask_before_list,_avg_fea_list=x       
      
        _mask1,_lasso1, _scorel, maskl=self.mb1(x)                
        _mask_list.append(_mask1)
        _lasso_loss.append(_lasso1)
        _mask_before_list.append(_scorel)       
        
        conv_out = self.conv(x)
        bn_out = self.batch_norm(conv_out)
        out = self.relu(bn_out)
    
        _avg_fea_list.append(F.adaptive_avg_pool2d(out,1))
        #out = out*_scorel.unsqueeze(-1).unsqueeze(-1)#only attention
        out=out* _mask1.unsqueeze(-1).unsqueeze(-1)#attention for sarse
        #out=out* maskl.unsqueeze(-1).unsqueeze(-1)
    
        return [out,_mask_list,_lasso_loss,_mask_before_list,_avg_fea_list]
    

    def do_pruning(self, in_channel_mask, out_channel_mask):
        conv_layer=self.conv
        bn_layer=self.batch_norm
        mask_layer= self.mb1    
        with torch.no_grad():
            conv_weight: torch.Tensor = conv_layer.weight.data.clone()
    
            # #############################################prune the input channel of the conv layer##############################
    
            idx_in = np.squeeze(np.argwhere(np.asarray(in_channel_mask)))
            if len(idx_in.shape) == 0:
                # expand the single scalar to array
                idx_in = np.expand_dims(idx_in, 0)
           
            # prune the input of the conv layer(prune channels in each filter of conv in this block)
            if isinstance(conv_layer, nn.Conv2d):
                conv_weight = conv_weight[:, idx_in.tolist(), :, :]
    
            elif isinstance(conv_layer, nn.Linear):
                conv_weight = conv_weight[:, idx_in.tolist()]
            else:
                raise ValueError(f"unsupported conv layer type: {conv_layer}")
    
            # *****************************prune the output channel of the conv layer(prune filters of conv in this block)**********************
           
            idx_out: np.ndarray = np.squeeze(np.argwhere(np.asarray(out_channel_mask)))
            if len(idx_out.shape) == 0:
                # 0-d scalar which means there in only on channel left in mask,idx is a scalar whose shape is zero
                idx_out = np.expand_dims(idx_out, 0)#change the inx to a one-dimentional array whose shape is one row
            
            if isinstance(conv_layer, nn.Conv2d):
                conv_weight = conv_weight[idx_out.tolist(), :, :, :]
            elif isinstance(conv_layer, nn.Linear):
                conv_weight = conv_weight[idx_out.tolist(), :]
                linear_bias = conv_layer.bias.clone()
                linear_bias = linear_bias[idx_out.tolist()]
            else:
                raise ValueError(f"unsupported conv layer type: {conv_layer}")
    
            # change the property of the conv layer
           
            if isinstance(conv_layer, nn.Conv2d):
                conv_layer.in_channels = len(idx_in)
                conv_layer.out_channels = len(idx_out)
            elif isinstance(conv_layer, nn.Linear):
                conv_layer.in_features = len(idx_in)
                conv_layer.out_features = len(idx_out)
            conv_layer.weight.data = conv_weight
            if isinstance(conv_layer, nn.Linear):
                conv_layer.bias.data = linear_bias
    
            # prune the bn layer
            if bn_layer is not None:
                bn_layer.weight.data = bn_layer.weight.data[idx_out.tolist()].clone()
                bn_layer.bias.data = bn_layer.bias.data[idx_out.tolist()].clone()
                bn_layer.running_mean = bn_layer.running_mean[idx_out.tolist()].clone()
                bn_layer.running_var = bn_layer.running_var[idx_out.tolist()].clone()
                # set bn properties
                bn_layer.num_features = len(idx_out)
            
            if mask_layer is not None:
                fc1_weight = mask_layer.fc1.weight.data.clone()
                fc1_weight = fc1_weight[:, idx_in.tolist()]
                mask_layer.fc1.in_features = len(idx_in)
                mask_layer.fc1.weight.data = fc1_weight
                
                fc2_weight = mask_layer.fc2.weight.data.clone()
                fc2_weight = fc2_weight[idx_out.tolist(), :]
                mask_layer.fc2.out_features = len(idx_out)
                mask_layer.fc2.weight.data = fc2_weight
                
                fc2_bias = mask_layer.fc2.bias.data.clone()
                fc2_bias = fc2_bias[idx_out.tolist()]
                mask_layer.fc2.bias.data = fc2_bias
        return 

    def config(self) -> Tuple[int]:
        if isinstance(self.conv, nn.Conv2d):
            return (self.conv.out_channels,)

        else:
            raise ValueError(f"Unsupport conv type: {self.conv}")
    def get_sparse_modules(self) -> Tuple[nn.Module]:
 
        if self.is_batch_norm:
            return (self.batch_norm,)
        else:
            raise ValueError("No sparse layer available")

class VGG(nn.Module):
    def __init__(self, num_classes, depth, cfg = None, init_weights=True, att = False, linear=False, bn_init_value=1, width_multiplier=1.):
        super(VGG, self).__init__()
        self.att =  att
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
                layers.append(MaxPool())
                #layers +=[nn.MaxPool2d(kernel_size=1, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                layers.append(MaskVGGBlock(conv=conv2d, batch_norm=batch_norm, input_channel=in_channels, output_channel=v))
                in_channels = v              
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.feature([x,[],[],[],[]])
        _mask_list=x[1]
        _lasso_loss = x[2]
        _mask_before_list=x[3]
        _avg_fea_list=x[4]     
        x = x[0].view(x[0].size(0), -1)  
        x = self.classifier(x)        
        return x,_mask_list,_lasso_loss,_mask_before_list,_avg_fea_list
    

       
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

    def prune_model(self, x) -> None:
        x,_mask_list,_lasso_loss,_mask_before_list,_avg_fea_list = self.forward(x)
        
        input_mask = np.ones(3)
        pruned_mask = [input_mask]
        conv_index=0
        batch_size = _mask_before_list[0].shape[0]
        for i in range(len(_mask_before_list)):
            total_mask_before = torch.sum(_mask_before_list[i],dim=0)
            sco,sort = torch.sort(total_mask_before, descending = True)
            maskl = torch.ge(total_mask_before , sco[int(sco.shape[0]*0.7)])
            pruned_mask.append(maskl)
            #sco,total_mask_before,maaskl,_mask_before_list[i] torch.Size([64]) torch.Size([64]) torch.Size([64]) torch.Size([20000, 64])
        print('Start to Prune')            
        for submodule in self.modules():
            if isinstance(submodule, MaskVGGBlock):                
                submodule: MaskVGGBlock                               
                submodule.do_pruning(in_channel_mask=pruned_mask[conv_index], out_channel_mask=pruned_mask[conv_index+1])
                conv_index+=1
        # prune the last linear layer
        linear_weight = self._logit_layer.weight.data.clone()
        idx_in = np.squeeze(np.argwhere(np.asarray(pruned_mask[-1])))
        if len(idx_in.shape) == 0:
            # expand the single scalar to array
            idx_in = np.expand_dims(idx_in, 0)
        linear_weight = linear_weight[:, idx_in.tolist()]
        self._logit_layer.in_features = len(idx_in)
        self._logit_layer.weight.data = linear_weight

    @property
    def _logit_layer(self) -> nn.Linear:
        if self._linear:
            return self.classifier[-1]
        else:
            return self.classifier

    def get_sparse_layers(self) -> List[nn.Module]:
        sparse_layers: List[nn.Module] = []
        for submodule in self.modules():
            if isinstance(submodule, MaskVGGBlock):
                submodule: MaskVGGBlock

                if submodule.is_batch_norm:
                    sparse_layers.append(submodule.batch_norm)
                else:
                    raise ValueError("No sparse modules available.")

        return sparse_layers



    @property
    def building_block(self):
        return MaskVGGBlock

    def config(self) -> List[int]:
        config = []
        for submodule in self.modules():
            if isinstance(submodule, self.building_block):
                for c in submodule.config():
                    config.append(c)
            elif isinstance(submodule, MaxPool):
                config.append('M')

        return config

def maskvgg11(num_classes,cfg=None):
    depth = 11
    return VGG(num_classes, depth, cfg=cfg)

def vgg16_linear(num_classes):
    depth = 16
    return VGG(num_classes, depth=16, init_weights=True, linear=True)


def vgg16(num_classes):
    depth = 16
    return VGG(num_classes, depth)


def _test_load_state_dict(net: nn.Module, net_ref: nn.Module):
    conv_list = []
    bn_list = []
    linear_list = []

    conv_idx = 0
    bn_idx = 0
    linear_idx = 0

    for submodule in net.modules():
        if isinstance(submodule, nn.Conv2d):
            conv_list.append(submodule)
        elif isinstance(submodule, nn.BatchNorm2d) or isinstance(submodule, nn.BatchNorm1d):
            bn_list.append(submodule)
        elif isinstance(submodule, nn.Linear):
            linear_list.append(submodule)

    for submodule in net_ref.modules():
        if isinstance(submodule, nn.Conv2d):
            conv_list[conv_idx].load_state_dict(submodule.state_dict())
            conv_idx += 1
        elif isinstance(submodule, nn.BatchNorm2d) or isinstance(submodule, nn.BatchNorm1d):
            bn_list[bn_idx].load_state_dict(submodule.state_dict())
            bn_idx += 1
        elif isinstance(submodule, nn.Linear):
            linear_list[linear_idx].load_state_dict(submodule.state_dict())
            linear_idx += 1


def _check_model_same(net_wo_gate: nn.Module, net_w_gate: nn.Module):
    state_dict = {}
    for key, value in net_w_gate.state_dict().items():
        if key in net_wo_gate.state_dict():
            state_dict[key] = net_wo_gate.state_dict()[key]
        else:
            state_dict[key] = net_w_gate.state_dict()[key]
            print(f"Missing param: {key}")

    print()
    net_w_gate.load_state_dict(state_dict)
