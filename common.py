import typing
from abc import abstractmethod, ABCMeta
from typing import Union, Callable
import torch

from torch import nn
import numpy as np
def l1_norm_threshold(weight: np.ndarray, ratio = 0.5) -> np.ndarray:
    """return a bool array"""
    assert len(weight.shape) == 4, f"Only support conv weight, got shape: {weight.shape}"
    weight = np.abs(weight)
    out_channels = weight.shape[0]

    l1_norm = np.sum(weight, axis=(1, 2, 3))  # the length is same as output channel number
    num_keep = int(out_channels * (1 - ratio))

    arg_max = np.argsort(l1_norm)
    arg_max_rev = arg_max[::-1][:num_keep]
    mask = np.zeros(out_channels, dtype=np.bool)
    mask[arg_max_rev.tolist()] = True

    return mask


def prune_conv_layer(conv_layer: Union[nn.Conv2d, nn.Linear],
                     bn_layer: nn.BatchNorm2d,
                     sparse_layer: nn.BatchNorm2d,
                     pruner,
                     in_channel_mask,                                                     
                     sparse_layer_in = None,
                     prune_on="factor") -> typing.Tuple[np.ndarray, np.ndarray]:

    assert isinstance(conv_layer, nn.Conv2d) or isinstance(conv_layer, nn.Linear), f"conv_layer got {conv_layer}"


    if in_channel_mask is not None and sparse_layer_in is not None:
        raise ValueError("Conflict option: in_channel_mask and sparse_layer_in")



    with torch.no_grad():
        conv_weight: torch.Tensor = conv_layer.weight.data.clone()

        # #############################################prune the input channel of the conv layer##############################


        # convert mask to channel indexes
        idx_in = np.squeeze(np.argwhere(np.asarray(in_channel_mask)))
        if len(idx_in.shape) == 0:
            # expand the single scalar to array
            idx_in = np.expand_dims(idx_in, 0)
       
        # prune the input of the conv layer(prune channels in each filter of conv in this block)
        if isinstance(conv_layer, nn.Conv2d):
            if conv_layer.groups == 1:
                conv_weight = conv_weight[:, idx_in.tolist(), :, :]
            else:
                assert conv_weight.shape[1] == 1, "only works for groups == num_channels"
        elif isinstance(conv_layer, nn.Linear):
            conv_weight = conv_weight[:, idx_in.tolist()]
        else:
            raise ValueError(f"unsupported conv layer type: {conv_layer}")

        # *****************************prune the output channel of the conv layer(prune filters of conv in this block)**********************
       
        if True:
            # the sparse_layer.weight need to be flatten, because the weight of SparseGate is not 1d
            sparse_weight: np.ndarray = sparse_layer.weight.view(-1).data.cpu().numpy()
            max_index = np.argmax(sparse_weight)
            # prune according the bn layer
            output_threshold = pruner(sparse_weight)
            out_channel_mask: np.ndarray = sparse_weight > output_threshold
            #print('out_channel_mask',out_channel_mask)
        else:
            #sparse_weight: np.ndarray = sparse_layer.weight.view(-1).data.cpu().numpy()
            # in this case, the sparse weight should be the conv or linear weight
            out_channel_mask: np.ndarray = pruner(conv_weight.data.cpu().numpy())

        
        if not np.any(out_channel_mask):
            # there is no channel left
            #return out_channel_mask, in_channel_mask
            out_channel_mask[max_index] = True
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
        if isinstance(conv_layer, nn.Conv2d) and conv_layer.groups != 1:
            # set the new groups for dw layer (for MobileNet)
            conv_layer.groups = conv_layer.in_channels
            pass

        # prune the bn layer
        if bn_layer is not None:
            bn_layer.weight.data = bn_layer.weight.data[idx_out.tolist()].clone()
            bn_layer.bias.data = bn_layer.bias.data[idx_out.tolist()].clone()
            bn_layer.running_mean = bn_layer.running_mean[idx_out.tolist()].clone()
            bn_layer.running_var = bn_layer.running_var[idx_out.tolist()].clone()

            # set bn properties
            bn_layer.num_features = len(idx_out)
    return out_channel_mask, in_channel_mask

def search_threshold(weight: np.ndarray):

    hist_y, hist_x = np.histogram(weight, bins=100)   
    hist_y_diff = np.diff(hist_y)
    for i in range(len(hist_y_diff) - 1):
        if hist_y_diff[i] <= 0 <= hist_y_diff[i + 1]:
            threshold = hist_x[i + 1]
            #if threshold > 0.2:
            #    print(f"WARNING: threshold might be too large: {threshold}")
            #print('hist_x, hist_y, threshold:',)
            #print( weight, hist_y, threshold)
            return threshold
