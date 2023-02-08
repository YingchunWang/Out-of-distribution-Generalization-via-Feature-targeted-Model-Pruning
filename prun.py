from common import *
#from maskvgg import *

import copy
import os
from typing import Any, Dict
import torch

import torch
import torch.nn as nn


        

def bn_sparsity(model, sparsity, t, alpha):
    """

    :type model: torch.nn.Module
    :type alpha: float
    :type t: float
    :type sparsity: float
    :type loss_type: LossType
    """
    bn_modules = model.get_sparse_layers()

  
    # compute global mean of all sparse vectors
    n_ = sum(map(lambda m: m.weight.data.shape[0], bn_modules))
    sparse_weights_mean = torch.sum(torch.stack(list(map(lambda m: torch.sum(m.weight), bn_modules)))) / n_

    sparsity_loss = 0.

    for m in bn_modules:
        sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(torch.abs(m.weight - alpha * sparse_weights_mean))

        sparsity_loss += sparsity * sparsity_term

        return sparsity_loss



def prune_vgg(num_classes, sparse_model, data):
    pruned_model = copy.deepcopy(sparse_model)
    pruned_model.cpu()
    if True:
      pruned_model.prune_model(data)
    else:
      pruned_model.prune_model(pruner=lambda weight: search_threshold(weight))
      pruned_model.prune_model(pruner=lambda weight: l1_norm_threshold(weight))       
    print("Pruning finished. cfg:")
    print(pruned_model.config())
    for n,p in pruned_model.named_parameters():
        print(n,p.size())
    
    '''
    # load weight to finetuning model
    saved_model = vgg11(num_classes=num_classes, cfg=pruned_model.config() )

    pruned_state_dict = {}
    # remove fc param from model
    for param_name, param in pruned_model.state_dict().items():
        if param_name in saved_model.state_dict():
            pruned_state_dict[param_name] = param
        else:
            if "_conv" not in param_name:
                # when the entire block is pruned, the conv parameter will miss, which is expected
                print(f"[WARNING] abandon parameter: {param_name}")

    saved_model.load_state_dict(pruned_state_dict,strict=False)
    '''

    return pruned_model



 
def prune_while_training(model,num_classes: int, data):

    saved_model = prune_vgg(num_classes=num_classes, sparse_model=model, data = data)
    
   
    return saved_model 
