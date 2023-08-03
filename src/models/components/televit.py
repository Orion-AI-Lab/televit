from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce
import torchsummary
import timm
from . import vit_base
import einops
import segmentation_models_pytorch as smp

class TeleViT(nn.Module):
    def __init__(self,in_channels,local_input_size=128,use_indices=False,use_global_input=False, patch_size=16,use_decoder=False,global_patch_size=None):
        super().__init__()
        self.use_indices = use_indices
        self.use_global_input = use_global_input
        self.decoder = use_decoder
        self.local_input_size = local_input_size
        sources = []
        if self.use_indices:
            sources.append('Indices')
        if self.use_global_input:
            sources.append('Global')

        print('Constructing TeleViT with input sources: ',sources)
        
        if self.decoder:
            print('Attach conv-based decoder')
        
        global_in_channels = in_channels
        
        self.model = vit_base.FireFormer(img_size=local_input_size, patch_size=patch_size, in_chans=in_channels, decoder=self.decoder, global_in_channels=global_in_channels, embed_dim=768, depth=8, num_heads=12, num_classes=2*local_input_size*local_input_size, use_indices=self.use_indices, use_global_input = self.use_global_input, global_patch_size=global_patch_size)
    
    def forward(self,x,x_indices,x_global=None):
        
        if self.use_indices:
            x_indices = x_indices.unsqueeze(1)
        else:
            x_indices = None
        
        out = self.model(x, x_indices, x_global= x_global)
        if not self.decoder:
            out = out.view((out.shape[0],2,self.local_input_size,self.local_input_size))
        return out

