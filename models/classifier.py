from icecream import ic
from sys import exit as e

import torch 
from torch import nn

from einops.layers.torch import Rearrange

from models import MoViNet, _C

class Classifier(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.movinet = MoViNet(_C.MODEL.MoViNetA0, causal = False, pretrained = False )
    self.one_by_one = nn.Sequential(
      nn.Conv3d(480, 1, kernel_size=(1, 1, 1)), 
      Rearrange('b c t h w -> b (c t) (h w)')
    )
  
  def forward(self, x):
    x = self.movinet(x)
    x = self.one_by_one(x)
    return x