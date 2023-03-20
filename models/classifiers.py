from icecream import ic
from sys import exit as e

import torch 
from torch import nn
from torch.nn import functional as F

from einops.layers.torch import Rearrange

from models.movinets import MoViNet, _C
from models.transformer_model import ViT, ViT_face

class ClassifierVideo(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.movinet = MoViNet(_C.MODEL.MoViNetA0, causal = False, pretrained = False )
    self.one_by_one = nn.Sequential(
      nn.Conv3d(480, 1, kernel_size=(1, 1, 1)), 
      Rearrange('b c t h w -> b (c t) (h w)')
    )
    self.vit = ViT(cfg)
    self.bn = nn.BatchNorm1d(cfg.TRANSFORMER.DIM_OUT, affine=False)
  
  def forward(self, x):
    x = self.movinet(x)
    x = self.one_by_one(x)
    x = self.vit(x)
    x = F.normalize(x, dim=1)
    return x
  

class ClassifyImage(nn.Module):
  def __init__(self, cfg) -> None:
    super().__init__()

    self.vit = ViT_face(cfg)
  
  def forward(self, x):
    x = self.vit(x)
    return F.softmax(x, dim=-1)