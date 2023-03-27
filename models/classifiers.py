from icecream import ic
from sys import exit as e

import torch 
from torch import nn
from torch.nn import functional as F

from einops.layers.torch import Rearrange

from models.movinets import MoViNet, _C
from models.transformer_model import ViT, ViT_face
from models.inception import InceptionResnetV1
from models.resnet_big import SupConResNet

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
    self.cfg = cfg
  
  def forward(self, x):
    x = self.vit(x)
    if self.cfg.DATASET.AUTH:
      x = F.normalize(x, dim=1)
    else:
      x = F.softmax(x, dim=-1)
    return x 
  
class AuthImage(nn.Module):
  def __init__(self, cfg) -> None:
    super().__init__()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.encoder = InceptionResnetV1(pretrained=None, in_chan=cfg.DATASET.N_CHAN, \
      dropout_prob=0.1, device=device)
    # self.encoder = SupConResNet(name="resnet18")
    self.head = nn.Sequential(
                nn.Linear(cfg.MODEL.ENC_DIM, cfg.MODEL.ENC_DIM),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.MODEL.ENC_DIM, cfg.TRAINING.FEAT)
            )

  def forward(self, x):
    x = self.encoder(x)
    x = self.head(x)
    x = F.normalize(x, dim=1)
    return x

class LinearClassifier(nn.Module):
    def __init__(self, cfg):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(cfg.MODEL.ENC_DIM, cfg.DATASET.N_CLASS)

    def forward(self, features):
        return self.fc(features)
