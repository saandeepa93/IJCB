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
from models.flow import Glow

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicCNN(nn.Module):
  def __init__(self, cfg) -> None:
    super().__init__()

    self.conv1 = BasicConv2d(3, 32, 3, 3, 0)
    self.conv2 = BasicConv2d(32, 64, 3, 3, 0)
    self.conv3 = BasicConv2d(64, 128, 3, 3, 0)
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.last_linear = nn.Linear(128, 256, bias=False)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.avg_pool(x)
    x = self.last_linear(x.view(x.shape[0], -1))
    x = F.normalize(x, p=2, dim=1)
    return x

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
    
    self.encoder = InceptionResnetV1(pretrained="vggface2", in_chan=cfg.DATASET.N_CHAN, \
      dropout_prob=0.1, device=device)
    # self.encoder = SupConResNet(name="resnet18")
    # self.encoder = BasicCNN(cfg)

    self.head = nn.Sequential(
                nn.Linear(cfg.MODEL.ENC_DIM, cfg.MODEL.ENC_DIM),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.MODEL.ENC_DIM, cfg.MODEL.FEAT)
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
