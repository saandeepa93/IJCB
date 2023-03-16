import torch 
from torch import nn 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os
from sys import exit as e
from icecream import ic


class VideoLoader(Dataset):
  def __init__(self, cfg, dtype) -> None:
    super().__init__()

    self.cfg = cfg

