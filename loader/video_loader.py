import torch 
from torch import nn 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from decord import VideoReader, AudioReader
from decord import cpu, gpu

import os
from sys import exit as e
from icecream import ic

from utils import Iterator

class VideoLoader(Dataset):
  def __init__(self, cfg) -> None:
    super().__init__()
    self.cfg = cfg
    self.root_dir = cfg.PATHS.VID_DIR
    self.iterator = Iterator(self.root_dir, ".mp4", cfg.DATASET.CAMERA_VIEW)
    self.all_files = self.iterator.__getallfiles__()
    
  def __len__(self):
    return len(self.all_files)


  


