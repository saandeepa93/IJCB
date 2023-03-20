import torch 
from torch import nn 
from torch.utils.data import Dataset
from torchvision import transforms, io
from PIL import Image

from decord import VideoReader, AudioReader, AVReader
from decord import cpu, gpu

import albumentations as A
from  albumentations.pytorch.transforms import ToTensorV2

import os
import numpy as np
import pandas as pd
import mediapipe as mp
from sys import exit as e
from icecream import ic

from loader.base_loader import ImageIterator

class ImageLoader(Dataset):
  def __init__(self, cfg, mode) -> None:
    super().__init__()
    self.cfg = cfg
    self.mode = mode
    self.ted_dir = cfg.PATHS.TED_DIR
    self.root_dir = os.path.join(cfg.PATHS.OPENFACE_DIR, "aligned")
    self.iterator = ImageIterator(cfg, self.root_dir, ".bmp", mode)
    self.all_files_dict = self.iterator.__getallfiles__()
    self.all_files = list(self.all_files_dict.keys())
    self.transforms = self.get_augmentation()
    ic(len(self.all_files_dict))

  def get_augmentation(self):
    transforms = A.Compose([
      A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
      A.Normalize((0, 0, 0), (1, 1, 1), p=1),
      ToTensorV2()
      ])
    
    return transforms
    
  def __len__(self):
    return len(self.all_files)
  
  def __getitem__(self, idx):
    aligned_path = self.all_files[idx]
    x_img = np.array(Image.open(aligned_path))
    
    if self.mode == "train":
      x_img = self.transforms(image=x_img)['image']
    elif self.mode == "val":
      x_img = self.transforms(image=x_img)['image']
    
    label = self.all_files_dict[aligned_path]
    subject = aligned_path.split('/')[-4]

    
    return x_img, label, subject





  

