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
from utils import TwoCropTransform

class ImageLoader(Dataset):
  def __init__(self, cfg, mode, aug=False) -> None:
    super().__init__()
    self.cfg = cfg
    self.mode = mode
    self.aug = aug
    self.ted_dir = cfg.PATHS.TED_DIR
    self.root_dir = os.path.join(cfg.PATHS.OPENFACE_DIR, "aligned")
    self.iterator = ImageIterator(cfg, self.root_dir, ".bmp", mode)
    self.all_files_dict, self.classwise_dict = self.iterator.__getallfiles__()
    self.all_files = list(self.all_files_dict.keys())
    self.train_transforms_single, self.val_transforms = self.get_augmentation()
    self.train_transforms = TwoCropTransform(self.train_transforms_single)
    ic(len(self.all_files_dict))
    ic(self.classwise_dict)

  
  def get_augmentation(self):
    transforms = A.Compose([
      A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
      A.Normalize((0, 0, 0), (1, 1, 1), p=1),
      ToTensorV2()
      ])
    
    train_transform = A.Compose([
        A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1),
        A.HorizontalFlip(p = 0.5),
        A.ColorJitter(0.4, 0.4, 0.4, 0.1),
        A.ToGray(p=0.2),
        ToTensorV2(),
      ])

    val_transform = transforms = A.Compose([
      A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1),
      ToTensorV2()
      ])
    
    return train_transform, val_transform
    
  def __len__(self):
    return len(self.all_files)
  
  def __getitem__(self, idx):
    aligned_path = self.all_files[idx]
    x_img = np.array(Image.open(aligned_path))
    
    if self.aug and self.mode == "train":
      x_img = self.train_transforms_single(image=x_img)['image']
    if not self.aug and self.mode == "train":
      x_img = self.val_transforms(image=x_img)['image']
        # if not self.cfg.DATASET.AUTH:
        #   x_img = self.train_transforms_single(image=x_img)['image']
        # else:
        #   x_img = self.train_transforms(x_img)
    elif self.mode == "val":
      x_img = self.val_transforms(image=x_img)['image']

    
    label = self.all_files_dict[aligned_path]
    subject = aligned_path.split('/')[-4]
    sess = aligned_path.split('/')[-3].split('_')[0]
    view = aligned_path.split('/')[-2].split('_')[-1]
    fname = aligned_path.split('/')[-1].split('.')[0]


    
    return x_img, label, f"{subject}_{sess}_{view}_{fname}"





  

