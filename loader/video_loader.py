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

from utils import Iterator



class VideoLoader(Dataset):

  def __init__(self, cfg, mode) -> None:
    super().__init__()
    self.cfg = cfg
    self.root_dir = cfg.PATHS.VID_DIR
    self.ted_dir = cfg.PATHS.TED_DIR
    self.aligned_dir = os.path.join(cfg.PATHS.OPENFACE_DIR, "aligned")
    # self.iterator = Iterator(self.root_dir, ".mp4", cfg.DATASET.CAMERA_VIEW, train=cfg.DATASET.TRAIN)
    self.iterator = Iterator(cfg, self.root_dir, ".mp4", mode)
    self.all_files, self.all_labels = self.iterator.__getallfiles__()
    self.transforms = self.get_augmentation()
    ic(len(self.all_files), len(self.all_labels))

  def get_augmentation(self):
    video_dict = {}
    for i in range(1, self.cfg.DATASET.NUM_FRAMES):
      video_dict[f'image{i}'] = "image"

    transforms = A.Compose([
      A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
      A.Normalize((0, 0, 0), (1, 1, 1), p=1),
      ToTensorV2()
      ], 
      additional_targets=video_dict, p=1)
    
    return transforms
    
  def __len__(self):
    return len(self.all_files)
  
  def __getitem__(self, idx):
    vid_path = self.all_files[idx]
    # GET METADATA
    split_lst = vid_path.split('/')
    subject = split_lst[-3]
    session = split_lst[-2]
    fname = split_lst[-1].split('.')[0]
    # GET PATHS
    ted_path = os.path.join(self.ted_dir, subject, session, f"{fname}.csv")
    aligned_path = os.path.join(self.aligned_dir, subject, session, fname)
    # SELECT EXPRESSIVE FRAMES
    df = pd.read_csv(ted_path)
    sorted_df = df.sort_values(['ted'], ascending=False)
    sorted_df = sorted_df.iloc[::2]
    top_40 = sorted_df.iloc[:15]['frame'].tolist()
    bottom_10 = sorted_df.iloc[-5:]['frame'].tolist()
    frames = sorted(top_40 + bottom_10)
    # AUGMENT
    augment_dict = {}
    for f, frame in enumerate(frames, 0):
      img_path = os.path.join(aligned_path, f"frame_det_00_{str(frame+1).zfill(6)}.bmp")
      im = np.array(Image.open(img_path))
      crop_im = im.copy()
      if f == 0:
        augment_dict["image"] = crop_im
      else:
        augment_dict[f"image{f}"] = crop_im

    transformed_data = self.transforms(**augment_dict)
    image_list = []
    for f in range(0, self.cfg.DATASET.NUM_FRAMES):
      if f == 0:
        im = transformed_data['image']
      else:
        im = transformed_data[f"image{f}"]
      image_list.append(im)
    x_img = torch.stack(image_list)
    # LABELS
    if subject == self.cfg.SPLIT.SUBJECT:
      label = 1
    else: label = 0
    
    return x_img, label, subject





  


