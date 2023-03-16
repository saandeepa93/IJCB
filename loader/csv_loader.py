import torch 
from torch import nn 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import pandas as pd

import os
from sys import exit as e
from icecream import ic

from phys_image import read_json


class OpenFaceLoader(Dataset):

  def __init__(self, cfg) -> None:
    super().__init__()

    self.cfg = cfg
    self.root_dir = os.path.join(self.cfg.PATHS.OPENFACE_DIR, "csv")
    self.all_files = []
    self.__getallfiles__()
    ic(len(self.all_files))

    au_int = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
    self.au_int_cols = [f"AU{str(AU).zfill(2)}_r" for AU in au_int]

  def __getallfiles__(self):
    all_files = []
    # SUBJECTS
    for entry in os.scandir(self.root_dir):
      if not entry.is_dir():
        continue
      sub_dir = entry.path
      sub_name = entry.name

      # SESSION
      for entry2 in os.scandir(sub_dir):
        if not entry2.is_dir():
          continue
        sess_dir = entry2.path
        sess_name = entry2.name

        # FILES
        for entry3 in os.scandir(sess_dir):
          if self.cfg.DATASET.CAMERA_VIEW not in entry3.name:
            continue
          fpath = entry3.path
          fname = entry3.name
          self.all_files.append(fpath)

  def get_target_mat(self, df, target_AUs):
    """
    Return processed OpenFace file 
    """
    col_names = df.columns.tolist()
    
    # create a new dataframe 
    # TODO - take care of the naming issue 
    meta = df.iloc[:, :col_names.index('success')+1]
    gaze_loc = df.iloc[:, col_names.index('gaze_0_x'):col_names.index('gaze_1_z')+1]
    gaze_rot = df.iloc[:, col_names.index('gaze_angle_x'):col_names.index('gaze_angle_y')+1]
    pose_loc = df.iloc[:, col_names.index('pose_Tx'):col_names.index('pose_Tz')+1]
    pose_rot = df.iloc[:, col_names.index('pose_Rx'):col_names.index('pose_Rz')+1]
    landmarks = df.iloc[:, col_names.index('x_0'):col_names.index('y_67')+1]
    au_inten_OFPAU = df[target_AUs]

    return meta, gaze_loc, gaze_rot, pose_loc, pose_rot, landmarks, au_inten_OFPAU

  def __len__(self):
    return len(self.all_files)
  
  def __getitem__(self, idx):
    fpath = self.all_files[idx]
    of_df = pd.read_csv(fpath, encoding = "ISO-8859-1")
    meta, gaze_loc, gaze_rot, pose_loc, pose_rot, landmarks, au_inten_OFPAU = self.get_target_mat(of_df, self.au_int_cols) 
    return meta, gaze_loc, gaze_rot, pose_loc, pose_rot, landmarks, au_inten_OFPAU, fpath

    


