from icecream import ic
from sys import exit as e
import os
import pandas as pd
import numpy as np
import random


class ImageIterator:
  # def __init__(self, root, ext, camera_view, train=False, mode="train"):
  def __init__(self, cfg, root, ext, mode="all"):
    self.root_dir = root
    self.ext = ext
    self.mode = mode
    self.cfg = cfg
    self.camera = cfg.DATASET.CAMERA_VIEW
    self.train = cfg.SPLIT.SUBJECT
    self.val_split = cfg.SPLIT.VAL_SPLIT
    self.split_value = cfg.SPLIT.SPLIT_VALUE
    self.affect_frames = int(cfg.DATASET.TED_SPLIT[0])
    self.non_affect_frames = int(cfg.DATASET.TED_SPLIT[1])

    
  def __getallfiles__(self):
    all_sub_dict = {}
    classwise_dict = {0:0, 1:0}
    ctr = 0
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

        # SESSION WISE SPLIT
        if self.mode != "all":
          if self.val_split == "sess":
            if self.mode == "val":
              if self.split_value not in sess_name:
                continue
            else:
              if self.split_value in sess_name:
                continue
        # CAMERA
        for entry3 in os.scandir(sess_dir):
          if not entry3.is_dir():
            continue
          if self.camera not in entry3.name:
            continue
          cam_dir = entry3.path
          cam_name = entry3.name
          
          # HANDLE TED FRAMES
          openface_path = os.path.join(self.cfg.PATHS.OPENFACE_DIR, "csv", sub_name, sess_name, f"{cam_name}.csv")
          df_of = pd.read_csv(openface_path,  encoding = "ISO-8859-1")
          valid_ind = df_of[df_of['success'] == 1]['frame'].tolist()
          valid_ind = [ind-1 for ind in valid_ind]
          
          # ic(sub_name, sess_name, cam_name, len(valid_ind))

          # SELECT VALID OPENFACE RECORD
          ted_path = os.path.join(self.cfg.PATHS.TED_DIR, sub_name, sess_name, f"{cam_name}.csv")
          df = pd.read_csv(ted_path)
          df = df[df['frame'].isin(valid_ind)]
          sorted_df = df.sort_values(['ted'], ascending=False)
          sorted_df = sorted_df.iloc[::2]
          valid_ted_ind = sorted_df['frame'].tolist()
          top_40 = sorted_df.iloc[:self.affect_frames]['frame'].tolist()
          bottom_10 = sorted_df.iloc[-self.non_affect_frames:]['frame'].tolist()
          valid_ind = valid_ind[::2]

          # TED FOR TRAIN; RANDOM FOR VAL
          if self.mode == "train":
            if self.train == sub_name:
              frames = valid_ind[:1000]
            else:
              frames = sorted(top_40 + bottom_10) if self.cfg.DATASET.TRAIN_HIGH \
                else sorted(list(random.sample(valid_ind, self.affect_frames + self.non_affect_frames)))
          elif self.mode == "val":
            if self.train == sub_name:
              frames = valid_ind[:1000]
            else:
              frames =  sorted(top_40 + bottom_10) if self.cfg.DATASET.VAL_HIGH \
                else sorted(list(random.sample(valid_ind, self.affect_frames + self.non_affect_frames)))
            
          # FRAME LEVEL
          for frame in frames:
            frame_path = os.path.join(cam_dir, f"frame_det_00_{str(frame+1).zfill(6)}.bmp")
            # all_sub_dict[frame_path] = ctr
            

            # LABELLING FOR AUTHENTICATION VS IDENTIFICATION
            if self.cfg.DATASET.AUTH:
              if self.mode == "train":
                if classwise_dict[int(sub_name == self.train)] >= self.cfg.SPLIT.COUNT:
                  continue
              all_sub_dict[frame_path] = int(sub_name == self.train)
              classwise_dict[int(sub_name == self.train)] +=1
              # if sub_name == self.train:
                #   all_sub_dict[frame_path] = 1
                #   classwise_dict[1] +=1
                # else:
                #   all_sub_dict[frame_path] = 0
                #   classwise_dict[0] +=1
            else:
              all_sub_dict[frame_path] = ctr

      ctr+=1
    return all_sub_dict, classwise_dict

class VideoIterator:
  # def __init__(self, root, ext, camera_view, train=False, mode="train"):
  def __init__(self, cfg, root, ext, mode="all"):
    self.root_dir = root
    self.ext = ext
    self.mode = mode
    self.camera = cfg.DATASET.CAMERA_VIEW
    self.train = cfg.SPLIT.SUBJECT
    self.val_split = cfg.SPLIT.VAL_SPLIT
    self.split_value = cfg.SPLIT.SPLIT_VALUE
    
  def __getallfiles__(self):
    all_files = []
    all_labels = []
    all_sub_dict = {}
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

        # SESSION WISE SPLIT
        if self.mode != "all":
          if self.val_split == "sess":
            if self.mode == "val":
              if self.split_value not in sess_name:
                continue
            else:
              if self.split_value in sess_name:
                continue
        # FILES
        for entry3 in os.scandir(sess_dir):
          if os.path.splitext(entry3.name)[-1] != self.ext:
            continue
          if self.camera not in entry3.name:
            continue
          if self.train:
            if sub_name == self.train:
              all_labels.append(1)
            else:
              all_labels.append(0)

          fpath = entry3.path
          fname = entry3.name
          all_files.append(fpath)
    return all_files, all_labels