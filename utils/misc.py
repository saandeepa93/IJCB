import yaml
import json
import os 
import argparse
from icecream import ic
from sys import exit as e

import av
import cv2
import random 
import numpy as np
import torch 

class Iterator:
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


def save_frames(vid):
  save_path = "/home/saandeepaath/Desktop/projects/ca_proj/temp"
  vid = vid.squeeze(0).permute(0, 2, 3, 1).numpy()
  ctr = 0
  for frame in vid:
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = cv2.normalize(img, None, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path, f"{ctr}.png"), img)
    ctr +=1 

def read_yaml():
  with open("./phys_image/variables.yaml", "r") as stream:
    try:
        return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def mkdir(path):
  if not os.path.isdir(path):
    os.mkdir(path)

def read_json(path):
  with open(path, 'r') as fp:
    json_file = json.load(fp)
  return json_file


def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

def get_args():
  parser = argparse.ArgumentParser(description="Vision Transformers")
  parser.add_argument('--config', type=str, default='default', help='configuration to load')
  args = parser.parse_args()
  return args

def plot_loader_imgs(arr, exp, cfg, ctr):
  arr = arr.permute(0, 1, 3, 4, 2)
  arr = arr.detach().cpu().numpy()
  b, t, h, w, c = arr.shape
  for i in range(b):
    img = arr[i, 0]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (img*255).astype(np.uint8)

    label = exp[i].item()
    img = cv2.putText(img, str(label), (int(h-100), int(w-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 2)
    ic(os.path.join(cfg.PATHS.VIS_PATH, f"img_{ctr}.png"))
    cv2.imwrite(os.path.join(cfg.PATHS.VIS_PATH, f"img_{ctr}.png"), img)
    ctr+=1