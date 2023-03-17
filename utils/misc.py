import yaml
import json
import os 
import argparse

import random 
import numpy as np
import torch 

class Iterator:
  def __init__(self, root, ext, camera_view):
    self.root_dir = root
    self.ext = ext
    self.camera = camera_view
    
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
          if os.path.splitext(entry3.name)[-1] != self.ext:
            continue
          if self.camera not in entry3.name:
            continue
          fpath = entry3.path
          fname = entry3.name
          all_files.append(fpath)
    return all_files


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