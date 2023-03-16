
import os 
import sys
import argparse
from tqdm import tqdm 
from icecream import ic
from sys import exit as e

import torch 
from torch.utils.data import DataLoader

sys.path.append('.')
from loader import OpenFaceLoader
from phys_image import get_args
from configs.config import get_cfg_defaults
from utils import calculate_ted_score, mkdir, plot_ted


def my_collate(batch):
  return batch

if __name__ == "__main__":  
  args = get_args()
  config_path = os.path.join("./configs/experiments", f"{args.config}.yaml")

   # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()
  print(cfg)

  dataset = OpenFaceLoader(cfg)
  loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn= my_collate)

  for b, (openface_data, ) in enumerate(tqdm(loader), 0):
    meta, gaze_loc, gaze_rot, pose_loc, pose_rot, landmarks, au_inten_OFPAU, fpath = openface_data
    ted_score = calculate_ted_score(au_inten_OFPAU.values, gaze_loc.values, \
                                    gaze_rot.values, pose_loc.values, pose_rot.values,\
                                        landmarks.values, cfg.TED.WINDOW)
    subject = fpath.split("/")[-3]
    session = fpath.split("/")[-2]
    fname = fpath.split('/')[-1]

    dest_sub_dir = os.path.join(cfg.PATHS.TED_DIR, subject)
    mkdir(dest_sub_dir)
    dest_sess_dir = os.path.join(dest_sub_dir, session)
    mkdir(dest_sess_dir)
    dest_file_path = os.path.join(dest_sess_dir, fname)

    ted_score.to_csv(dest_file_path)
    plot_ted(ted_score, os.path.join(dest_sess_dir, f"{fname.split('.')[0]}.html"))
    ic(f"{subject}|{session}|{fname}")
  