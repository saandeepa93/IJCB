import os 
import sys
import argparse
from tqdm import tqdm 
from icecream import ic
from sys import exit as e

import torch 
from torch.utils.data import DataLoader

sys.path.append('.')
from loader import VideoLoader, OpenFaceLoader
from phys_image import get_args
from configs.config import get_cfg_defaults
from utils import calculate_ted_score, mkdir, plot_ted

if __name__ == "__main__":  
  args = get_args()
  config_path = os.path.join("./configs/experiments", f"{args.config}.yaml")

   # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()
  print(cfg)

  dataset = VideoLoader(cfg)
  e()