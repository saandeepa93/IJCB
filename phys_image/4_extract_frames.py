import os 
import sys
import argparse
from tqdm import tqdm 
from icecream import ic
from sys import exit as e

import torch 
from torchvision import io
from torch.utils.data import DataLoader

sys.path.append('.')
from loader import VideoLoader, OpenFaceLoader
from phys_image import get_args
from configs.config import get_cfg_defaults
from utils import save_frames

if __name__ == "__main__":  
  args = get_args()
  config_path = os.path.join("./configs/experiments", f"{args.config}.yaml")

   # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()
  print(cfg)

  dataset = VideoLoader(cfg)
  loader = DataLoader(dataset, batch_size=1, shuffle=False)

  for b, (x, ) in enumerate(loader, 0):
    # X_vid = x.permute(0, 3, 1, 2)
    # io.write_video(f"./data/test.mp4", x , fps=2)
    save_frames(x)
    e("done saveing")
  e()