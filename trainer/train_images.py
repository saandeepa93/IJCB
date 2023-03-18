import os 
import sys
import argparse
from tqdm import tqdm 
import time
from icecream import ic
from sys import exit as e

import torch 
from torchvision import io
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from einops import rearrange

sys.path.append('.')
from loader import VideoLoader, OpenFaceLoader
from configs.config import get_cfg_defaults
from utils import save_frames, seed_everything, get_args, plot_loader_imgs

from models import Classifier


def prepare_model(cfg):
  model = Classifier(cfg)
  return model


def prepare_dataset(cfg):
  train_dataset = VideoLoader(cfg, "train")
  train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True)
  val_dataset = VideoLoader(cfg, "val")
  val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True)
  return train_loader, val_loader

if __name__ == "__main__":  
  seed_everything(42)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  args = get_args()
  config_path = os.path.join("./configs/experiments", f"{args.config}.yaml")

  # SET TENSORBOARD PATH
  writer = SummaryWriter(f'./runs/{args.config}')

   # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()
  print(cfg)

  train_loader, val_loader = prepare_dataset(cfg)
  model = prepare_model(cfg)

  start = time.time()
  for b, (x, label, subject) in enumerate(tqdm(val_loader), 0):
    x = rearrange(x, 'b t c h w -> b c t h w')
    start = time.time()
    x = model(x)
    ic(f"Total: {time.time() - start}")
    ic(x.size())
    e()
    # plot_loader_imgs(x, label, cfg, b * cfg.TRAINING.BATCH)
    # save_frames(x)