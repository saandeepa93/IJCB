import os 
import sys
import argparse
from tqdm import tqdm 
import time
from icecream import ic
from sys import exit as e

import torch 
from torch import nn, optim
from torchvision import io
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from einops import rearrange

sys.path.append('.')
from configs.config import get_cfg_defaults
from utils import save_frames, seed_everything, get_args, plot_loader_imgs, get_metrics

from loader import VideoLoader, OpenFaceLoader
from models import Classifier
from losses import SupConLoss


def prepare_model(cfg):
  model = Classifier(cfg)
  criterion = SupConLoss()
  return model, criterion

def prepare_dataset(cfg):
  train_dataset = VideoLoader(cfg, "train")
  train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True)
  val_dataset = VideoLoader(cfg, "val")
  val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True)
  return train_loader, val_loader

def train(cfg, loader, model, optimizer, criterion):
  avg_loss = []
  avg_acc = []
  model.train()
  for b, (x, label, subject) in enumerate(tqdm(val_loader), 0):
    x = x.to(device)
    label = label.to(device)

    x = rearrange(x, 'b t c h w -> b c t h w')
    out = model(x)
    ic(out.size())
    e()

def validate(cfg, loader, model, criterion):
  pass

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
  model, criterion = prepare_model(cfg)
  model = model.to(device)

  optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY)

  pbar = tqdm(range(cfg.TRAINING.ITER))
  for epoch in pbar:
    train(cfg, train_loader, model, optimizer, criterion)

    # plot_loader_imgs(x, label, cfg, b * cfg.TRAINING.BATCH)
    # save_frames(x)