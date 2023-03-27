import os 
import sys
import argparse
from tqdm import tqdm 
import time
from icecream import ic
from sys import exit as e
import numpy as np

import torch 
from torch import nn, optim
from torchvision import io
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from einops import rearrange

sys.path.append('.')
from configs.config import get_cfg_defaults
from utils import seed_everything, get_args, plot_loader_imgs, grad_flow, plot_umap, mkdir
from loader import ImageLoader
from models import AuthImage
from losses import SupConLoss
from train_images_auth import prepare_model, prepare_dataset


def save_embedding(args, loader, model, mode, device):
  with torch.no_grad():
    all_features = []
    all_fname = []
    all_label = []
    for b, (x, label, fname) in enumerate(loader, 0):
      x = x.to(device)
      label = label.to(device)
      features = model.module.encoder(x)

      all_features.append(features)
      all_fname += list(fname)
      all_label.append(label)

    all_features = torch.cat(all_features, dim=0)
    all_label = torch.cat(all_label, dim=0)
    
    plot_umap(all_features.cpu(), all_label.cpu(), f"{args.config}", all_fname, 2, mode)
    plot_umap(all_features.cpu(), all_label.cpu(), f"{args.config}", all_fname, 3, mode)

if __name__ == "__main__":
  # SET DEVICE
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
  cfg.DATASET.W_SAMPLER = False
  cfg.DATASET.NUM_WORKERS = 1
  cfg.TRAINING.BATCH = 32
  cfg.freeze()
  print(cfg)

  # GET MODEL, DATASET ETC
  model, criterion = prepare_model(cfg)
  checkpoint = torch.load(f"./checkpoint/{args.config}_model_final.pt", map_location=device)
  model.load_state_dict(checkpoint)
  model = model.to(device)
  model.eval()

  train_loader, val_loader = prepare_dataset(cfg, False)
  print("Saving Train dataset...")
  save_embedding(args, train_loader, model, "train", device)
  print("Saving Val dataset...")
  save_embedding(args, val_loader, model, "val", device)
  
