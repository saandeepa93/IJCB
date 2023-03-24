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
from torch.utils.tensorboard import SummaryWriter

from einops import rearrange

sys.path.append('.')
from configs.config import get_cfg_defaults
from utils import seed_everything, get_args, plot_loader_imgs, get_metrics
from loader import ImageLoader
from models import ClassifyImage


def prepare_model(cfg):
  model = ClassifyImage(cfg)
  model = nn.DataParallel(model)
  criterion = nn.CrossEntropyLoss()
  return model, criterion

def prepare_dataset(cfg):
  train_dataset = ImageLoader(cfg, "train")
  val_dataset = ImageLoader(cfg, "val")

  if cfg.TRAINING.SAMPLER:
    all_labels = list(train_dataset.all_files_dict.values())
    unique = np.unique(np.array(all_labels), return_counts=True)
    sample_weight = torch.tensor([0.5 if l==1 else 0.1 for l in all_labels])
    sampler = WeightedRandomSampler(sample_weight.type('torch.DoubleTensor'), len(sample_weight))
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=False, sampler=sampler)
  else:
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True)

  val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True)
  return train_loader, val_loader

def train(loader, epoch, model, optimizer, criterion, cfg, device):
  model.train()
  avg_loss = []
  y_pred_all = []
  y_true_all = []
  for b, (x, label, _) in enumerate(loader, 0):
    
    x = x.to(device)
    label = label.type(torch.LongTensor)
    label = label.to(device)

    out = model(x)
    loss = criterion(out, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
      y_pred_all += torch.argmax(out, dim=-1).tolist()
      y_true_all += label.tolist()
      avg_loss.append(loss.item())

  with torch.no_grad():
    avg_loss = sum(avg_loss)/len(avg_loss)
    acc, f1, mcc, conf_mat = get_metrics(y_true_all, y_pred_all)
  
  return avg_loss, acc, f1, conf_mat
  
def validate(loader, epoch, model, criterion, cfg, device):
  model.eval()
  avg_loss = []
  y_pred_all = []
  y_true_all = []
  with torch.no_grad():
    for b, (x, label, _) in enumerate(loader, 0):
      # plot_loader_imgs(x, label, cfg, b)
      x = x.to(device)
      label = label.type(torch.LongTensor)
      label = label.to(device)

      out = model(x)
      loss = criterion(out, label)

      y_pred_all += torch.argmax(out, dim=-1).tolist()
      y_true_all += label.tolist()
      avg_loss.append(loss.item())
    avg_loss = sum(avg_loss)/len(avg_loss)
    acc, f1, mcc, conf_mat = get_metrics(y_true_all, y_pred_all)
    
  return avg_loss, acc, f1, conf_mat


if __name__ == "__main__":  
  # FIXED SEEDING FOR REPRODUCIBILITY
  seed_everything(42)

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
  cfg.freeze()
  print(cfg)

  # GET MODEL, DATASET ETC
  train_loader, val_loader = prepare_dataset(cfg)
  model, criterion = prepare_model(cfg)
  model = model.to(device)
  optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY)

  # TRAINING
  min_loss = 1e5
  pbar = tqdm(range(cfg.TRAINING.ITER))
  for epoch in pbar:
    avg_train_loss, avg_train_acc, f1_train, conf_mat_train = train(train_loader, epoch, model, optimizer, criterion, cfg, device)
    avg_val_loss, avg_val_acc, f1_val, conf_mat_val = validate(val_loader, epoch, model, criterion, cfg, device)

    pbar.set_description(
                          f"epoch: {epoch}; train_loss: {round(avg_train_loss, 2)}; val_loss: {round(avg_val_loss, 2)}; train_acc: {round(avg_train_acc, 2)}; val_acc: {round(avg_val_acc, 2)}"\
                          # f"epoch: {epoch}; train_loss: {avg_train_loss}; val_loss: {avg_val_loss}; train_f1: {f1_train}; val_f1: {f1_val}"\
                        ) 
    
    writer.add_scalar("Train/Loss", round(avg_train_loss, 2), epoch)
    writer.add_scalar("Train/Acc", round(avg_train_acc, 2), epoch)
    writer.add_scalar("Val/Loss", round(avg_val_loss, 2), epoch)
    writer.add_scalar("Val/Acc", round(avg_val_acc, 2), epoch)



