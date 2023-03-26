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
from utils import seed_everything, get_args, plot_loader_imgs, grad_flow
from loader import ImageLoader
from models import AuthImage
from losses import SupConLoss

def warmup_learning_rate(cfg, epoch, batch_id, total_batches, optimizer):
  warm_epochs= cfg.LR.WARM_ITER
  warmup_from = cfg.LR.WARMUP_FROM
  warmup_to = cfg.TRAINING.LR
  if cfg.LR.WARM and epoch <= warm_epochs:
    p = (batch_id + (epoch - 1) * total_batches) / \
        (warm_epochs * total_batches)
    lr = warmup_from + p * (warmup_to - warmup_from)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def prepare_model(cfg):
  model = AuthImage(cfg)
  model = nn.DataParallel(model)
  if cfg.DATASET.AUTH:
    criterion = SupConLoss()
  else:
    criterion = nn.CrossEntropyLoss()
  return model, criterion

def prepare_dataset(cfg, aug):
  train_dataset = ImageLoader(cfg, "train", aug)
  val_dataset = ImageLoader(cfg, "val", False)

  if cfg.TRAINING.SAMPLER:
    all_labels = list(train_dataset.all_files_dict.values())
    unique = np.unique(np.array(all_labels), return_counts=True)
    sample_weight = torch.tensor([0.4 if l==1 else 0.1 for l in all_labels])
    sampler = WeightedRandomSampler(sample_weight.type('torch.DoubleTensor'), len(sample_weight))
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=False, sampler=sampler)
  else:
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True)

  val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True)
  return train_loader, val_loader

def train(loader, epoch, model, optimizer, criterion, cfg, device):
  model.train()
  avg_loss = []
  for b, (x, label, _) in enumerate(loader, 0):
    # x = torch.cat([x[0], x[1]], dim=0)
    x = x.to(device)
    label = label.type(torch.LongTensor)
    label = label.to(device)
    
    bsz = label.size(0)
    features = model(x)


    # f1, f2 = torch.split(features, [bsz, bsz], dim=0)
    # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    loss = criterion(features, label)

    with torch.no_grad():
      avg_loss.append(loss.item())

    warmup_learning_rate(cfg, epoch, b, len(loader), optimizer)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  avg_loss = sum(avg_loss)/len(avg_loss)
  return avg_loss
  
def validate(loader, epoch, model, criterion, cfg, device):
  model.eval()
  avg_loss = []
  with torch.no_grad():
    for b, (x, label, _) in enumerate(loader, 0):
      unique = torch.unique(label, return_counts=True)
      # plot_loader_imgs(x, label, cfg, b)
      x = x.to(device)
      label = label.type(torch.LongTensor)
      label = label.to(device)

      out = model(x)
      loss = criterion(out, label)
      avg_loss.append(loss.item())

  avg_loss = sum(avg_loss)/len(avg_loss)
  return avg_loss


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
  model, criterion = prepare_model(cfg)
  model = model.to(device)
  print("Total Trainable Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
  train_loader, val_loader = prepare_dataset(cfg, True)

  optimizer = optim.SGD(model.parameters(), lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY)
  scheduler = CosineAnnealingLR(optimizer, cfg.LR.T_MAX, cfg.LR.MIN_LR)

  # TRAINING
  min_loss = 1e5
  pbar = tqdm(range(cfg.TRAINING.ITER))
  for epoch in pbar:
    avg_train_loss = train(train_loader, epoch, model, optimizer, criterion, cfg, device)
    # avg_val_loss = validate(val_loader, epoch, model, criterion, cfg, device)
    curr_lr = optimizer.param_groups[0]["lr"] 
    avg_grad = grad_flow(model.named_parameters()).item()
    
    if cfg.LR.ADJUST:
      scheduler.step()
    
    if avg_train_loss < min_loss:
      min_loss = avg_train_loss
      torch.save(model.state_dict(), f"checkpoint/{args.config}_model_final.pt")

    pbar.set_description(
                          f"epoch: {epoch}; train_loss: {round(avg_train_loss, 4)}; LR: {round(curr_lr, 4)}; avg_grad: {avg_grad}; min_loss: {round(min_loss, 4)}"
                        ) 
    writer.add_scalar("Train/Loss", round(avg_train_loss, 4), epoch)
    writer.add_scalar("avg_grad", round(avg_grad, 5), epoch)
    writer.add_scalar("Train/Min_Loss", round(min_loss, 5), epoch)
    # writer.add_scalar("Val/Loss", round(avg_val_loss, 2), epoch)



