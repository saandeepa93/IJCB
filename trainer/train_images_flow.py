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
from models import AuthImage, LinearClassifier, Glow
from losses import SupConLoss, FlowConLoss

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
  n_bins = 2.0 ** cfg.FLOW.N_BITS
  model_single = Glow(
        cfg.FLOW.N_CHAN, cfg.FLOW.N_FLOW, cfg.FLOW.N_BLOCK, affine=True, conv_lu=True, mlp_dim=cfg.FLOW.MLP_DIM, \
          dropout = cfg.MODEL.DROPOUT
    )
  model = nn.DataParallel(model_single)

  if cfg.DATASET.AUTH:
    criterion = FlowConLoss(cfg, n_bins, device)
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
  avg_con_loss = []
  avg_nll_loss = []
  for b, (x, label, _) in enumerate(loader, 0):
    # x = torch.cat([x[0], x[1]], dim=0)
    x = x.to(device)
    label = label.type(torch.LongTensor)
    label = label.to(device)

    if cfg.FLOW.N_BITS < 8:
      x = torch.floor(x / 2 ** (8 - cfg.FLOW.N_BITS))
    x = x / n_bins - 0.5

    means, log_sds, logdet, features, log_p  = model(x + torch.rand_like(x) / n_bins)
    # LOSS
    nll_loss, log_p, _, log_p_all = criterion.nllLoss(features, logdet, means, log_sds)
    con_loss = criterion.conLoss(log_p_all, label)

    con_loss_mean = con_loss.mean()
    loss = con_loss_mean + (cfg.TRAINING.LMBD * nll_loss)

    with torch.no_grad():
      avg_con_loss += con_loss.tolist()
      avg_nll_loss.append(nll_loss.item())
    
    warmup_learning_rate(cfg, epoch, b, len(loader), optimizer)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  avg_con_loss = sum(avg_con_loss)/len(avg_con_loss)
  avg_nll_loss = sum(avg_nll_loss)/len(avg_nll_loss)

  return avg_con_loss, avg_nll_loss, log_p
  
def validate(loader, epoch, model, criterion, cfg, device):
  model.eval()
  avg_con_loss = []
  avg_nll_loss = []
  with torch.no_grad():
    for b, (x, label, _) in enumerate(loader, 0):
      unique = torch.unique(label, return_counts=True)
      # plot_loader_imgs(x, label, cfg, b)
      x = x.to(device)
      label = label.type(torch.LongTensor)
      label = label.to(device)

      if cfg.FLOW.N_BITS < 8:
        x = torch.floor(x / 2 ** (8 - cfg.FLOW.N_BITS))
      x = x / n_bins - 0.5

      means, log_sds, logdet, features, log_p  = model(x + torch.rand_like(x) / n_bins)

      nll_loss, log_p, log_det, log_p_all = criterion.nllLoss(features, logdet, means, log_sds)
      con_loss = criterion.conLoss(log_p_all, label)
      con_loss_mean = con_loss.mean()
      loss = con_loss_mean + (cfg.TRAINING.LMBD * nll_loss)

      avg_con_loss += con_loss.tolist()
      avg_nll_loss.append(nll_loss.item())

  avg_con_loss = sum(avg_con_loss)/len(avg_con_loss)
  avg_nll_loss = sum(avg_nll_loss)/len(avg_nll_loss)
  return avg_con_loss, avg_nll_loss


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
  e()

  # optimizer = optim.SGD(model.parameters(), lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY, momentum=0.9)
  optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY)
  scheduler = CosineAnnealingLR(optimizer, cfg.LR.T_MAX, cfg.LR.MIN_LR)



  # TRAINING
  n_bins = 2.0 ** cfg.FLOW.N_BITS
  min_loss = 1e5
  pbar = tqdm(range(cfg.TRAINING.ITER))
  for epoch in pbar:
    train_con_loss, train_nll_loss, log_p = train(train_loader, epoch, model, optimizer, criterion, cfg, device)
    val_con_loss, val_nll_loss = validate(val_loader, epoch, model, criterion, cfg, device)
    curr_lr = optimizer.param_groups[0]["lr"] 
    avg_grad = grad_flow(model.named_parameters()).item()
    
    if cfg.LR.ADJUST:
      scheduler.step()
    
    if val_con_loss < min_loss:
      min_loss = val_con_loss
      torch.save(model.state_dict(), f"checkpoint/{args.config}_model_final.pt")

    pbar.set_description(
      f"Train NLL Loss: {round(train_nll_loss, 4):.5f}; Train Con Loss: {round(train_con_loss, 4)};\
        Val NLL Loss: {round(val_nll_loss, 4)}; Val Con Loss: {round(val_con_loss, 4)}\
        logP: {log_p.item():.5f}; lr: {curr_lr:.7f}; Min NLL: {round(min_loss, 3)} "
      )
    writer.add_scalar("Train/Contrastive", round(train_con_loss, 4), epoch)
    writer.add_scalar("Train/NLL", round(train_nll_loss, 4), epoch)
    writer.add_scalar("Val/Contrastive", round(val_con_loss, 4), epoch)
    writer.add_scalar("Val/NLL", round(val_nll_loss, 4), epoch)


