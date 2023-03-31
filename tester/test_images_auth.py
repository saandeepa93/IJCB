import os 
import sys
import argparse
from tqdm import tqdm 
import time
from icecream import ic
from sys import exit as e
import numpy as np
import pickle

import torch 
from torch import nn, optim
from torchvision import io
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from einops import rearrange

sys.path.append('.')
from configs.config import get_cfg_defaults
from utils import seed_everything, get_args, plot_loader_imgs, grad_flow, mkdir
# from loader import ImageLoader
from models import AuthImage, LinearClassifier, Glow
from losses import SupConLoss

from PIL import Image
import albumentations as A
from  albumentations.pytorch.transforms import ToTensorV2


def prepare_model(cfg):
  model = AuthImage(cfg)
  model = nn.DataParallel(model)
  criterion = SupConLoss(temperature=cfg.MODEL.TEMP)
  return model, criterion

def get_transforms(cfg):
  val_transform = A.Compose([
    A.Resize(cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE, p=1),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1),
    ToTensorV2()
  ])
  return val_transform


def get_data(path, transform):
  x = np.array(Image.open(path))
  x = transform(image=x)['image']
  x = x.to(device)
  return x

if __name__ == "__main__":
   # FIXED SEEDING FOR REPRODUCIBILITY
  seed_everything(42)

  # SET DEVICE
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  args = get_args()
  config_path = os.path.join("./configs/experiments", f"{args.config}.yaml")


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
  checkpoint = torch.load(f"./checkpoint/{args.config}/model_final.pt", map_location=device)
  model.load_state_dict(checkpoint)
  model = model.to(device)
  model.eval()

  transform = get_transforms(cfg)

  img_path_anchor = "/data/dataset/ca_data/openface/aligned/P0002/S2_April_12_2022/face_view_v2/frame_det_00_001519.bmp"
  img_path_pos = "/data/dataset/ca_data/openface/aligned/P0002/S3_April_29_2022/face_view_v1/frame_det_00_001283.bmp"
  # img_path_neg = "/data/dataset/ca_data/openface/aligned/P0026/S1_Aug_15_2022/face_view_v4/frame_det_00_000327.bmp"
  img_path_neg = "/data/dataset/ca_data/openface/aligned/P0004/S3_May_26_2022/face_view_v3/frame_det_00_001039.bmp"
  
  x_anc = get_data(img_path_anchor, transform)
  x_pos = get_data(img_path_pos, transform)
  x_neg = get_data(img_path_neg, transform)

  x_all = torch.stack([x_anc, x_pos, x_neg], dim=0)

  with torch.no_grad():
    f_all = model.module.encoder(x_all)

  f_anc, f_pos, f_neg = torch.unbind(f_all, dim=0)
  ic(f_anc.size(), f_pos.size(), f_neg.size())

  dist_pos = torch.sum(((f_anc - f_pos)**2))
  dist_neg = torch.sum(((f_anc - f_neg)**2))
  ic(dist_pos, dist_neg)
  e()