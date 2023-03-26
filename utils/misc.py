import yaml
import json
import os 
import argparse
from icecream import ic
from sys import exit as e

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

import av
import cv2
import random 
import numpy as np
import torch 
from umap import UMAP
import pandas as pd
import plotly.express as px

def plot_umap(X_lst_un, y_lst, name, fname_all, dim):
  b = X_lst_un.size(0)
  X_lst = UMAP(n_components=dim, random_state=0, init='random').fit_transform(X_lst_un.view(b, -1))
  y_lst_label = [str(i) for i in y_lst.detach().numpy()]

  if dim == 3:
    df = pd.DataFrame(X_lst, columns=["x", "y", "z"])
  else:
    df = pd.DataFrame(X_lst, columns=["x", "y"])
  df_color = pd.DataFrame(y_lst_label, columns=["class"])
  df_fname = pd.DataFrame(fname_all, columns=["fname"])
  df = df.join(df_color)
  df = df.join(df_fname)
  
  if dim == 3:
    fig = px.scatter_3d(df, x='x', y='y', z='z',color='class', title=f"{name}", \
      hover_data=[df.fname])
  else:
    fig = px.scatter(df, x='x', y='y',color='class', title=f"{name}", \
      hover_data=[df.fname])
  
  fig.update_traces(marker=dict(size=6))
  fig.update_layout(legend=dict(
    yanchor="top",
    y=0.60,
    xanchor="left",
    x=0.70
    ))
  # fig.update_traces(hovertemplate = 'fname=%{customdata[0]}<br>')
  fig.write_html(f"./data/vis/umap/{dim}d_{name}.html")


class TwoCropTransform:
  """Create two crops of the same image"""
  def __init__(self, transform):
      self.transform = transform

  def __call__(self, x):
      return [self.transform(image=x)['image'], self.transform(image=x)['image']]


def grad_flow(named_parameters):
  ave_grads = 0
  layers_cnt = 0
  for n, p in named_parameters:
    if p.requires_grad and ("bias" not in n) and p.grad is not None:
      ave_grads += p.grad.abs().mean()
      layers_cnt += 1
  ave_grads /= layers_cnt
  return ave_grads

def save_frames(vid):
  save_path = "/home/saandeepaath/Desktop/projects/ca_proj/temp"
  vid = vid.squeeze(0).permute(0, 2, 3, 1).numpy()
  ctr = 0
  for frame in vid:
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = cv2.normalize(img, None, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path, f"{ctr}.png"), img)
    ctr +=1 

def read_yaml():
  with open("./phys_image/variables.yaml", "r") as stream:
    try:
        return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def mkdir(path):
  if not os.path.isdir(path):
    os.mkdir(path)

def read_json(path):
  with open(path, 'r') as fp:
    json_file = json.load(fp)
  return json_file


def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

def get_args():
  parser = argparse.ArgumentParser(description="Vision Transformers")
  parser.add_argument('--config', type=str, default='default', help='configuration to load')
  args = parser.parse_args()
  return args

def get_metrics(y_true, y_pred):
  acc = accuracy_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred, average=None)
  f1 = [round(i, 2) for i in f1]
  mcc = matthews_corrcoef(y_true, y_pred)
  conf = confusion_matrix(y_true, y_pred)
  return round(acc, 3), f1, round(mcc, 2), conf

def plot_loader_vid(arr, exp, cfg, ctr):
  arr = arr.permute(0, 1, 3, 4, 2)
  arr = arr.detach().cpu().numpy()
  b, t, h, w, c = arr.shape
  for i in range(b):
    img = arr[i, 0]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (img*255).astype(np.uint8)

    label = exp[i].item()
    img = cv2.putText(img, str(label), (int(h-100), int(w-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 2)
    cv2.imwrite(os.path.join(cfg.PATHS.VIS_PATH, f"img_{ctr}.png"), img)
    ctr+=1

def plot_loader_imgs(arr, exp, cfg, ctr):
  arr = arr.permute(0, 2, 3, 1)
  arr = arr.detach().cpu().numpy()
  b, h, w, c = arr.shape
  for i in range(b):
    img = arr[i]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (img*255).astype(np.uint8)

    label = exp[i].item()
    img = cv2.putText(img, str(label), (int(h-100), int(w-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 2)
    cv2.imwrite(os.path.join(cfg.PATHS.VIS_PATH, f"img_{ctr}.png"), img)
    ctr+=1