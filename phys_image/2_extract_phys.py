import os 
from sys import exit as e
from icecream import ic
from tqdm import tqdm
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import math
import json
import yaml
import datetime

from utils import read_yaml, read_json, mkdir

add_seconds = lambda x, s: x + datetime.timedelta(0, s)

def process_acc(path, cols):
  df = pd.read_csv(path, header=None, names = cols)
  start_time_ms = df.iloc[0][cols[0]]
  start_time = datetime.datetime.fromtimestamp(start_time_ms)
  sampling_freq = int(df.iloc[1][cols[0]].item())

  data_df = df.iloc[2:].copy()
  data_df['ts'] = start_time

  ctr = 0
  for i in range(0, data_df.shape[0], sampling_freq):
    data_df.loc[i:i+sampling_freq,'ts'] = add_seconds(start_time, ctr)
    ctr+=1

  return data_df


if __name__ == "__main__":
  vars = read_yaml()

  # GET TOUCH DATA
  touch_ts_path = vars['json']['touch_ts']
  touch_ts_all = read_json(touch_ts_path)

  # VALID AND INVALID SUBJECTS
  valid_subjects = [key for key in touch_ts_all.keys() if touch_ts_all[key]]
  error_data = vars['error']
  
  # PATHS
  phys_root = vars['paths']['phys']
  dest_root = vars['paths']['phys_affect_split']
  
  # PHYS FILES AND COLUMNS
  acc_cols = ['x', 'y', 'z']
  cols = ['value']
  acc_file = "ACC.csv"
  bvp_file = "BVP.csv"
  eda_file = "EDA.csv"
  hr_file = "HR.csv"
  temp_file = "TEMP.csv"
  # ibi_file = "IBI.csv"

  pbar = tqdm(valid_subjects)
  for sub in pbar:
    sub = "P0039"
    sub_dir = os.path.join(phys_root, sub)
    if not os.path.isdir(sub_dir):
      continue
    dest_sub_dir = os.path.join(dest_root, sub)
    mkdir(dest_sub_dir)

    for sess in touch_ts_all[sub]:
      sess_num = sess.split('_')[0]
      sess_name = '_'.join(sess.split('_')[1:])
      sess_path = os.path.join(sub_dir, sess_name)
      if not os.path.isdir(sess_path):
        continue
      if sub in error_data:
        if sess_num in error_data[sub]:
          continue
      dest_sess_dir = os.path.join(dest_sub_dir, sess)
      mkdir(dest_sess_dir)

      pbar.set_description(f"{sub}|{sess_num}")
      touch_ts = touch_ts_all[sub][sess]['affect']
      for vid in touch_ts:
        # CREATE DEST
        dest_vid_dir = os.path.join(dest_sess_dir, vid)
        mkdir(dest_vid_dir)

        # GET START AND END TIME
        start_time = touch_ts[vid]['start']
        end_time = touch_ts[vid]['end']
        # ACC DATA
        acc_path = os.path.join(sess_path, acc_file)
        acc_df = process_acc(acc_path, acc_cols)
        acc_vid_df = acc_df[(acc_df['ts']>start_time) & (acc_df['ts']<end_time)]
        acc_vid_df.to_csv(os.path.join(dest_vid_dir, acc_file))
        
        # BVP DATA
        bvp_path = os.path.join(sess_path, bvp_file)
        bvp_df = process_acc(bvp_path, cols)
        bvp_vid_df = bvp_df[(bvp_df['ts']>start_time) & (bvp_df['ts']<end_time)]
        bvp_vid_df.to_csv(os.path.join(dest_vid_dir, bvp_file))
        
        # EDA DATA
        eda_path = os.path.join(sess_path, eda_file)
        eda_df = process_acc(eda_path, cols)
        eda_vid_df = eda_df[(eda_df['ts']>start_time) & (eda_df['ts']<end_time)]
        eda_vid_df.to_csv(os.path.join(dest_vid_dir, eda_file))
        
        # HR DATA
        hr_path = os.path.join(sess_path, hr_file)
        hr_df = process_acc(hr_path, cols)
        hr_vid_df = hr_df[(hr_df['ts']>start_time) & (hr_df['ts']<end_time)]
        hr_vid_df.to_csv(os.path.join(dest_vid_dir, hr_file))
        
        # TEMP DATA
        temp_path = os.path.join(sess_path, temp_file)
        temp_df = process_acc(temp_path, cols)
        temp_vid_df = temp_df[(temp_df['ts']>start_time) & (temp_df['ts']<end_time)]
        temp_vid_df.to_csv(os.path.join(dest_vid_dir, temp_file))
