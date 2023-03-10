import argparse
from tqdm import tqdm
import os
from tqdm import tqdm 
import json
import matplotlib.pyplot as plt
import yaml

from datetime import timedelta, datetime
import datetime


import librosa
import numpy as np
from scipy import signal

from icecream import ic
from sys import exit as e


def find_offset(within_file, find_file, window):
  y_within, sr_within = librosa.load(within_file, sr=None)
  y_find, _ = librosa.load(find_file, sr=sr_within)

  c = signal.correlate(y_within, y_find[:sr_within*window], mode='valid', method='fft')
  peak = np.argmax(c)
  offset = round(peak / sr_within, 2)

  str_dt = str(datetime.timedelta(seconds=int(offset)))
  return datetime.datetime.strptime(str_dt, "%H:%M:%S")

def read_yaml():
  with open("./variables.yaml", "r") as stream:
    try:
        return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


if __name__ == "__main__":
  vid_dir = "/data/dataset/ca_data/merged_compressed_audio"
  affect_audio_dir = "/data/dataset/ca_data/affect_audio"
  
  affect_audio_child_dir = "/data/dataset/ca_data/affect_audio_children"

  sess_duration = [
    ["00:01:24", "00:01:00", "00:00:58", "00:00:19"],
    ["00:01:07", "00:01:01", "00:01:06", "00:00:49"],
    ["00:00:59", "00:00:58", "00:00:59", "00:00:57"]
  ]
  sess_duration = [
    [84, 60, 58, 19],
    [67, 61, 66, 49],
    [59, 58, 59, 57]
  ]

  # completed_subs = ["P0002", "P0003", "P0004", "P0005", "P0006", "P0007", "P0008", "P0009", "P0010", "P0011", "P0012"]
  # completed_subs = ["P0017", "P0021"]
  # completed_subs = ["P0022"]
  completed_subs = []
  # READ VARIABLES
  vars = read_yaml()
  children = vars['children']

  timestamp_csv = {"fname":[], "session_num":[], "session_dt":[], "vid1":[], "vid2":[], "vid3":[], "vid4":[], "annot":[]}
  timestamp_dict = {}

  pbar = tqdm(sorted(os.listdir(vid_dir)))
  for sub in pbar:
    sub = "P0016"
    sub_dir = os.path.join(vid_dir, sub)
    if not os.path.isdir(sub_dir):
      continue
    
    if sub in completed_subs:
      continue
    timestamp_dict[sub] = {}

    for session in os.listdir(sub_dir):
      session = "S3_August_29_2022"
      session_dir = os.path.join(sub_dir, session)
      if not os.path.isdir(session_dir):
        continue
      
      sess_num = int(session.split("_")[0][1])
      sess_date_lst = session.split("_")[1:]
      sess_date = "_".join(sess_date_lst)

      timestamp_dict[sub][f"S{sess_num}"] = {}
      timestamp_dict[sub][f"S{sess_num}"]["date"] = sess_date

      for aud in os.listdir(session_dir):
        aud_path = os.path.join(session_dir, aud)
        if os.path.splitext(aud_path)[-1] != ".wav":
          continue
        if aud == "desktop_view.wav":
          continue
        ic(aud_path)

        aud_name = aud.split(".")[0]
        timestamp_dict[sub][f"S{sess_num}"][aud_name] = {}
        offset_lst = []
        for i in range(4):
          affect_audio_path = os.path.join(affect_audio_dir, f"S{sess_num}_v{i+1}.wav")
          ic(affect_audio_path)

          
          try:
            start = find_offset(aud_path, affect_audio_path, window=10)
          except:
            print(f"Could not process for {sub}, {session}, {aud_name}, {i}")
            continue
          duration = sess_duration[sess_num-1][i]
          end = start + datetime.timedelta(seconds=duration)
          timestamps = [start.strftime("%H:%M:%S"), end.strftime("%H:%M:%S")]
          ic(timestamps)
          timestamp_dict[sub][f"S{sess_num}"][aud_name][f"vid{i+1}"] = timestamps
        e()
  # with open(f'./sample_data/affect_ts.json', 'w') as fp:
  #   json.dump(timestamp_dict, fp, indent=4)

      


