import os
import subprocess
import pandas as pd
from tqdm import tqdm 
from icecream import ic
from sys import exit as e
import glob 
from utils import read_yaml
import json


def build_command(vid_path, dest_split_path):
  ffmpeg = "/usr/bin/ffmpeg"
  # ffmpeg = "ffmpeg"
  commandList = [
    ffmpeg, 
    "-hide_banner", 
    "-loglevel",
    "error",
    "-i", 
    vid_path, 
    "-vf",
    "scale=640:480",
    "-y",
    dest_split_path
  ]
  return commandList

def runFFmpeg(commands):  
  print("running ffmpeg")  
  if subprocess.run(commands).returncode == 0:
    return 1
  else:
    return 0
  
def find_empty_dirs(root_dir='.'):

  for dirpath, dirs, files in os.walk(root_dir):
    if not dirs and not files:
      if len(dirpath.split('/')) > 6:
        yield os.path.join(*(dirpath.split(os.path.sep)[:-1]))
      else:
        yield dirpath


if __name__ == "__main__":

  # READ VARIABLES
  vars = read_yaml()

  # MOUNT DIRECTORY
  root_mnt_dir = vars['paths']['root_mount']
  backup_dir = vars['paths']['cur_backup']
  data_dir = vars['paths']['data']

  # DESTINATION DIRECTORY
  dest_dir = vars['paths']['compressed']
  
  # READ ALL SUBJECTS AND SESSIONS LIST
  vid_dir = os.path.join(root_mnt_dir, backup_dir, data_dir)
  all_files = glob.glob(os.path.join(vid_dir, "*", "*"))
  all_dirs = [os.path.join(*(d.split(os.path.sep)[-2:])) for d in all_files if os.path.isdir(d)]


  all_local_files = glob.glob(os.path.join(dest_dir, "*", "*"))
  empty_local_files = ['/'+d for d in set(list(find_empty_dirs(dest_dir)))]
  empty_local_files = list(set(empty_local_files))

  local_files = list(set(all_local_files) - set(empty_local_files))
  local_dirs = [os.path.join(*(d.split(os.path.sep)[-2:])) for d in local_files if os.path.isdir(d)]
  match_lst = [1 if d in local_dirs else 0 for d in all_dirs]

  # CREATE DICTIONARY
  all_session_dict = {}
  for i in range(len(all_dirs)):
    subject, session = all_dirs[i].split('/')
    if subject not in all_session_dict:
      all_session_dict[subject] = {}
    all_session_dict[subject][session] = 0
    if match_lst[i]:
      all_session_dict[subject][session] = 1

  with open('./data/synced2.json', 'w') as fp:
    json.dump(all_session_dict, fp,  indent=4)


  pbar = tqdm(os.listdir(vid_dir))
  for sub in pbar:
    sub_dir = os.path.join(vid_dir, sub)
    if not os.path.isdir(sub_dir):
      continue
    dest_sub_dir = os.path.join(dest_dir, sub)
    
    for session in os.listdir(sub_dir):
      session_dir = os.path.join(sub_dir, session, "videos")
      if not os.path.isdir(session_dir):
        continue
      dest_session_dir = os.path.join(dest_sub_dir, session)

      if all_session_dict[sub][session]:
        continue

      for video_type in os.listdir(session_dir):
        vid_type_path = os.path.join(session_dir, video_type)
        if not os.path.isdir(vid_type_path):
          continue
        dest_vid_type_dir = os.path.join(dest_session_dir, video_type)
        
        for video in os.listdir(vid_type_path):
          vid_path = os.path.join(vid_type_path, video)
          if os.path.splitext(vid_path)[-1] != ".mp4":
            continue
          
          if not os.path.isdir(dest_sub_dir):
            os.mkdir(dest_sub_dir)
          if not os.path.isdir(dest_session_dir):
            os.mkdir(dest_session_dir)
          if not os.path.isdir(dest_vid_type_dir):
            os.mkdir(dest_vid_type_dir)

          
          vid_name = os.path.splitext(video)[0] + "_compressed.mp4"
          dest_vid_path = os.path.join(dest_vid_type_dir, vid_name)
          pbar.set_description(f"{sub}|{session}|{video_type}|{video}")
          cmd = build_command(vid_path, dest_vid_path)
          stat = runFFmpeg(cmd)
          if stat == 0:
            print(f"Could not run {sub} video")