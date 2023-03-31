import os
import subprocess
import pandas as pd
from tqdm import tqdm 
import shutil
from datetime import datetime
import difflib
from icecream import ic
from sys import exit as e

import sys
sys.path.append('.')
from utils import read_yaml

def make_directories(path):
  if not os.path.isdir(path):
    os.mkdir(path)

if __name__ == "__main__":
  # READ VARIABLES
  vars = read_yaml()
  root_mnt_dir = vars['paths']['root_mount']
  backup_dir = vars['paths']['cur_backup']
  data_dir = vars['paths']['data']
  root_dir = os.path.join(root_mnt_dir, backup_dir, data_dir)

  dest_annot_dir = "/data/dataset/ca_data/touch_data"
  months_dict = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, \
    "July": 7, "August": 8, "September": 9, "October": 10, "Nov": 11, "December": 12}

  desktop_key = "keylogger_desktop"
  phone_key = "keylogger_phone"
  desktop_mouse = "mouselogger_desktop"
  desktop_ca = "CA_output_desktop"
  phone_ca = "CA_output_phone"


  completed_data = []

  # MANUALLY IDENTIFIED DUMMY ANNOTATIONS. ADD MORE AS YOU GO
  # ignore_annot = {
  #   "P0014": {"May_23_2022": "user_data_2022.05.23.11.11.43.5130.txt"}, 
  #   "P0010": {"May_24_2022": "user_data_2022.05.24.17.59.00.2000.txt", "May_12_2022": "user_data_2022.05.12.15.38.09.3700.txt"}, 
  #   "P0015": {"July_20_2022": "user_data_2022.07.20.12.10.01.0560.txt", "May_17_2022": "user_data_2022.05.17.16.39.52.2480.txt"},
  #   "P0029": {"Nov_14_2022": "user_data_2022.11.14.09.55.32.6540.txt"},
  # }

  sub_ctr = 0
  pbar = tqdm(os.listdir(root_dir))
  for sub in pbar:
    sub_dir = os.path.join(root_dir, sub)
    if not os.path.isdir(sub_dir):
      continue
    dest_sub_annot_dir = os.path.join(dest_annot_dir, sub)
    if not os.path.isdir(dest_sub_annot_dir):
      os.mkdir(dest_sub_annot_dir)
    
    sessions = os.listdir(sub_dir)
    session_date_lst = []
    session_date_dict = {}
    session_date_order = {}
    for sess in sessions:
      if "_2022" not in sess and "_2023" not in sess:
        continue 
      dt = sess.split("_")
      month = dt[0]
      day = dt[1]
      year = dt[2]
      
      
      orig_month = difflib.get_close_matches(month, months_dict.keys(), cutoff=0.4)[0]
      session_date_lst.append(f"{year}-{months_dict[orig_month]}-{day}")
      session_date_dict[session_date_lst[-1]] = sess
    
    # Find session number
    session_date_lst.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
    for i in range(len(session_date_lst)):
      act_session = session_date_dict[session_date_lst[i]]
      session_date_order[act_session] = i
    
    for session in os.listdir(sub_dir):
      session_dir = os.path.join(sub_dir, session)
      if "_2022" not in session and "_2023" not in session:
        continue 
      if not os.path.isdir(session_dir):
        continue
      pbar.set_description(session_dir)

      desktop_key_dir = os.path.join(sub_dir, session, desktop_key)
      phone_key_dir = os.path.join(sub_dir, session, phone_key)
      desktop_ca_dir = os.path.join(sub_dir, session, desktop_ca)
      phone_ca_dir = os.path.join(sub_dir, session, phone_ca)
      
      # Add session # while creating destination directory
      order = session_date_order[session]
      dest_session_dir = os.path.join(dest_sub_annot_dir, f"S{order+1}_{session}")
      make_directories(dest_session_dir)


      desktop_key_file = os.path.join(desktop_key_dir, "key_logs.txt")
      dest_desktop_key_dir = os.path.join(dest_session_dir, desktop_key)
      make_directories(dest_desktop_key_dir)
      
      phone_key_file = os.path.join(phone_key_dir, "keylogger.txt")
      dest_phone_key_dir = os.path.join(dest_session_dir, phone_key)
      make_directories(dest_phone_key_dir)
      
      desktop_ff_file = os.path.join(desktop_ca_dir, "task_free_form_pass.docx")
      desktop_essay_file = os.path.join(desktop_ca_dir, "task_essay.docx")
      dest_desktop_ca_dir = os.path.join(dest_session_dir, desktop_ca)
      make_directories(dest_desktop_ca_dir)
      
      phone_ff_file = os.path.join(phone_ca_dir, "ca_home_act_2.txt")
      phone_pwd_file = os.path.join(phone_ca_dir, "ca_home_act_1.txt")
      dest_phone_ca_dir = os.path.join(dest_session_dir, phone_ca)
      make_directories(dest_phone_ca_dir)

      if os.path.isfile(desktop_key_file):
        shutil.copy(desktop_key_file, dest_desktop_key_dir)
      else:
        print(f"{desktop_key_file} is not present")
      
      if os.path.isfile(phone_key_file):
        shutil.copy(phone_key_file, dest_phone_key_dir)
      else:
        print(f"{phone_key_file} is not present")
      
      if os.path.isfile(phone_key_file):
        shutil.copy(desktop_ff_file, dest_desktop_ca_dir)
      else:
        print(f"{phone_key_file} is not present")
      
      if os.path.isfile(desktop_essay_file):
        shutil.copy(desktop_essay_file, dest_desktop_ca_dir)
      else:
        print(f"{desktop_essay_file} is not present")
      
      if os.path.isfile(phone_ff_file):
        shutil.copy(phone_ff_file, dest_phone_ca_dir)
      else:
        print(f"{phone_ff_file} is not present")
      
      if os.path.isfile(phone_pwd_file):
        shutil.copy(phone_pwd_file, dest_phone_ca_dir)
      else:
        print(f"{phone_pwd_file} is not present")





